"""Structured run logging utilities for Stackbench."""

from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from stackbench.schemas import RunEvent, RunRecord, RunStep, RunTotals

ISO_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utc_now() -> datetime:
    """Return a timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def _isoformat(dt: datetime) -> str:
    """Convert a datetime to an ISO 8601 string with a trailing Z."""
    return dt.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")


class RunLogger:
    """Append-only JSONL run logger that tracks events, steps, and aggregates."""

    def __init__(self, run_id: str, run_dir: Path, metadata: Optional[Dict[str, Any]] = None):
        self.run_id = run_id
        self._run_dir = Path(run_dir)
        self._event_path = self._run_dir / "events.jsonl"
        self._summary_path = self._run_dir / "run_summary.json"
        self._base_metadata = metadata or {}

        self._lock = asyncio.Lock()
        self._initialized = False
        self._run_summary: Optional[RunRecord] = None
        self._step_index: Dict[str, int] = {}
        self._step_start_times: Dict[str, datetime] = {}
        self._prompt_cache: Dict[str, Any] = {}
        self._start_time: Optional[datetime] = None

    @property
    def run_dir(self) -> Path:
        """Return the root directory for this run."""
        return self._run_dir

    @property
    def event_log_path(self) -> Path:
        """Return the path to the JSONL event log."""
        return self._event_path

    @property
    def summary_path(self) -> Path:
        """Return the path to the compact run summary JSON."""
        return self._summary_path

    async def start_run(self, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Initialise run logging and emit a run start event."""
        additional_metadata = metadata or {}
        combined_metadata = {**self._base_metadata, **additional_metadata}

        async with self._lock:
            if not self._initialized:
                self._run_dir.mkdir(parents=True, exist_ok=True)
                if not self._event_path.exists():
                    self._event_path.touch()

                self._start_time = _utc_now()
                created_at = _isoformat(self._start_time)

                self._run_summary = RunRecord(
                    run_id=self.run_id,
                    status="running",
                    repo_url=self._base_metadata.get("repo_url"),
                    branch=self._base_metadata.get("branch"),
                    docs_path=self._base_metadata.get("docs_path"),
                    library=self._base_metadata.get("library"),
                    library_version=self._base_metadata.get("library_version"),
                    created_at=created_at,
                    totals=RunTotals(),
                    metadata=combined_metadata,
                    steps=[],
                    total_steps=0,
                    total_events=0,
                )

                self._initialized = True

                start_event = RunEvent(
                    event_id=uuid.uuid4().hex,
                    run_id=self.run_id,
                    timestamp=created_at,
                    type="info",
                    name="run_start",
                    metadata={"message": "Run started"},
                )
                self._record_event_locked(start_event)
                self._write_summary_locked()
            else:
                # Merge in additional metadata for subsequent initialisation calls
                if self._run_summary:
                    self._run_summary.metadata.update(additional_metadata)
                    self._write_summary_locked()

    async def complete_run(self, status: str, error: Optional[str] = None) -> None:
        """Mark the run as complete and emit a completion event."""
        self._ensure_initialized()
        completed_at_dt = _utc_now()
        completed_at = _isoformat(completed_at_dt)

        async with self._lock:
            if not self._run_summary:
                return

            self._run_summary.status = status
            self._run_summary.completed_at = completed_at
            if self._start_time:
                duration_ms = int((completed_at_dt - self._start_time).total_seconds() * 1000)
                self._run_summary.duration_ms = duration_ms

            if error:
                self._run_summary.metadata["error"] = error
                self._run_summary.totals.errors += 1

            completion_event = RunEvent(
                event_id=uuid.uuid4().hex,
                run_id=self.run_id,
                timestamp=completed_at,
                type="error" if error else "info",
                name="run_complete",
                error=error,
                metadata={"status": status},
            )
            self._record_event_locked(completion_event)
            self._write_summary_locked()

    async def log_info(self, message: str, metadata: Optional[Dict[str, Any]] = None, step_id: Optional[str] = None) -> None:
        """Log a generic informational event."""
        self._ensure_initialized()
        event = RunEvent(
            event_id=uuid.uuid4().hex,
            run_id=self.run_id,
            step_id=step_id,
            timestamp=_isoformat(_utc_now()),
            type="info",
            name="info",
            metadata={"message": message, **(metadata or {})},
        )
        async with self._lock:
            self._record_event_locked(event)
            self._write_summary_locked()

    async def log_error(
        self,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
        step_id: Optional[str] = None,
    ) -> None:
        """Log an error event and increment error counters."""
        self._ensure_initialized()
        event = RunEvent(
            event_id=uuid.uuid4().hex,
            run_id=self.run_id,
            step_id=step_id,
            timestamp=_isoformat(_utc_now()),
            type="error",
            name="error",
            error=message,
            metadata=metadata or {},
        )
        async with self._lock:
            self._record_event_locked(event)
            self._apply_metrics_locked(step_id, error=True)
            self._write_summary_locked()

    async def start_step(
        self,
        name: str,
        stage: str,
        document: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Start a new step and emit a step_start event."""
        self._ensure_initialized()
        started_dt = _utc_now()
        started_at = _isoformat(started_dt)
        step_id = uuid.uuid4().hex

        step = RunStep(
            step_id=step_id,
            name=name,
            stage=stage,
            document=document,
            status="running",
            started_at=started_at,
            metadata=metadata or {},
            completed_at=None,
            latency_ms=None,
            tokens_input=0,
            tokens_output=0,
            cost=0.0,
            tool_calls=0,
            errors=0,
            error=None,
        )

        async with self._lock:
            if not self._run_summary:
                return step_id

            self._run_summary.steps.append(step)
            self._run_summary.total_steps = len(self._run_summary.steps)
            self._step_index[step_id] = self._run_summary.total_steps - 1
            self._step_start_times[step_id] = started_dt

            event = RunEvent(
                event_id=uuid.uuid4().hex,
                run_id=self.run_id,
                step_id=step_id,
                timestamp=started_at,
                type="step_start",
                name=name,
                metadata={"stage": stage, **(metadata or {})},
            )
            self._record_event_locked(event)
            self._write_summary_locked()

        return step_id

    async def end_step(
        self,
        step_id: str,
        status: str,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Finish a step and emit a step_end event."""
        self._ensure_initialized()
        completed_dt = _utc_now()
        completed_at = _isoformat(completed_dt)

        async with self._lock:
            step = self._get_step_locked(step_id)
            if not step:
                return

            step.status = status
            step.completed_at = completed_at

            started_dt = self._step_start_times.get(step_id)
            if started_dt:
                latency_ms = int((completed_dt - started_dt).total_seconds() * 1000)
                step.latency_ms = latency_ms
                self._run_summary.totals.latency_ms += latency_ms

            if error:
                step.error = error
                step.errors += 1
                self._run_summary.totals.errors += 1

            event = RunEvent(
                event_id=uuid.uuid4().hex,
                run_id=self.run_id,
                step_id=step_id,
                timestamp=completed_at,
                type="step_end",
                name=step.name,
                error=error,
                metadata={**(metadata or {}), "status": status},
            )
            self._record_event_locked(event)
            self._write_summary_locked()

    async def log_prompt_start(
        self,
        step_id: str,
        name: str,
        prompt: str,
        variables: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Log the beginning of a model prompt and return its cache key."""
        self._ensure_initialized()
        variables = variables or {}
        metadata = metadata or {}

        cache_payload = prompt + json.dumps(variables, sort_keys=True, default=str)
        cache_key = hashlib.sha256(cache_payload.encode("utf-8")).hexdigest()

        event = RunEvent(
            event_id=uuid.uuid4().hex,
            run_id=self.run_id,
            step_id=step_id,
            timestamp=_isoformat(_utc_now()),
            type="prompt",
            name=name,
            role="user",
            prompt=prompt,
            variables=variables,
            model=model,
            cache_key=cache_key,
            metadata={**metadata, "direction": "request"},
        )

        async with self._lock:
            self._record_event_locked(event)
            self._write_summary_locked()

        return cache_key

    async def log_prompt_completion(
        self,
        step_id: str,
        name: str,
        cache_key: str,
        output: Any,
        *,
        model: Optional[str] = None,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        cost: Optional[float] = None,
        latency_ms: Optional[int] = None,
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log the completion of a model prompt with its response."""
        self._ensure_initialized()
        metadata = metadata or {}

        event = RunEvent(
            event_id=uuid.uuid4().hex,
            run_id=self.run_id,
            step_id=step_id,
            timestamp=_isoformat(_utc_now()),
            type="prompt",
            name=name,
            role="assistant",
            output=output,
            model=model,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            cost=cost,
            latency_ms=latency_ms,
            error=error,
            cache_key=cache_key,
            metadata={**metadata, "direction": "response"},
        )

        async with self._lock:
            self._record_event_locked(event)
            self._apply_metrics_locked(
                step_id,
                tokens_input=tokens_input,
                tokens_output=tokens_output,
                cost=cost,
                latency_ms=latency_ms,
                error=error is not None,
            )
            self._prompt_cache[cache_key] = {
                "output": output,
                "model": model,
                "tokens_input": tokens_input,
                "tokens_output": tokens_output,
                "cost": cost,
                "latency_ms": latency_ms,
                "error": error,
            }
            self._write_summary_locked()

    def _apply_metrics_locked(
        self,
        step_id: Optional[str],
        *,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        cost: Optional[float] = None,
        latency_ms: Optional[int] = None,
        tool_calls: Optional[int] = None,
        error: bool = False,
    ) -> None:
        """Apply metric deltas to the run and step aggregates (lock must be held)."""
        if not self._run_summary:
            return

        step = self._get_step_locked(step_id) if step_id else None

        if tokens_input:
            self._run_summary.totals.tokens_input += tokens_input
            if step:
                step.tokens_input += tokens_input

        if tokens_output:
            self._run_summary.totals.tokens_output += tokens_output
            if step:
                step.tokens_output += tokens_output

        if cost:
            self._run_summary.totals.cost += cost
            if step:
                step.cost += cost

        if latency_ms:
            self._run_summary.totals.latency_ms += latency_ms
            if step and step.latency_ms is not None:
                # latency_ms for step_end already captured; no update here
                pass

        if tool_calls:
            self._run_summary.totals.tool_calls += tool_calls
            if step:
                step.tool_calls += tool_calls

        if error:
            self._run_summary.totals.errors += 1
            if step:
                step.errors += 1
                if step.error is None:
                    step.error = "Error recorded"

    def _get_step_locked(self, step_id: Optional[str]) -> Optional[RunStep]:
        """Return a step model while the lock is held."""
        if not step_id or not self._run_summary:
            return None
        index = self._step_index.get(step_id)
        if index is None:
            return None
        try:
            return self._run_summary.steps[index]
        except IndexError:
            return None

    def _record_event_locked(self, event: RunEvent) -> None:
        """Append an event to the JSONL log and update counters (lock must be held)."""
        with self._event_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json")) + "\n")
        if self._run_summary:
            self._run_summary.total_events += 1

    def _write_summary_locked(self) -> None:
        """Persist the current run summary to disk (lock must be held)."""
        if not self._run_summary:
            return
        data = self._run_summary.model_dump(mode="json")
        with self._summary_path.open("w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)

    def _ensure_initialized(self) -> None:
        if not self._initialized or not self._run_summary:
            raise RuntimeError("RunLogger.start_run must be called before logging events")

    def get_cached_prompt_output(self, cache_key: str) -> Optional[Any]:
        """Return cached prompt output for replay tooling."""
        return self._prompt_cache.get(cache_key)


__all__ = ["RunLogger"]
