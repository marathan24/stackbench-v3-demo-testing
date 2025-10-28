"""Helpers for loading and inspecting Stackbench run artefacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple

from stackbench.schemas import RunEvent, RunRecord

ISO_Z_SUFFIX = "Z"


def _parse_iso(timestamp: str) -> datetime:
    """Parse an ISO timestamp, handling trailing Z values."""
    if timestamp.endswith(ISO_Z_SUFFIX):
        timestamp = timestamp[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(timestamp)
    except ValueError:
        return datetime.fromtimestamp(0, tz=timezone.utc)


def load_run_summary(run_dir: Path) -> RunRecord:
    """Load the run summary JSON as a RunRecord."""
    summary_path = Path(run_dir) / "run_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Run summary not found at {summary_path}")

    with summary_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    return RunRecord.model_validate(data)


def stream_run_events(run_dir: Path) -> Iterator[RunEvent]:
    """Yield RunEvent entries from the events JSONL log."""
    event_log_path = Path(run_dir) / "events.jsonl"
    if not event_log_path.exists():
        return iter(())

    def _iterator() -> Iterator[RunEvent]:
        with event_log_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                    yield RunEvent.model_validate(payload)
                except json.JSONDecodeError:
                    continue

    return _iterator()


@dataclass
class RunListing:
    """Lightweight view of a run for listing commands."""

    summary: RunRecord
    path: Path

    @property
    def created_at_dt(self) -> datetime:
        return _parse_iso(self.summary.created_at)


class RunStore:
    """Filesystem-backed store for Stackbench run artefacts."""

    def __init__(self, data_dir: Path):
        self.data_dir = Path(data_dir)

    def list_runs(
        self,
        *,
        status: Optional[Iterable[str]] = None,
        since: Optional[datetime] = None,
    ) -> List[RunListing]:
        """Return run listings filtered by status and creation time."""
        statuses = {s for s in status} if status else None
        since_dt = since

        runs: List[RunListing] = []
        if not self.data_dir.exists():
            return runs

        for entry in sorted(self.data_dir.iterdir(), reverse=True):
            if not entry.is_dir():
                continue
            summary_path = entry / "run_summary.json"
            if not summary_path.exists():
                continue

            try:
                summary = load_run_summary(entry)
            except Exception:
                continue

            if statuses and summary.status not in statuses:
                continue

            created_at_dt = _parse_iso(summary.created_at)
            if since_dt and created_at_dt < since_dt:
                continue

            runs.append(RunListing(summary=summary, path=entry))

        return runs

    def get_run(self, run_id: str) -> RunListing:
        """Load a specific run by ID."""
        run_dir = self.data_dir / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found for {run_id}")

        summary = load_run_summary(run_dir)
        return RunListing(summary=summary, path=run_dir)

    def iter_events(self, run_id: str) -> Iterator[RunEvent]:
        """Stream events for a specific run."""
        listing = self.get_run(run_id)
        return stream_run_events(listing.path)

    def get_step_sequence(self, run_id: str) -> List[Tuple[int, str]]:
        """Return (index, step_id) tuples preserving execution order."""
        listing = self.get_run(run_id)
        sequence: List[Tuple[int, str]] = []
        for index, step in enumerate(listing.summary.steps, start=1):
            sequence.append((index, step.step_id))
        return sequence


__all__ = ["RunStore", "RunListing", "load_run_summary", "stream_run_events"]
