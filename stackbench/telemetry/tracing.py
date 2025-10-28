"""OpenTelemetry tracing helpers for Stackbench."""

from __future__ import annotations

import logging
import os
import threading
from typing import Optional

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

try:
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
except ModuleNotFoundError:  # pragma: no cover - handled gracefully if exporter missing
    OTLPSpanExporter = None  # type: ignore

logger = logging.getLogger(__name__)

_INITIALIZED = False
_INITIALIZATION_LOCK = threading.Lock()
_DEFAULT_SERVICE_NAME = "stackbench"


def _resolve_otlp_endpoint(override: Optional[str] = None) -> Optional[str]:
    """Resolve the OTLP endpoint from overrides or environment variables."""
    if override:
        return override

    return (
        os.getenv("STACKBENCH_OTLP_ENDPOINT")
        or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
        or "http://localhost:4318/v1/traces"
    )


def initialize_tracer(service_name: str = _DEFAULT_SERVICE_NAME, otlp_endpoint: Optional[str] = None):
    """Initialise the global OpenTelemetry tracer provider."""
    global _INITIALIZED

    if _INITIALIZED:
        return trace.get_tracer(service_name)

    with _INITIALIZATION_LOCK:
        if _INITIALIZED:
            return trace.get_tracer(service_name)

        resource = Resource.create({"service.name": service_name})
        sampler = ParentBased(TraceIdRatioBased(1.0))
        provider = TracerProvider(resource=resource, sampler=sampler)

        # Configure OTLP exporter if available
        if OTLPSpanExporter is not None:
            endpoint = _resolve_otlp_endpoint(otlp_endpoint)
            try:
                otlp_exporter = OTLPSpanExporter(endpoint=endpoint, timeout=5)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            except Exception as exc:  # pragma: no cover - depends on environment configuration
                logger.debug("Skipping OTLP exporter initialisation: %s", exc)
        else:  # pragma: no cover - only hits if dependency missing entirely
            logger.debug("OTLP exporter package not available; skipping remote export")

        # Always add console exporter for local debugging (structured JSON)
        console_exporter = ConsoleSpanExporter()
        provider.add_span_processor(SimpleSpanProcessor(console_exporter))

        trace.set_tracer_provider(provider)
        _INITIALIZED = True

        logger.debug("OpenTelemetry tracer initialised for service '%s'", service_name)

        return trace.get_tracer(service_name)


def ensure_tracing_initialized(service_name: str = _DEFAULT_SERVICE_NAME):
    """Ensure tracing is initialised and return a tracer instance."""
    return initialize_tracer(service_name)


def get_tracer(name: Optional[str] = None):
    """Get a tracer, initialising the provider if required."""
    ensure_tracing_initialized(_DEFAULT_SERVICE_NAME)
    tracer_name = name or __name__
    return trace.get_tracer(tracer_name)


__all__ = ["initialize_tracer", "ensure_tracing_initialized", "get_tracer"]
