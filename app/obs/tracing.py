"""OpenTelemetry tracing bootstrap for the agent runtime.

The tracer is a no-op unless :func:`configure_tracing` is called with a real exporter. This
keeps the runtime safe to import in tests and on machines without a collector, while letting
operators turn spans on with environment variables in production.

Spans follow the GenAI semantic conventions: ``gen_ai.*`` for LLM inferences and
``execute_tool`` for backend calls. The conversation and turn ids flow as attributes so a
single trace ties every node, tool call, retrieval and LLM call of one turn together.

The OTel exporter URL is computed from two environment variables:

* ``OTEL_EXPORTER_OTLP_ENDPOINT``: the host URL. For Langfuse self-hosted (v3+) this is the same
  host as the Langfuse UI (``http://localhost:3000``); the exporter appends
  ``/api/public/otel/v1/traces`` automatically.
* ``OTEL_EXPORTER_OTLP_HEADERS`` (optional): extra headers, comma-separated ``key=value`` pairs.
  For Langfuse this carries the ``Authorization=Basic <pk:sk>`` credential.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

_LOGGER = logging.getLogger("app.obs.tracing")

_LOCK = threading.Lock()
_CONFIGURED = False
_EXPORTER: SpanExporter | None = None


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_endpoint() -> str | None:
    endpoint = os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT")
    return endpoint.strip() if endpoint and endpoint.strip() else None


def _parse_headers(raw: str | None) -> dict[str, str]:
    """Parse ``OTEL_EXPORTER_OTLP_HEADERS`` (``k1=v1,k2=v2``) into a dict."""
    if not raw:
        return {}
    result: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        result[key.strip()] = value.strip()
    return result


def _resolve_otlp_exporter() -> SpanExporter | None:
    """Build an OTLPSpanExporter from env vars, with Langfuse v3+ path and auth."""
    endpoint = _resolve_endpoint()
    if endpoint is None or not _env_flag("OTEL_ENABLED", True):
        return None
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    base = endpoint.rstrip("/")
    # Langfuse v3+ exposes an OTLP-compatible endpoint under /api/public/otel/v1/traces.
    # A bare host (no path) gets the Langfuse default; a host that already carries /v1/traces
    # is treated as a generic collector and left untouched.
    if base.endswith("/v1/traces"):
        url = base
    else:
        url = f"{base}/api/public/otel/v1/traces"
    headers = _parse_headers(os.environ.get("OTEL_EXPORTER_OTLP_HEADERS"))
    # Convenience: if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY are set, build the header.
    pk = os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = os.environ.get("LANGFUSE_SECRET_KEY")
    if pk and sk and "Authorization" not in headers:
        token = base64.b64encode(f"{pk}:{sk}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    return OTLPSpanExporter(endpoint=url, headers=headers or None)


def configure_tracing(
    *,
    service_name: str = "financial-agent",
    service_version: str = "0.1.0",
    exporter: SpanExporter | None = None,
) -> bool:
    """Configure a global TracerProvider. Returns True if a provider was installed.

    When ``exporter`` is None the function honours ``OTEL_EXPORTER_OTLP_ENDPOINT`` and
    ``OTEL_ENABLED``/``OTEL_EXPORTER_OTLP_ENABLED``. Without either, the global tracer stays
    a no-op and ``configure_tracing`` returns False.
    """
    global _CONFIGURED, _EXPORTER
    with _LOCK:
        if _CONFIGURED:
            return _EXPORTER is not None
        resource = Resource.create(
            {
                "service.name": service_name,
                "service.version": service_version,
            }
        )
        provider = TracerProvider(resource=resource)
        chosen: SpanExporter | None = exporter
        if chosen is None:
            chosen = _resolve_otlp_exporter()
        if chosen is not None:
            provider.add_span_processor(BatchSpanProcessor(chosen))
            _EXPORTER = chosen
            _CONFIGURED = True
            trace.set_tracer_provider(provider)
            resolved = _resolve_endpoint()
            endpoint_repr = (
                "custom"
                if exporter
                else (
                    f"{resolved}/api/public/otel/v1/traces"
                    if resolved is not None
                    else "in-memory"
                )
            )
            _LOGGER.info("tracing configured endpoint=%s", endpoint_repr)
            return True
        _CONFIGURED = True
        return False


def get_tracer(name: str = "app.obs.tracing") -> trace.Tracer:
    """Return the configured tracer. Falls back to a no-op when tracing is off."""
    if not _CONFIGURED:
        configure_tracing()
    return trace.get_tracer(name)


def is_tracing_enabled() -> bool:
    """Whether the global provider actually exports spans."""
    return _EXPORTER is not None


@contextmanager
def span(
    name: str,
    *,
    attributes: dict[str, Any] | None = None,
) -> Iterator[trace.Span | None]:
    """Open a span when tracing is enabled, otherwise yield None.

    Usage::

        with span("execute_tool", attributes={"tool.name": name, "tool.status": status}):
            ...
    """
    tracer = get_tracer()
    if not is_tracing_enabled():
        yield None
        return
    with tracer.start_as_current_span(name, attributes=attributes or {}) as current:
        yield current


def record_attribute(span_obj: trace.Span | None, key: str, value: Any) -> None:
    """Attach an attribute to an open span. No-op when tracing is off or the span is None."""
    if span_obj is None:
        return
    span_obj.set_attribute(key, value)


def shutdown_tracing() -> None:
    """Flush and shut down the global provider. Safe to call multiple times."""
    global _CONFIGURED, _EXPORTER
    with _LOCK:
        if not _CONFIGURED:
            return
        provider = trace.get_tracer_provider()
        if isinstance(provider, TracerProvider):
            provider.force_flush()
            provider.shutdown()
        _CONFIGURED = False
        _EXPORTER = None
