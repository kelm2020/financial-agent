from __future__ import annotations

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from app.obs import tracing
from app.obs.tracing import (
    configure_tracing,
    is_tracing_enabled,
    record_attribute,
    shutdown_tracing,
    span,
)


@pytest.fixture(autouse=True)
def _reset_tracing() -> None:
    """Make every test start with an unconfigured tracer."""
    from opentelemetry import trace as _trace

    shutdown_tracing()
    # OpenTelemetry forbids replacing the global TracerProvider once it has been set; tests need
    # a clean slate between configure() calls, so we reset the internal one-shot flag too.
    _trace._TRACER_PROVIDER_SET_ONCE._done = False


def test_configure_without_endpoint_is_a_no_op(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    monkeypatch.delenv("OTEL_ENABLED", raising=False)
    assert configure_tracing() is False
    assert is_tracing_enabled() is False


def test_configure_with_endpoint_registers_exporter(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter)
    assert is_tracing_enabled() is True
    with span("demo", attributes={"k": "v"}):
        pass
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["demo"]
    assert dict(spans[0].attributes or {})["k"] == "v"


def test_span_is_noop_when_tracing_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    configure_tracing()
    with span("demo") as current:
        assert current is None


def test_record_attribute_on_none_is_safe() -> None:
    record_attribute(None, "anything", 42)  # no raise


def test_shutdown_resets_state(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter)
    assert is_tracing_enabled() is True
    shutdown_tracing()
    assert is_tracing_enabled() is False
    # Reconfigure cleanly afterwards.
    configure_tracing(exporter=exporter)
    assert is_tracing_enabled() is True


def test_get_tracer_returns_tracer_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    configure_tracing()
    tracer = tracing.get_tracer()
    assert tracer is not None


def test_configure_tracing_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter)
    assert is_tracing_enabled() is True
    # A second call reports the same enabled state without re-installation.
    assert configure_tracing(exporter=exporter) is True
    assert is_tracing_enabled() is True


def test_configure_tracing_uses_otlp_endpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318")
    monkeypatch.setenv("OTEL_ENABLED", "true")
    try:
        assert configure_tracing() is True
        assert is_tracing_enabled() is True
    finally:
        shutdown_tracing()
        monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
        monkeypatch.delenv("OTEL_ENABLED", raising=False)


def test_env_flag_truthy_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_TEST", "TRUE")
    assert tracing._env_flag("OTEL_TEST") is True
    monkeypatch.setenv("OTEL_TEST", "yes")
    assert tracing._env_flag("OTEL_TEST") is True
    monkeypatch.setenv("OTEL_TEST", "0")
    assert tracing._env_flag("OTEL_TEST") is False
    monkeypatch.setenv("OTEL_TEST", "no")
    assert tracing._env_flag("OTEL_TEST") is False
    assert tracing._env_flag("OTEL_MISSING") is False
    assert tracing._env_flag("OTEL_MISSING", default=True) is True


def test_record_attribute_attaches_to_span() -> None:
    exporter = InMemorySpanExporter()
    configure_tracing(exporter=exporter)
    with span("rec") as current:
        assert current is not None
        record_attribute(current, "answer.id", 42)
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert dict(spans[0].attributes or {})["answer.id"] == 42
