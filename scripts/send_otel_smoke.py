"""Send a synthetic OpenTelemetry span to Langfuse (or any OTLP/HTTP collector).

Useful as a smoke test after configuring ``OTEL_EXPORTER_OTLP_ENDPOINT`` and
``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY``. Without those env vars the script prints
the plan and exits.

Usage::

    OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:3000 \\
    LANGFUSE_PUBLIC_KEY=pk-lf-... LANGFUSE_SECRET_KEY=sk-lf-... \\
        uv run python -m scripts.send_otel_smoke
"""
from __future__ import annotations

import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from app.obs.tracing import configure_tracing, get_tracer  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", default="smoke.trace", help="Span name")
    parser.add_argument("--message", default="hello from the agent runtime")
    args = parser.parse_args()

    enabled = configure_tracing()
    if not enabled:
        print(
            "tracing not enabled: set OTEL_EXPORTER_OTLP_ENDPOINT "
            "and LANGFUSE_PUBLIC_KEY/SECRET_KEY"
        )
        return
    tracer = get_tracer()
    with tracer.start_as_current_span(args.name) as span:
        span.set_attribute("smoke.message", args.message)
        span.set_attribute("smoke.runtime", "financial-agent")
    print(f"sent span name={args.name!r}")
    # Flush so the BatchSpanProcessor exports before we exit.
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    provider = trace.get_tracer_provider()
    assert isinstance(provider, TracerProvider)
    provider.force_flush()


if __name__ == "__main__":
    main()
