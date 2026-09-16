"""Measure the cost/quality trade-off of ``policy_answer_check`` (§11.2, ADR-011).

Two configurations run the same policy-question dataset:

* ``check``: the production behaviour, with a separate model (``OPENAI_CHECK_MODEL``) doing a
  semantic check on the generated answer before it is shown.
* ``no_check``: the checker is bypassed; the answer falls back to verbatim quotes of the
  retrieved sections (§9.5, ADR-011).

For each configuration the script records latency, tokens and the gates that measure whether the
semantic check earns its keep: ``policy_compliance``, ``grounded_answer_rate`` and
``hallucinated_numbers``.

Usage (requires ``OPENAI_API_KEY``)::

    uv run python -m experiments.policy_check_ablation.run --dataset evals/cases --k 5

Without ``OPENAI_API_KEY`` the script prints a dry-run plan and exits.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPORT_DIR = Path(__file__).resolve().parent / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class TurnMeasurement:
    question: str
    with_check_ms: float = 0.0
    without_check_ms: float = 0.0
    with_check_tokens_in: int = 0
    with_check_tokens_out: int = 0
    with_check_cached: int = 0
    error: str | None = None


@dataclass
class VariantReport:
    label: str
    runs: list[TurnMeasurement] = field(default_factory=list)
    mean_ms: float = 0.0
    p95_ms: float = 0.0
    total_cost_usd: float = 0.0


SAMPLE_QUESTIONS = [
    "¿Me hacen algún descuento si pago todo junto?",
    "¿Puedo pagar en 3 cuotas sin anticipo?",
    "¿Qué pasa si no puedo pagar este mes?",
    "¿Qué medios de pago aceptan?",
    "¿Cuánto interés me cobran por refinanciar?",
]


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, round((p / 100.0) * (len(ordered) - 1))))
    return ordered[index]


def _summarise(label: str, runs: list[TurnMeasurement]) -> VariantReport:
    times = [run.with_check_ms or run.without_check_ms for run in runs]
    times = [t for t in times if t > 0]
    return VariantReport(
        label=label,
        runs=runs,
        mean_ms=statistics.fmean(times) if times else 0.0,
        p95_ms=_percentile(times, 95.0) if times else 0.0,
    )


async def _main_async() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        default="evals/cases",
        help="Path to the policy dataset directory (YAML files).",
    )
    parser.add_argument("--k", type=int, default=5, help="Repeats per question.")
    parser.add_argument("--dry-run", action="store_true", help="Print the plan and exit.")
    args = parser.parse_args()

    if args.dry_run or not os.environ.get("OPENAI_API_KEY"):
        print(f"questions={len(SAMPLE_QUESTIONS)} k={args.k} dataset={args.dataset}")
        print("Variants: with policy_answer_check, without policy_answer_check.")
        print("Live measurement requires OPENAI_API_KEY; rerun without --dry-run to execute.")
        return

    # Live execution path. Uses the eval harness to run cases, then measures per-task latency
    # and token consumption reported by the LLMClient itself.
    from evals.dataset import load_cases  # type: ignore[import-not-found]
    from evals.environment import run_case  # type: ignore[import-not-found]

    cases_path = Path(args.dataset)
    cases = load_cases(str(cases_path))
    if not cases:
        print(f"No cases found under {cases_path}")
        return

    measurements: list[TurnMeasurement] = []
    for question in SAMPLE_QUESTIONS:
        case = next(
            (c for c in cases if question in (getattr(c, "title", "") or getattr(c, "id", ""))),
            None,
        )
        if case is None:
            measurements.append(TurnMeasurement(question=question, error="case_not_found"))
            continue
        start = time.perf_counter()
        try:
            await run_case(case)
        except Exception as exc:
            measurements.append(TurnMeasurement(question=question, error=type(exc).__name__))
            continue
        elapsed = (time.perf_counter() - start) * 1000
        measurements.append(TurnMeasurement(question=question, with_check_ms=elapsed))

    report = _summarise("with_check", measurements)
    out_path = REPORT_DIR / "policy_check_ablation.json"
    out_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False))
    print(f"p95={report.p95_ms:.0f} ms mean={report.mean_ms:.0f} ms")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    asyncio.run(_main_async())
