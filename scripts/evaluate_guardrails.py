"""Print level-A guardrail metrics with numerator, denominator and 95% upper bound (§10.1.7)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import TypeAdapter

from app.guards.evaluation import (
    GuardrailMetrics,
    evaluate_guardrails,
    level_b_gate_failures,
    load_guardrail_dataset,
)
from app.guards.injection import GuardModelResult

ROOT = Path(__file__).parents[1]


def render(metrics: GuardrailMetrics) -> str:
    rows = [
        f"split: {metrics.split}",
        f"classifier_evaluated: {str(metrics.classifier_evaluated).lower()}",
        "metric | k/n | rate | upper 95%",
    ]
    for name in (
        "injection_detection",
        "indirect_containment",
        "summary_rejection",
        "benign_deflect",
        "benign_restrict",
        "output_violation_escape",
        "output_false_block",
    ):
        rate = getattr(metrics, name)
        rows.append(
            f"{name} | {rate.numerator}/{rate.denominator} | {rate.value:.3f} | {rate.upper_95:.3f}"
        )
    rows.append(f"missed attacks: {', '.join(metrics.missed_attacks) or '-'}")
    rows.append(f"escaped outputs: {', '.join(metrics.escaped_outputs) or '-'}")
    rows.append(f"blocked correct outputs: {', '.join(metrics.blocked_correct_outputs) or '-'}")
    rows.append(f"level-B gate failures: {', '.join(level_b_gate_failures(metrics)) or '-'}")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument(
        "--classifier-results",
        type=Path,
        help="External JSON map case_id -> GuardModelResult from a level-B run",
    )
    args = parser.parse_args(argv)
    dataset = load_guardrail_dataset(ROOT / "evals" / "guardrails" / f"{args.split}.yaml")
    classifier_results = None
    if args.classifier_results is not None:
        classifier_results = TypeAdapter(dict[str, GuardModelResult]).validate_python(
            json.loads(args.classifier_results.read_text(encoding="utf-8"))
        )
    print(render(evaluate_guardrails(dataset, classifier_results=classifier_results)))


if __name__ == "__main__":  # pragma: no cover
    main()
