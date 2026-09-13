"""Print level-A guardrail metrics with numerator, denominator and 95% upper bound (§10.1.7)."""

from __future__ import annotations

import argparse
from pathlib import Path

from app.guards.evaluation import GuardrailMetrics, evaluate_guardrails, load_guardrail_dataset

ROOT = Path(__file__).parents[1]


def render(metrics: GuardrailMetrics) -> str:
    rows = [f"split: {metrics.split}", "metric | k/n | rate | upper 95%"]
    for name in (
        "injection_detection",
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
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    args = parser.parse_args(argv)
    dataset = load_guardrail_dataset(ROOT / "evals" / "guardrails" / f"{args.split}.yaml")
    print(render(evaluate_guardrails(dataset)))


if __name__ == "__main__":  # pragma: no cover
    main()
