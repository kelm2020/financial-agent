from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from evals.judge import CriterionAgreement, JudgeCalibration, calibrate_judge, load_labels
from evals.judge_workflow import load_judge_results


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Per-criterion agreement (TPR, TNR, Cohen's kappa) between humans and judge"
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test", "all"), default="test")
    parser.add_argument("--min-per-class", type=int, default=5)
    return parser.parse_args(argv)


def _number(value: float | None) -> str:
    return "n/d" if value is None else f"{value:.3f}"


def _row(item: CriterionAgreement) -> str:
    return (
        f"| {item.criterion} | {item.samples} | {item.human_pass}/{item.human_fail} | "
        f"{item.agreement:.3f} | {_number(item.tpr)} | {_number(item.tnr)} | "
        f"{_number(item.cohens_kappa)} |"
    )


def render(calibration: JudgeCalibration) -> str:
    lines = [
        f"Judge conversacional · split={calibration.split} · muestras={calibration.samples}",
        "",
        "| Criterio | n | humano pass/fail | acuerdo | TPR | TNR | κ |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *(_row(item) for item in calibration.criteria),
        _row(calibration.overall),
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    calibration = calibrate_judge(
        load_labels(args.dataset),
        load_judge_results(args.results),
        split=args.split,
        min_per_class=args.min_per_class,
    )
    print(render(calibration))


if __name__ == "__main__":  # pragma: no cover
    main()
