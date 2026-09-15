"""§11.5 merge gates that level A decides, and regressions against the recorded baselines."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.guards.evaluation import (
    evaluate_guardrails,
    level_a_gate_failures,
    load_guardrail_dataset,
)
from evals import run as eval_run
from scripts import evaluate_guardrails as guardrails_script

ROOT = Path(__file__).parents[1]


def test_level_a_guardrail_gates_fail_on_escape_false_block_and_regression() -> None:
    metrics = evaluate_guardrails(load_guardrail_dataset(ROOT / "evals/guardrails/test.yaml"))
    assert level_a_gate_failures(metrics, baseline_detection=1.0) == ()
    detection = metrics.injection_detection
    dropped = metrics.model_copy(
        update={"injection_detection": detection.model_copy(update={"numerator": 16})}
    )
    assert level_a_gate_failures(dropped, baseline_detection=1.0) == (
        "injection_detection_regression",
    )
    escaped = metrics.model_copy(
        update={
            "output_violation_escape": metrics.output_violation_escape.model_copy(
                update={"numerator": 1}
            ),
            "output_false_block": metrics.output_false_block.model_copy(update={"numerator": 1}),
        }
    )
    assert level_a_gate_failures(escaped) == ("output_violation_escape", "output_false_block")


def test_guardrail_script_exits_non_zero_on_a_failed_gate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    guardrails_script.main(["--split", "test"])
    assert "merge gate failures: -" in capsys.readouterr().out

    impossible = tmp_path / "baseline.json"
    impossible.write_text(
        json.dumps({"test": {"injection_detection": {"numerator": 2, "denominator": 1}}}),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as regression:
        guardrails_script.main(["--split", "test", "--baseline", str(impossible)])
    assert regression.value.code == 1
    # Without classifier results the level-B gates are unmeasured, never green.
    with pytest.raises(SystemExit):
        guardrails_script.main(["--split", "test", "--require-level-b"])
    assert "classifier_not_evaluated" in capsys.readouterr().out
    assert guardrails_script.baseline_detection(tmp_path / "missing.json", "test") is None
    assert guardrails_script.baseline_detection(impossible, "dev") is None


async def test_tool_selection_regression_below_the_baseline_fails_the_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert eval_run.baseline_f1("level-a", "canonical") == 1.0
    assert eval_run.baseline_f1("live", "canonical") is None
    strict = tmp_path / "baselines.json"
    strict.write_text(json.dumps({"level-a:canonical": {"tool_selection_f1": 2.0}}), "utf-8")
    monkeypatch.setattr(eval_run, "BASELINES_PATH", strict)
    report = await eval_run.evaluate(case_filter="C-01__cuanto_debo")
    assert "tool_selection_f1_regression" in report.metrics.gate_failures
    monkeypatch.setattr(eval_run, "BASELINES_PATH", tmp_path / "missing.json")
    assert eval_run.baseline_f1("level-a", "canonical") is None
