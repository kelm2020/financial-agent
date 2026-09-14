"""Answerability split and runner, offline: routing plus the model-free policy extract."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest

from evals.answerability import (
    AnswerabilityDataset,
    classify,
    evaluate_answerability,
    load_answerability_dataset,
)
from scripts import evaluate_answerability as answerability_script
from tests.agent_support import StaticRetriever, corpus_chunk


def _dataset() -> AnswerabilityDataset:
    return AnswerabilityDataset.model_validate(
        {
            "split": "test",
            "effective_on": date(2026, 9, 12),
            "customer_id": "CUST-00125",
            "positive": [
                {
                    "id": "P-1",
                    "query": "¿puedo pagar con tarjeta?",
                    "expected_section_ids": ["PAY-MET-001"],
                },
                {
                    "id": "P-2",
                    "query": "¿puedo pagar con tarjeta?",
                    "expected_section_ids": ["FAQ-009"],
                },
                {
                    "id": "P-3",
                    "query": "¿tienen app para el celular?",
                    "expected_section_ids": ["PAY-MET-001"],
                },
            ],
            "negative": [
                {"id": "N-1", "query": "¿tienen app para el celular?"},
                {"id": "N-2", "query": "¿puedo pagar con tarjeta?"},
            ],
        }
    )


def test_answerability_split_covers_every_section_and_unique_ids(tmp_path: Path) -> None:
    dataset = load_answerability_dataset()
    assert (len(dataset.positive), len(dataset.negative)) == (35, 15)
    first = {case.expected_section_ids[0] for case in dataset.positive}
    assert len(first) == 35
    duplicated = tmp_path / "dup.yaml"
    duplicated.write_text(
        "split: test\neffective_on: 2026-09-12\ncustomer_id: CUST-00125\n"
        "positive: [{id: X, query: a, expected_section_ids: [FAQ-001]}]\n"
        "negative: [{id: X, query: b}]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="únicos"):
        load_answerability_dataset(duplicated)


def test_classify_outcomes() -> None:
    assert classify("Sin cita.", ("FAQ-001",)) == ((), "abstained")
    assert classify("Texto [faq-001] [FAQ-001]", ("FAQ-001",)) == (("FAQ-001",), "correct")
    assert classify("Texto [ESC-004]", ("FAQ-001",)) == (("ESC-004",), "wrong_section")
    assert classify("Texto [ESC-004]", ()) == (("ESC-004",), "answered")


async def test_evaluate_answerability_through_the_agent() -> None:
    retriever = StaticRetriever([corpus_chunk("PAY-MET-001")])
    report = await evaluate_answerability(_dataset(), llm=None, retriever=retriever)
    outcomes = {result.id: (result.outcome, result.passed) for result in report.results}
    assert outcomes == {
        "P-1": ("correct", True),
        "P-2": ("wrong_section", False),
        "P-3": ("abstained", False),
        "N-1": ("abstained", True),
        "N-2": ("answered", False),
    }
    assert report.results[0].intent == "consulta_general"
    assert report.results[0].retrieved == ("PAY-MET-001",)
    assert (report.answered_correctly, report.positive_cases) == (1, 3)
    assert (report.abstained_negatives, report.negative_cases, report.wrong_citations) == (1, 2, 2)
    assert [result.id for result in report.failures] == ["P-2", "P-3", "N-2"]


async def test_answerability_script_renders_the_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    async def dependencies(*_: Any) -> tuple[None, StaticRetriever]:
        return None, StaticRetriever([corpus_chunk("PAY-MET-001")])

    monkeypatch.setattr(answerability_script, "_live_dependencies", dependencies)
    monkeypatch.setattr(answerability_script, "load_answerability_dataset", _dataset)
    report = await answerability_script.run()
    output = capsys.readouterr().out
    assert "| 1/3 | 1/2 | 2 |" in output
    assert "FAIL P-3 [positive] intent=ambiguo outcome=abstained cited=[]" in output
    assert report.positive_cases == 3


def test_answerability_script_main_runs_the_measurement(monkeypatch: pytest.MonkeyPatch) -> None:
    called: list[bool] = []

    async def fake_run() -> None:
        called.append(True)

    monkeypatch.setattr(answerability_script, "run", fake_run)
    answerability_script.main()
    assert called == [True]
