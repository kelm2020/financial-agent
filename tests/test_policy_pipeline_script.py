"""Policy pipeline script: labeled sections count, and a failed measurement is never a negative."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
import yaml  # type: ignore[import-untyped]

from evals.judge import CriterionVerdict, JudgeVerdicts
from scripts import evaluate_policy_pipeline as pipeline

_TRANSFER: dict[str, Any] = {
    "id": "R-04",
    "query": "¿puedo pagar con transferencia y cuánto tarda?",
    "source": "rag",
    "answerable": True,
    "sections": ["PAY-MET-002"],
    "required": "Transferencia: 24 horas hábiles.",
}
_RATE: dict[str, Any] = {
    "id": "N-01",
    "query": "¿cuál es la tasa nominal anual que aplican?",
    "source": "rag",
    "answerable": False,
    "sections": [],
    "required": "Abstenerse: la documentación no responde.",
}


def _args(tmp_path: Path, cases: list[dict[str, Any]], *extra: str) -> Any:
    dataset = tmp_path / "cases.yaml"
    dataset.write_text(
        yaml.safe_dump({"effective_on": "2026-09-12", "cases": cases}, allow_unicode=True),
        encoding="utf-8",
    )
    return pipeline.parse_args(
        [
            "--dataset",
            str(dataset),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--output",
            str(tmp_path / "report.json"),
            *extra,
        ]
    )


def _verdicts(result: Literal["pass", "fail"]) -> JudgeVerdicts:
    item = CriterionVerdict(verdict=result, reason="checked")
    return JudgeVerdicts(
        responde_lo_pedido=item,
        proximo_paso=item,
        tono_adecuado=item,
        claridad=item,
        reconoce_vulnerabilidad=item,
    )


class _FakeLLM:
    def __init__(self, **kwargs: Any) -> None:
        del kwargs

    async def aclose(self) -> None:
        return None


def _fake_session(outcomes: dict[str, dict[str, Any]]) -> Any:
    @asynccontextmanager
    async def session(
        customer_id: str, *, llm: Any = None, retriever: Any = None
    ) -> AsyncIterator[Any]:
        del customer_id, llm, retriever
        recorder = SimpleNamespace(events=[], tool_calls=[], agreement_writes=[])

        async def send(text: str) -> Any:
            outcome = outcomes[text]
            recorder.events.extend(outcome["events"])
            recorder.tool_calls.extend(SimpleNamespace(name=name) for name in outcome["tools"])
            return SimpleNamespace(
                text=outcome["text"],
                state={"selected_source": outcome["source"]},
                llm_tasks=("guard_classifier", "grounded_response", "policy_answer_check"),
            )

        yield SimpleNamespace(send=send, recorder=recorder)

    return session


def test_score_row_requires_route_safety_and_a_labeled_section() -> None:
    answer = {
        "source": "rag",
        "answerable": True,
        "tools": ["search_policies"],
        "writes": 0,
        "text": "Por transferencia, 24 horas hábiles. [PAY-MET-002]",
    }
    assert pipeline.score_row(_TRANSFER, answer)
    assert not pipeline.score_row(_TRANSFER, {**answer, "text": "Sí. [FAQ-010]"})
    assert not pipeline.score_row(_TRANSFER, {**answer, "writes": 1})
    assert not pipeline.score_row(_TRANSFER, {**answer, "source": "backend"})
    escalation = {"source": "escalation", "answerable": False, "sections": []}
    derived = {"source": "escalation", "answerable": False, "tools": ["request_human"], "writes": 0}
    assert pipeline.score_row(escalation, {**derived, "text": ""})
    assert not pipeline.score_row(
        escalation, {**derived, "text": "", "tools": ["search_policies", "request_human"]}
    )
    assert pipeline.answered([{"type": "policy_answer", "outcome": "extract"}])
    assert not pipeline.answered([{"type": "policy_answer", "outcome": "abstained"}])


async def test_offline_run_scores_sections_and_reports_failed_retrieval_as_unavailable(
    tmp_path: Path,
) -> None:
    uncached = {
        "id": "X-01",
        "query": "¿hay quita zqx para pagos sin caché de embeddings?",
        "source": "rag",
        "answerable": True,
        "sections": ["POL-NEG-003"],
        "required": "",
    }
    report = await pipeline.run(_args(tmp_path, [_TRANSFER, _RATE, uncached]))
    rows = {row["id"]: row for row in report["cases"]}
    # The calibrated extract answers from FAQ-010 (a relative paying), not PAY-MET-002.
    assert rows["R-04"]["answerable"] and not rows["R-04"]["passed"]
    assert rows["N-01"]["passed"] and not rows["N-01"]["semantic_checked"]
    assert rows["X-01"] | {"available": False, "error": "retriever_unavailable"} == rows["X-01"]
    assert report["metrics"]["unavailable"] == 1 and report["mode"] == "offline"
    saved = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    assert (saved["passed"], saved["total"]) == (1, 3)


async def test_live_run_judges_answers_and_never_counts_a_failure_as_a_verdict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    answer = {
        "events": [{"type": "policy_answer", "outcome": "model_answer"}],
        "tools": ["search_policies"],
        "text": "Por transferencia, 24 horas hábiles. [PAY-MET-002]",
        "source": "rag",
    }
    outage = {
        **answer,
        "events": [
            {"type": "response_model_unavailable"},
            {"type": "policy_answer", "outcome": "extract"},
        ],
    }
    broken = {**_TRANSFER, "id": "BROKEN", "query": "q-judge-down"}
    down = {**_TRANSFER, "id": "OUTAGE", "query": "q-model-down"}
    outcomes = {_TRANSFER["query"]: answer, "q-judge-down": answer, "q-model-down": outage}

    async def judge(llm: Any, sample: Any) -> JudgeVerdicts:
        del llm
        if sample.id == "BROKEN":
            raise RuntimeError("judge down")
        assert sample.situation == pipeline._SITUATION  # never the labeled expectation
        return _verdicts("pass")

    monkeypatch.setattr(pipeline, "OpenAIResponsesLLM", _FakeLLM)
    monkeypatch.setattr(pipeline, "agent_session", _fake_session(outcomes))
    monkeypatch.setattr(pipeline, "judge_response", judge)
    report = await pipeline.run(_args(tmp_path, [_TRANSFER, broken, down], "--live"))
    rows = {row["id"]: row for row in report["cases"]}
    assert rows["R-04"]["passed"] and rows["R-04"]["semantic_checked"]
    assert (rows["BROKEN"]["available"], rows["BROKEN"]["error"]) == (False, "RuntimeError")
    assert rows["OUTAGE"]["error"] == "response_model_unavailable"
    assert report["mode"] == "live" and report["metrics"]["unavailable"] == 2


async def test_live_run_requires_a_key_and_a_judge_distinct_from_the_agent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        await pipeline.run(_args(tmp_path, [_RATE], "--live"))
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_AGENT_MODEL", "gpt-5-nano")
    with pytest.raises(ValueError, match="different model"):
        await pipeline.run(_args(tmp_path, [_RATE], "--live", "--judge-model", "gpt-5-nano"))


async def test_reranked_run_refuses_to_report_without_a_reranker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report = await pipeline.run(_args(tmp_path, [_RATE], "--reranker"))
    assert report["reranker"] is True and report["total"] == 1

    @asynccontextmanager
    async def missing(*args: Any, **kwargs: Any) -> AsyncIterator[None]:
        del args, kwargs
        yield None

    monkeypatch.setattr(pipeline, "reranker_client", missing)
    with pytest.raises(RuntimeError, match="Reranker unavailable"):
        await pipeline.run(_args(tmp_path, [_RATE], "--reranker"))


@pytest.mark.parametrize(("passed", "code"), [(1, 0), (0, 1)])
def test_main_exit_status_reflects_every_row(
    passed: int, code: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_run(args: Any) -> dict[str, Any]:
        del args
        return {"passed": passed, "total": 1, "cases": []}

    monkeypatch.setattr(pipeline, "run", fake_run)
    with pytest.raises(SystemExit) as exc:
        pipeline.main([])
    assert exc.value.code == code
