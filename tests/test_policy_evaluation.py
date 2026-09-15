from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.rag.evaluation import evaluate_retriever, load_retrieval_dataset
from app.rag.factory import build_retriever
from app.rag.ingest import ingest_corpus
from app.rag.store import InMemoryHybridStore
from scripts import evaluate_answerability, evaluate_retrieval
from scripts.evaluate_policy_pipeline import confusion
from tests.rag_support import cached_embeddings, offline_settings


def test_contaminated_policy_sets_are_reported_as_regression_not_held_out() -> None:
    # Terms of their phrasings entered the routing ontology after they were written (§11.3).
    for path in (Path("evals/policy_regression.yaml"), Path("evals/policy_challenge.yaml")):
        assert "NO es held-out" in path.read_text(encoding="utf-8")
    assert not Path("evals/policy_holdout.yaml").exists()


def test_confusion_excludes_unavailable_and_reports_both_error_types() -> None:
    rows = [
        {"expected": expected, "answerable": actual}
        for expected, actual in [(True, True), (True, False), (False, True), (False, False)]
    ]
    rows.append({"expected": True, "answerable": False, "available": False})
    result = confusion(rows)
    assert (result["tp"], result["fn"], result["fp"], result["tn"]) == (1, 1, 1, 1)
    assert result["precision"] == result["recall"] == result["f1"] == 0.5
    assert result["unavailable"] == 1


@pytest.mark.parametrize("script", [evaluate_answerability, evaluate_retrieval])
def test_eval_cli_returns_failure_status(script: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    from types import SimpleNamespace

    async def failure(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(failures=("failed criterion",))

    monkeypatch.setattr(script, "run", failure)
    with pytest.raises(SystemExit) as exc:
        script.main([]) if script is evaluate_retrieval else script.main()
    assert exc.value.code == 1


async def test_candidate_ranking_has_no_abstention_metric() -> None:
    store = InMemoryHybridStore()
    embeddings = cached_embeddings()
    dataset = load_retrieval_dataset("test")
    await ingest_corpus(store, embeddings, effective_on=dataset.effective_on)
    metrics = await evaluate_retriever(
        build_retriever(store, embeddings, offline_settings()), dataset, before_gate=True
    )
    assert metrics.negative_cases == 0 and metrics.abstentions == 0
    assert len(metrics.cases) == len(dataset.positive)
    assert all(case.status == "ok" for case in metrics.cases)
    assert metrics.recall_at_3 >= 0.8
    table = evaluate_retrieval.render(metrics, "memory", "no")
    assert "recall@1" in table and "Abstención" not in table
