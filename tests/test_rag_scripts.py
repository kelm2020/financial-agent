from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Coroutine
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from pydantic import SecretStr

import app.rag.evaluation as evaluation
from app.rag.embeddings import CachedEmbeddingClient, HashingEmbeddingClient
from app.rag.models import IndexMetadata
from app.rag.rerank import CachedReranker
from scripts import (
    build_embedding_cache,
    build_rerank_cache,
    calibrate_retrieval,
    evaluate_retrieval,
    ingest_knowledge,
)
from tests.rag_support import FixedScoreReranker, OracleReranker, offline_settings


def _asyncio_run_spy(monkeypatch: pytest.MonkeyPatch) -> list[Coroutine[Any, Any, Any]]:
    observed: list[Coroutine[Any, Any, Any]] = []

    def fake_run(coroutine: Coroutine[Any, Any, Any]) -> None:
        observed.append(coroutine)
        coroutine.close()

    monkeypatch.setattr(asyncio, "run", fake_run)
    return observed


@pytest.mark.parametrize(
    "main",
    [
        build_embedding_cache.main,
        lambda: calibrate_retrieval.main([]),
        lambda: build_rerank_cache.main([]),
        ingest_knowledge.main,
        lambda: evaluate_retrieval.main(["--split", "dev", "--store", "memory"]),
    ],
)
def test_script_entrypoints_use_asyncio_run(
    monkeypatch: pytest.MonkeyPatch, main: Callable[[], None]
) -> None:
    observed = _asyncio_run_spy(monkeypatch)
    main()
    assert len(observed) == 1


def test_evaluate_arguments_default_to_held_out_split_in_memory() -> None:
    args = evaluate_retrieval.parse_args([])
    assert (args.split, args.store, args.reranker) == ("test", "memory", False)
    assert evaluate_retrieval.parse_args(["--reranker"]).reranker is True


async def test_evaluate_script_prints_metrics_and_each_failure(capsys: Any) -> None:
    metrics = await evaluate_retrieval.run("test", "memory", settings=offline_settings())
    output = capsys.readouterr().out
    assert "| test | memory | no | 0,50 | 0,33 | 3/3 |" in output
    assert output.count("FAIL ") == len(metrics.failures) == 3
    assert "FAIL R-06 [positive] status=no_evidence dense=0.268" in output


async def test_evaluate_script_refuses_to_report_a_reranker_it_cannot_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def no_reranker(_: object, *, allow_network: bool) -> AsyncIterator[None]:
        assert allow_network is False  # evaluation never calls the provider
        yield None

    monkeypatch.setattr(evaluate_retrieval, "reranker_client", no_reranker)
    with pytest.raises(SystemExit, match="rerank_cache"):
        await evaluate_retrieval.run(
            "test", "memory", use_reranker=True, settings=offline_settings()
        )


async def test_evaluate_script_labels_the_reranker_model(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    @asynccontextmanager
    async def fake_reranker(_: object, *, allow_network: bool) -> AsyncIterator[FixedScoreReranker]:
        yield FixedScoreReranker(0.9)

    monkeypatch.setattr(evaluate_retrieval, "reranker_client", fake_reranker)
    await evaluate_retrieval.run("test", "memory", use_reranker=True, settings=offline_settings())
    assert "| test | memory | fixed-score |" in capsys.readouterr().out


async def test_calibration_script_only_reads_the_dev_split(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    requested: list[str] = []
    original = evaluation.load_retrieval_dataset

    def spy(split: evaluation.Split, path: Path | None = None) -> evaluation.RetrievalDataset:
        requested.append(split)
        return original(split, path)

    monkeypatch.setattr(calibrate_retrieval, "load_retrieval_dataset", spy)
    calibration = await calibrate_retrieval.run(offline_settings())
    assert requested == ["dev"]
    assert calibration.threshold == offline_settings().rag_min_dense_score
    assert "RAG_MIN_DENSE_SCORE=0.505" in capsys.readouterr().out


async def test_build_cache_requires_an_api_key() -> None:
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        await build_embedding_cache.run(offline_settings())
    with pytest.raises(SystemExit, match="OPENAI_API_KEY"):
        await build_embedding_cache.run(offline_settings(openai_api_key=SecretStr("  ")))


async def test_build_cache_embeds_corpus_and_both_splits_and_prunes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    path = tmp_path / "cache.json"

    @asynccontextmanager
    async def fake_client(
        _: object, *, allow_network: bool
    ) -> AsyncIterator[CachedEmbeddingClient]:
        assert allow_network is True
        yield CachedEmbeddingClient(HashingEmbeddingClient(8), path)

    seeded = CachedEmbeddingClient(HashingEmbeddingClient(8), path)
    await seeded.embed(["texto obsoleto"])
    monkeypatch.setattr(build_embedding_cache, "embedding_client", fake_client)
    await build_embedding_cache.run(offline_settings(openai_api_key=SecretStr("sk")))
    texts = build_embedding_cache.cache_texts()
    assert len(texts) == 35 + 42 + 9
    assert (
        f"Cached {len(texts)} embeddings with local-hashing-v1; pruned 1" in capsys.readouterr().out
    )


async def test_ingest_script_uses_cache_first_embeddings_and_an_explicit_date(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    observed: dict[str, object] = {}

    class FakeStore:
        def __init__(self, url: str) -> None:
            observed["url"] = url

        async def __aenter__(self) -> FakeStore:
            return self

        async def __aexit__(self, *_: object) -> None:
            observed["closed"] = True

    @asynccontextmanager
    async def fake_client(_: object, *, allow_network: bool) -> AsyncIterator[str]:
        observed["allow_network"] = allow_network
        yield "embeddings"

    async def fake_ingest(
        store: object, embeddings: object, *, effective_on: date
    ) -> IndexMetadata:
        observed["effective_on"] = effective_on
        return IndexMetadata(
            kb_version="1.0.0",
            embedding_model="text-embedding-3-large",
            embedding_dimensions=1536,
            corpus_sha256="a" * 64,
            chunk_count=35,
        )

    monkeypatch.setattr(ingest_knowledge, "PgVectorHybridStore", FakeStore)
    monkeypatch.setattr(ingest_knowledge, "embedding_client", fake_client)
    monkeypatch.setattr(ingest_knowledge, "ingest_corpus", fake_ingest)
    settings = offline_settings(database_url="postgresql://example/db")
    await ingest_knowledge.run(settings, effective_on=date(2026, 9, 12))
    assert observed == {
        "allow_network": True,
        "url": "postgresql://example/db",
        "effective_on": date(2026, 9, 12),
        "closed": True,
    }
    assert "Indexed 35 chunks with text-embedding-3-large" in capsys.readouterr().out
    await ingest_knowledge.run(settings)
    assert observed["effective_on"] == date.today()


async def test_rerank_calibration_script_requires_a_cache_and_prints_the_variable(
    monkeypatch: pytest.MonkeyPatch, capsys: Any
) -> None:
    @asynccontextmanager
    async def none(_: object, *, allow_network: bool) -> AsyncIterator[None]:
        yield None

    monkeypatch.setattr(calibrate_retrieval, "reranker_client", none)
    with pytest.raises(SystemExit, match="make rerank-cache"):
        await calibrate_retrieval.run(offline_settings(), use_reranker=True)

    dev = evaluation.load_retrieval_dataset("dev")
    oracle = OracleReranker({case.query: case.expected_section_ids for case in dev.positive})

    @asynccontextmanager
    async def cached(_: object, *, allow_network: bool) -> AsyncIterator[OracleReranker]:
        assert allow_network is False
        yield oracle

    monkeypatch.setattr(calibrate_retrieval, "reranker_client", cached)
    calibration = await calibrate_retrieval.run(offline_settings(), use_reranker=True)
    assert calibration.gate == "rerank"
    assert "RAG_MIN_RERANK_SCORE=0.500 (dev: 32/32" in capsys.readouterr().out
    assert calibrate_retrieval.parse_args(["--reranker"]).reranker is True


async def test_build_rerank_cache_requires_key_and_scores_every_candidate_list(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: Any
) -> None:
    with pytest.raises(SystemExit, match="COHERE_API_KEY"):
        await build_rerank_cache.run(offline_settings())
    keyed = offline_settings(cohere_api_key=SecretStr("k"))
    delegate = FixedScoreReranker(0.4)
    path = tmp_path / "rerank.json"

    @asynccontextmanager
    async def live(_: object, *, allow_network: bool) -> AsyncIterator[CachedReranker]:
        assert allow_network is True
        yield CachedReranker(delegate, path)

    monkeypatch.setattr(build_rerank_cache, "reranker_client", live)
    await build_rerank_cache.run(keyed)
    assert delegate.calls == 42 + 9
    assert "Reranked 51 candidate lists with fixed-score; pruned 0" in capsys.readouterr().out
    assert build_rerank_cache.parse_args(["--postgres"]).postgres is True

    @asynccontextmanager
    async def broken(_: object, *, allow_network: bool) -> AsyncIterator[None]:
        yield None

    monkeypatch.setattr(build_rerank_cache, "reranker_client", broken)
    with pytest.raises(SystemExit, match="No se pudo construir"):
        await build_rerank_cache.run(keyed)
