"""Retrieval against the real Postgres + pgvector engine (RUN_POSTGRES_TESTS=1).

Tests run in a dedicated `<db>_rag_test` database migrated from scratch, so they never
overwrite the developer's index.
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from pydantic import SecretStr

from app.rag.config import load_reranker_config
from app.rag.corpus import load_corpus
from app.rag.evaluation import evaluate_retriever, load_retrieval_dataset
from app.rag.factory import RERANK_CACHE_PATH, build_retriever
from app.rag.ingest import assert_index_current, ingest_corpus
from app.rag.models import IndexMetadata
from app.rag.rerank import CachedReranker
from app.rag.store import DENSE_LEG_SQL, InMemoryHybridStore, PgVectorHybridStore, lexemes
from scripts import build_rerank_cache, evaluate_retrieval, ingest_knowledge
from tests.rag_support import (
    REFERENCE_DATE,
    FixedScoreReranker,
    cached_embeddings,
    offline_settings,
)

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="Set RUN_POSTGRES_TESTS=1 to exercise the real pgvector engine",
)


@pytest.fixture(scope="module")
def database_url() -> Iterator[str]:
    base = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/collections")
    name = f"{conninfo_to_dict(base)['dbname']}_rag_test"
    admin = make_conninfo(base, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = _as_url(base, name)
    try:
        config = Config("alembic.ini")
        command.upgrade(config, "head")
        # Round-trip the F2 migration so its downgrade is exercised too.
        command.downgrade(config, "0002_phase2_knowledge")
        command.upgrade(config, "head")
        yield _as_url(base, name)
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous


def _as_url(base: str, name: str) -> str:
    prefix, _, _ = base.rpartition("/")
    return f"{prefix}/{name}"


@pytest.fixture
async def store(database_url: str) -> AsyncIterator[PgVectorHybridStore]:
    async with PgVectorHybridStore(database_url) as opened:
        yield opened


@pytest.fixture
async def indexed(store: PgVectorHybridStore) -> PgVectorHybridStore:
    await ingest_corpus(store, cached_embeddings(), effective_on=REFERENCE_DATE)
    return store


async def test_metadata_is_empty_before_ingest_and_shapes_are_checked(database_url: str) -> None:
    async with PgVectorHybridStore(database_url) as fresh:
        async with fresh._pool.connection() as connection:
            await connection.execute("TRUNCATE kb_index_metadata")
        assert await fresh.metadata() is None
        metadata = IndexMetadata(
            kb_version="1",
            embedding_model="m",
            embedding_dimensions=1536,
            corpus_sha256="a" * 64,
            chunk_count=2,
        )
        with pytest.raises(ValueError, match="chunk_count"):
            await fresh.replace_index(
                load_corpus(effective_on=REFERENCE_DATE)[:1], [[0.0] * 1536], metadata
            )


async def test_ingest_round_trip_and_index_check(indexed: PgVectorHybridStore) -> None:
    metadata = await assert_index_current(indexed, cached_embeddings())
    assert (metadata.chunk_count, metadata.embedding_model) == (35, "text-embedding-3-large")
    assert await indexed.metadata() == metadata


async def test_indexed_lexemes_are_exactly_the_python_lexemes(indexed: PgVectorHybridStore) -> None:
    expected = {
        chunk.chunk_id: set(lexemes(chunk)) for chunk in load_corpus(effective_on=REFERENCE_DATE)
    }
    async with indexed._pool.connection() as connection:
        cursor = await connection.execute(
            "SELECT chunk_id, tsvector_to_array(content_search) FROM kb_chunks"
        )
        stored = {row[0]: set(row[1]) for row in await cursor.fetchall()}
    assert stored == expected


async def test_lexical_leg_folds_accents_stems_and_matches_memory(
    indexed: PgVectorHybridStore,
) -> None:
    embeddings = cached_embeddings()
    memory = InMemoryHybridStore()
    await ingest_corpus(memory, embeddings, effective_on=REFERENCE_DATE)
    vector = (await embeddings.embed(["¿puedo pagar con transferencia y cuánto tarda?"]))[0]

    async def lexical_sections(
        store: InMemoryHybridStore | PgVectorHybridStore, query: str
    ) -> set[str]:
        hits = await store.search(
            query, vector, topic="any", effective_on=REFERENCE_DATE,
            limit=70, candidate_limit=35, rrf_k=60,
        )  # fmt: skip
        return {hit.chunk.section_id for hit in hits if hit.lexical_rank is not None}

    accented = await lexical_sections(indexed, "acreditación")
    assert "PAY-MET-002" in accented
    assert len(accented) == 5  # `spanish` alone matched 0 unaccented; es_unaccent lost stems (3)
    for query in ("acreditacion", "ACREDITACIÓN", "se acreditan", "acreditado"):
        assert await lexical_sections(indexed, query) == accented, query
    for query in ("acreditación tardía", "débito automático", "vencimientos de las cuotas"):
        assert await lexical_sections(indexed, query) == await lexical_sections(memory, query)


async def test_dense_score_is_real_for_lexical_only_candidates(
    indexed: PgVectorHybridStore,
) -> None:
    # Dense leg points at vulnerability (ESC-002); the lexical leg at payment methods.
    vector = (await cached_embeddings().embed(
        ["perdí el trabajo y estoy enfermo, no puedo afrontar nada ahora"]
    ))[0]  # fmt: skip
    hits = await indexed.search(
        "débito automático", vector, topic="any", effective_on=REFERENCE_DATE,
        limit=35, candidate_limit=1, rrf_k=60,
    )  # fmt: skip
    lexical_only = [hit for hit in hits if hit.dense_rank is None]
    assert lexical_only
    assert all(hit.dense_score != 0 for hit in lexical_only)


async def test_dense_leg_can_use_the_hnsw_index(indexed: PgVectorHybridStore) -> None:
    vector = (await cached_embeddings().embed(["¿qué formas de pago aceptan?"]))[0]
    async with indexed._pool.connection() as connection, connection.transaction():
        # With 35 rows a sequential scan is cheaper; disabling the alternatives proves the
        # dense leg's shape is index-eligible (the old materialized CTE was not).
        for setting in ("enable_seqscan", "enable_bitmapscan", "enable_sort"):
            await connection.execute(f"SET LOCAL {setting} = off")
        cursor = await connection.execute(
            f"EXPLAIN {DENSE_LEG_SQL}",
            {
                "embedding": str(list(vector)),
                "topic": "any",
                "effective_on": REFERENCE_DATE,
                "candidate_limit": 20,
            },
        )
        plan = "\n".join(row[0] for row in await cursor.fetchall())
    assert "ix_kb_chunks_embedding_hnsw" in plan


async def test_postgres_matches_memory_evidence_and_held_out_metrics(
    indexed: PgVectorHybridStore,
) -> None:
    embeddings = cached_embeddings()
    memory = InMemoryHybridStore()
    await ingest_corpus(memory, embeddings, effective_on=REFERENCE_DATE)
    settings = offline_settings()
    for split in ("dev", "test"):
        dataset = load_retrieval_dataset(split)
        pg_metrics = await evaluate_retriever(
            build_retriever(indexed, embeddings, settings), dataset
        )
        memory_metrics = await evaluate_retriever(
            build_retriever(memory, embeddings, settings), dataset
        )
        # pgvector stores float32, so similarities agree to ~1e-4; the verdicts must match.
        for pg_case, memory_case in zip(pg_metrics.cases, memory_metrics.cases, strict=True):
            assert pg_case.dense_evidence == pytest.approx(memory_case.dense_evidence, abs=1e-4)
            assert pg_case.status == memory_case.status, pg_case.id
        assert pg_metrics.abstentions == memory_metrics.abstentions
    test_metrics = await evaluate_retriever(
        build_retriever(indexed, embeddings, settings), load_retrieval_dataset("test")
    )
    reranked = await evaluate_retriever(
        build_retriever(
            indexed, embeddings, settings, reranker=CachedReranker(
                None, RERANK_CACHE_PATH, model_name=load_reranker_config().model
            ), min_dense_score=-1.0,
        ),
        load_retrieval_dataset("test"),
    )  # fmt: skip
    assert (reranked.recall_at_3, round(reranked.mrr, 2)) == (1.0, 0.77)
    assert (test_metrics.recall_at_3, round(test_metrics.mrr, 2), test_metrics.abstentions) == (
        0.6,
        0.37,
        3,
    )


async def test_scripts_run_against_postgres(
    database_url: str, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = offline_settings(database_url=database_url)
    await ingest_knowledge.run(settings, effective_on=REFERENCE_DATE)
    await evaluate_retrieval.run("test", "postgres", settings=settings)
    output = capsys.readouterr().out
    assert "Indexed 35 chunks with text-embedding-3-large" in output
    assert "| test | postgres | no | 0,60 | 0,37 | 3/3 |" in output


async def test_rerank_cache_script_covers_postgres_candidates(
    database_url: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = offline_settings(database_url=database_url, cohere_api_key=SecretStr("k"))
    await ingest_knowledge.run(settings, effective_on=REFERENCE_DATE)
    path = tmp_path / "rerank.json"

    @asynccontextmanager
    async def live(_: object, *, allow_network: bool) -> AsyncIterator[CachedReranker]:
        yield CachedReranker(FixedScoreReranker(0.6), path)

    monkeypatch.setattr(build_rerank_cache, "reranker_client", live)
    await build_rerank_cache.run(settings, include_postgres=True)
    assert "Reranked 100 candidate lists" in capsys.readouterr().out


async def test_policy_candidates_have_memory_postgres_parity(indexed: PgVectorHybridStore) -> None:
    """Both stores hand the policy model the same number of candidates with the same recall.

    k was chosen on dev without a reranker. On the test split that ranking still misses one
    section inside k (R-02: "descuento" for POL-NEG-003), which the reranker fixes (ADR-011).
    """
    from app.graph.nodes.respond import POLICY_CANDIDATES

    embeddings = cached_embeddings()
    memory = InMemoryHybridStore()
    await ingest_corpus(memory, embeddings, effective_on=REFERENCE_DATE)
    left = build_retriever(memory, embeddings, offline_settings())
    right = build_retriever(indexed, embeddings, offline_settings())
    covered = {"memory": 0, "postgres": 0}
    for case in load_retrieval_dataset("test").positive:
        for name, retriever in (("memory", left), ("postgres", right)):
            result = await retriever.search_for_generation(
                case.query, effective_on=REFERENCE_DATE, limit=POLICY_CANDIDATES
            )
            assert len(result.hits) == POLICY_CANDIDATES
            covered[name] += set(case.expected_section_ids) <= set(result.source_chunk_ids)
    assert covered["memory"] == covered["postgres"]
