from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from datetime import date
from types import TracebackType
from typing import Any, Protocol, Self

from pgvector import Vector
from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.rag.models import IndexMetadata, KnowledgeChunk, SearchHit, StoredChunk, Topic
from app.rag.text import cosine, tokenize


class HybridStore(Protocol):
    async def replace_index(
        self,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Sequence[Sequence[float]],
        metadata: IndexMetadata,
    ) -> None: ...

    async def search(
        self,
        query: str,
        query_embedding: Sequence[float],
        *,
        topic: Topic,
        effective_on: date,
        limit: int,
        candidate_limit: int,
        rrf_k: int,
    ) -> list[SearchHit]: ...

    async def metadata(self) -> IndexMetadata | None: ...


def lexemes(chunk: KnowledgeChunk) -> tuple[str, ...]:
    """Indexed lexical tokens; Postgres stores exactly these (migration 0003)."""
    return tokenize(f"{chunk.contextualized_content} {chunk.heading} {chunk.content}")


def _check_shapes(
    chunks: Sequence[KnowledgeChunk],
    embeddings: Sequence[Sequence[float]],
    metadata: IndexMetadata,
) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Cada chunk debe tener exactamente un embedding")
    if metadata.chunk_count != len(chunks):
        raise ValueError("chunk_count no coincide con el índice")
    if any(len(embedding) != metadata.embedding_dimensions for embedding in embeddings):
        raise ValueError("La dimensión de un embedding no coincide con la metadata")


def _normalized_rrf(ranks: Sequence[int | None], rrf_k: int) -> float:
    raw = sum(1 / (rrf_k + rank) for rank in ranks if rank is not None)
    return min(raw / (2 / (rrf_k + 1)), 1.0)


class InMemoryHybridStore:
    """Offline store with the same fusion contract as Postgres, used by unit tests.

    Lexical scoring is BM25 over the same stemmed, accent-folded text Postgres indexes; the
    dense score is reported for every candidate, exactly as the SQL implementation does.
    """

    def __init__(self) -> None:
        self._records: tuple[StoredChunk, ...] = ()
        self._metadata: IndexMetadata | None = None

    async def replace_index(
        self,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Sequence[Sequence[float]],
        metadata: IndexMetadata,
    ) -> None:
        _check_shapes(chunks, embeddings, metadata)
        self._records = tuple(
            StoredChunk(chunk=chunk, embedding=tuple(float(value) for value in embedding))
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        )
        self._metadata = metadata

    async def metadata(self) -> IndexMetadata | None:
        return self._metadata

    async def search(
        self,
        query: str,
        query_embedding: Sequence[float],
        *,
        topic: Topic,
        effective_on: date,
        limit: int,
        candidate_limit: int,
        rrf_k: int,
    ) -> list[SearchHit]:
        eligible = [
            record
            for record in self._records
            if (topic == "any" or record.chunk.topic == topic)
            and record.chunk.status == "approved"
            and record.chunk.valid_from <= effective_on
            and (record.chunk.valid_until is None or record.chunk.valid_until >= effective_on)
        ]
        if not eligible:
            return []
        lexical_scores = _bm25_scores(
            tokenize(query), [lexemes(record.chunk) for record in eligible]
        )
        dense_scores = [cosine(record.embedding, query_embedding) for record in eligible]
        indices = range(len(eligible))
        lexical_order = sorted(
            (index for index in indices if lexical_scores[index] > 0),
            key=lambda index: (-lexical_scores[index], eligible[index].chunk.chunk_id),
        )[:candidate_limit]
        dense_order = sorted(
            indices, key=lambda index: (-dense_scores[index], eligible[index].chunk.chunk_id)
        )[:candidate_limit]
        lexical_ranks = {item: rank for rank, item in enumerate(lexical_order, start=1)}
        dense_ranks = {item: rank for rank, item in enumerate(dense_order, start=1)}
        hits = [
            SearchHit(
                chunk=eligible[index].chunk,
                lexical_score=lexical_scores[index],
                dense_score=dense_scores[index],
                lexical_rank=lexical_ranks.get(index),
                dense_rank=dense_ranks.get(index),
                rrf_score=_normalized_rrf(
                    (lexical_ranks.get(index), dense_ranks.get(index)), rrf_k
                ),
            )
            for index in set(lexical_ranks) | set(dense_ranks)
        ]
        hits.sort(key=lambda hit: (-hit.rrf_score, -hit.dense_score, hit.chunk.chunk_id))
        return hits[:limit]


def _bm25_scores(query: Sequence[str], documents: Sequence[Sequence[str]]) -> list[float]:
    if not documents or not query:
        return [0.0] * len(documents)
    document_frequency = Counter(
        term for document in documents for term in set(document) if term in query
    )
    average_length = sum(len(document) for document in documents) / len(documents)
    k1, b = 1.5, 0.75
    scores: list[float] = []
    for document in documents:
        frequencies = Counter(document)
        score = 0.0
        for term in query:
            frequency = frequencies[term]
            if frequency == 0:
                continue
            doc_frequency = document_frequency[term]
            inverse_frequency = math.log(
                1 + (len(documents) - doc_frequency + 0.5) / (doc_frequency + 0.5)
            )
            denominator = frequency + k1 * (1 - b + b * len(document) / max(average_length, 1))
            score += inverse_frequency * frequency * (k1 + 1) / denominator
        scores.append(score)
    return scores


def lexical_query(query: str) -> str:
    """OR of the query's lexemes, for `to_tsquery('simple', ...)` (no further processing)."""
    return " | ".join(dict.fromkeys(tokenize(query)))


async def _configure(connection: AsyncConnection[Any]) -> None:
    await register_vector_async(connection)


class PgVectorHybridStore:
    """Postgres implementation; both retrieval legs and RRF run in one SQL query."""

    def __init__(
        self,
        database_url: str,
        *,
        min_size: int = 1,
        max_size: int = 4,
        statement_timeout_ms: int = 5_000,
    ) -> None:
        self._pool = AsyncConnectionPool(
            database_url,
            min_size=min_size,
            max_size=max_size,
            open=False,
            configure=_configure,
            kwargs={"options": f"-c statement_timeout={statement_timeout_ms}"},
        )

    async def open(self) -> None:
        await self._pool.open(wait=True)

    async def close(self) -> None:
        await self._pool.close()

    async def __aenter__(self) -> Self:
        await self.open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def replace_index(
        self,
        chunks: Sequence[KnowledgeChunk],
        embeddings: Sequence[Sequence[float]],
        metadata: IndexMetadata,
    ) -> None:
        _check_shapes(chunks, embeddings, metadata)
        async with self._pool.connection() as connection, connection.transaction():
            await connection.execute("SELECT pg_advisory_xact_lock(67242026)")
            await connection.execute("DELETE FROM kb_chunks")
            async with connection.cursor() as cursor:
                await cursor.executemany(
                    """
                    INSERT INTO kb_chunks (
                        chunk_id, document_id, section_id, topic, heading, content,
                        contextualized_content, policy_version, status, valid_from,
                        valid_until, audience, applicable_segments, embedding_model, embedding,
                        content_search
                    ) VALUES (
                        %(chunk_id)s, %(document_id)s, %(section_id)s, %(topic)s,
                        %(heading)s, %(content)s, %(contextualized_content)s,
                        %(policy_version)s, %(status)s, %(valid_from)s, %(valid_until)s,
                        %(audience)s, %(applicable_segments)s, %(embedding_model)s,
                        %(embedding)s, to_tsvector('simple', %(lexemes)s)
                    )
                    """,
                    [
                        {
                            **chunk.model_dump(mode="python"),
                            "audience": list(chunk.audience),
                            "applicable_segments": list(chunk.applicable_segments),
                            "embedding_model": metadata.embedding_model,
                            "embedding": Vector(list(embedding)),
                            "lexemes": " ".join(lexemes(chunk)),
                        }
                        for chunk, embedding in zip(chunks, embeddings, strict=True)
                    ],
                )
            await connection.execute(
                """
                INSERT INTO kb_index_metadata (
                    index_name, kb_version, embedding_model, embedding_dimensions,
                    corpus_sha256, chunk_count, indexed_at
                ) VALUES ('policies', %(kb_version)s, %(embedding_model)s,
                          %(embedding_dimensions)s, %(corpus_sha256)s,
                          %(chunk_count)s, CURRENT_TIMESTAMP)
                ON CONFLICT (index_name) DO UPDATE SET
                    kb_version = EXCLUDED.kb_version,
                    embedding_model = EXCLUDED.embedding_model,
                    embedding_dimensions = EXCLUDED.embedding_dimensions,
                    corpus_sha256 = EXCLUDED.corpus_sha256,
                    chunk_count = EXCLUDED.chunk_count,
                    indexed_at = EXCLUDED.indexed_at
                """,
                metadata.model_dump(mode="python"),
            )

    async def metadata(self) -> IndexMetadata | None:
        async with self._pool.connection() as connection:
            cursor = connection.cursor(row_factory=dict_row)
            await cursor.execute(
                """
                SELECT kb_version, embedding_model, embedding_dimensions,
                       corpus_sha256, chunk_count
                FROM kb_index_metadata WHERE index_name = 'policies'
                """
            )
            row = await cursor.fetchone()
        return IndexMetadata.model_validate(row, strict=True) if row is not None else None

    async def search(
        self,
        query: str,
        query_embedding: Sequence[float],
        *,
        topic: Topic,
        effective_on: date,
        limit: int,
        candidate_limit: int,
        rrf_k: int,
    ) -> list[SearchHit]:
        async with self._pool.connection() as connection:
            cursor = connection.cursor(row_factory=dict_row)
            await cursor.execute(
                HYBRID_SQL,
                {
                    "lexical_query": lexical_query(query),
                    "embedding": Vector(list(query_embedding)),
                    "topic": topic,
                    "effective_on": effective_on,
                    "candidate_limit": candidate_limit,
                    "limit": limit,
                    "rrf_k": rrf_k,
                },
            )
            rows = await cursor.fetchall()
        return [_row_to_hit(row) for row in rows]


_ELIGIBLE = """
      (%(topic)s = 'any' OR c.topic = %(topic)s)
      AND c.status = 'approved'
      AND c.valid_from <= %(effective_on)s
      AND (c.valid_until IS NULL OR c.valid_until >= %(effective_on)s)
"""

# Each leg reads kb_chunks directly (no shared CTE): a CTE referenced several times is
# materialized and hides the HNSW and GIN indexes from the planner.
DENSE_LEG_SQL = f"""
SELECT c.chunk_id, c.embedding <=> %(embedding)s AS distance
FROM kb_chunks c
WHERE c.embedding IS NOT NULL AND {_ELIGIBLE}
ORDER BY c.embedding <=> %(embedding)s
LIMIT %(candidate_limit)s
"""

HYBRID_SQL = f"""
WITH lexical AS (
    SELECT c.chunk_id,
           ts_rank_cd(c.content_search, to_tsquery('simple', %(lexical_query)s)) AS score,
           row_number() OVER (
               ORDER BY ts_rank_cd(c.content_search, to_tsquery('simple', %(lexical_query)s))
                        DESC,
                        c.chunk_id
           ) AS rank
    FROM kb_chunks c
    WHERE c.content_search @@ to_tsquery('simple', %(lexical_query)s) AND {_ELIGIBLE}
    ORDER BY rank
    LIMIT %(candidate_limit)s
),
dense AS (
    SELECT chunk_id, row_number() OVER (ORDER BY distance, chunk_id) AS rank
    FROM ({DENSE_LEG_SQL}) nearest
),
fused AS (
    SELECT chunk_id,
           lexical.score AS lexical_score,
           lexical.rank AS lexical_rank,
           dense.rank AS dense_rank,
           LEAST(
             (
               COALESCE(1.0 / (%(rrf_k)s + lexical.rank), 0) +
               COALESCE(1.0 / (%(rrf_k)s + dense.rank), 0)
             ) / (2.0 / (%(rrf_k)s + 1)),
             1.0
           ) AS rrf_score
    FROM lexical FULL OUTER JOIN dense USING (chunk_id)
)
SELECT c.chunk_id, c.document_id, c.section_id, c.topic, c.heading, c.content,
       c.contextualized_content, c.policy_version, c.status, c.valid_from,
       c.valid_until, c.audience, c.applicable_segments,
       COALESCE(f.lexical_score, 0) AS lexical_score,
       COALESCE(1 - (c.embedding <=> %(embedding)s), 0) AS dense_score,
       f.lexical_rank, f.dense_rank, f.rrf_score
FROM fused f JOIN kb_chunks c USING (chunk_id)
ORDER BY f.rrf_score DESC, dense_score DESC, c.chunk_id
LIMIT %(limit)s
"""

_CHUNK_FIELDS = (
    "chunk_id",
    "document_id",
    "section_id",
    "topic",
    "heading",
    "content",
    "contextualized_content",
    "policy_version",
    "status",
    "valid_from",
    "valid_until",
    "audience",
    "applicable_segments",
)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(str(value))


def _row_to_hit(row: dict[str, Any]) -> SearchHit:
    chunk_fields = {key: row[key] for key in _CHUNK_FIELDS}
    for key in ("audience", "applicable_segments"):
        chunk_fields[key] = tuple(chunk_fields[key])
    return SearchHit(
        chunk=KnowledgeChunk.model_validate(chunk_fields, strict=True),
        lexical_score=float(row["lexical_score"]),
        # Clamp float noise from `1 - cosine distance` into the model's [-1, 1] contract.
        dense_score=max(-1.0, min(1.0, float(row["dense_score"]))),
        lexical_rank=_optional_int(row["lexical_rank"]),
        dense_rank=_optional_int(row["dense_rank"]),
        rrf_score=float(row["rrf_score"]),
    )
