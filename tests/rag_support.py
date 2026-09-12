from __future__ import annotations

from collections.abc import Sequence
from datetime import date

from app.rag.config import load_embedding_config
from app.rag.embeddings import CachedEmbeddingClient, HashingEmbeddingClient
from app.rag.factory import EMBEDDING_CACHE_PATH
from app.rag.models import KnowledgeChunk, SearchHit
from config.settings import Settings

REFERENCE_DATE = date(2026, 9, 12)


def offline_settings(**overrides: object) -> Settings:
    """Settings that ignore the developer's .env, so no test can reach the network."""
    values: dict[str, object] = {"openai_api_key": None, "cohere_api_key": None, **overrides}
    return Settings(_env_file=None, **values)  # type: ignore[arg-type]


def cached_embeddings() -> CachedEmbeddingClient:
    config = load_embedding_config()
    return CachedEmbeddingClient(
        None, EMBEDDING_CACHE_PATH, model_name=config.model, dimensions=config.dimensions
    )


class NamedHashingEmbeddingClient(HashingEmbeddingClient):
    def __init__(self, name: str, dimensions: int) -> None:
        super().__init__(dimensions)
        self._name = name

    @property
    def model_name(self) -> str:
        return self._name


class FixedScoreReranker:
    """Test double: assigns the same relevance to every hit."""

    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    @property
    def model_name(self) -> str:
        return "fixed-score"

    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        self.calls += 1
        return [hit.model_copy(update={"rerank_score": self.score}) for hit in hits]


def hit(chunk: KnowledgeChunk, *, dense: float = 0.9, rrf: float = 1.0) -> SearchHit:
    return SearchHit(
        chunk=chunk,
        lexical_score=1.0,
        dense_score=dense,
        lexical_rank=1,
        dense_rank=1,
        rrf_score=rrf,
    )


class OracleReranker:
    """Test double that scores 0.9 the labeled sections of each query and 0.1 the rest."""

    def __init__(self, relevant: dict[str, tuple[str, ...]]) -> None:
        self._relevant = relevant

    @property
    def model_name(self) -> str:
        return "oracle"

    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        relevant = self._relevant.get(query, ())
        scored = [
            hit.model_copy(
                update={"rerank_score": 0.9 if hit.chunk.section_id in relevant else 0.1}
            )
            for hit in hits
        ]
        return sorted(scored, key=lambda hit: -(hit.rerank_score or 0.0))
