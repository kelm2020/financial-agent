from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from pydantic import SecretStr

from app.rag.config import load_embedding_config, load_reranker_config
from app.rag.embeddings import CachedEmbeddingClient, OpenAIEmbeddingClient
from app.rag.rerank import CachedReranker, CohereReranker, Reranker
from app.rag.retriever import PolicyRetriever
from app.rag.store import HybridStore
from config.settings import Settings

DATA_PATH = Path(__file__).parents[2] / "data"
EMBEDDING_CACHE_PATH = DATA_PATH / "query_cache.json"
RERANK_CACHE_PATH = DATA_PATH / "rerank_cache.json"


def _secret(value: SecretStr | None) -> str:
    return value.get_secret_value().strip() if value is not None else ""


@asynccontextmanager
async def embedding_client(
    settings: Settings,
    *,
    allow_network: bool,
    cache_path: Path = EMBEDDING_CACHE_PATH,
) -> AsyncIterator[CachedEmbeddingClient]:
    """Cache-first embeddings for the configured model.

    Offline (no key, or ``allow_network=False``) a cache miss raises instead of silently
    switching to a different embedding space.
    """
    config = load_embedding_config()
    api_key = _secret(settings.openai_api_key)
    if not allow_network or not api_key:
        yield CachedEmbeddingClient(
            None, cache_path, model_name=config.model, dimensions=config.dimensions
        )
        return
    live = OpenAIEmbeddingClient(api_key=api_key, model=config.model, dimensions=config.dimensions)
    try:
        yield CachedEmbeddingClient(live, cache_path)
    finally:
        await live.aclose()


@asynccontextmanager
async def reranker_client(
    settings: Settings,
    *,
    allow_network: bool,
    cache_path: Path = RERANK_CACHE_PATH,
) -> AsyncIterator[CachedReranker | None]:
    """Cache-first reranker: live Cohere only with a key and permission; otherwise the
    committed score cache offline; ``None`` when neither exists (nothing to measure)."""
    model = load_reranker_config().model
    api_key = _secret(settings.cohere_api_key)
    if allow_network and api_key:
        live = CohereReranker(api_key=api_key, model=model)
        try:
            yield CachedReranker(live, cache_path)
        finally:
            await live.aclose()
        return
    exists = await asyncio.to_thread(cache_path.exists)
    yield CachedReranker(None, cache_path, model_name=model) if exists else None


def build_retriever(
    store: HybridStore,
    embeddings: CachedEmbeddingClient,
    settings: Settings,
    *,
    reranker: Reranker | None = None,
    min_dense_score: float | None = None,
    evidence_gate: Literal["dense", "rerank"] | None = None,
) -> PolicyRetriever:
    gate = evidence_gate or settings.rag_evidence_gate
    return PolicyRetriever(
        store,
        embeddings,
        reranker=reranker,
        min_rrf_score=settings.rag_min_rrf_score,
        min_dense_score=(
            settings.rag_min_dense_score if min_dense_score is None else min_dense_score
        ),
        min_rerank_score=settings.rag_min_rerank_score,
        # Without a reranker only the dense gate exists.
        evidence_gate=gate if reranker is not None else "dense",
    )
