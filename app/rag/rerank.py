from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Protocol

import httpx

from app.rag.models import SearchHit


class Reranker(Protocol):
    @property
    def model_name(self) -> str: ...

    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]: ...


def _retry_after(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("Retry-After", "")
    try:
        return max(float(header), 0.0)
    except ValueError:
        return float(min(2 ** (attempt + 1), 60))


def _by_relevance(hits: Sequence[SearchHit]) -> list[SearchHit]:
    return sorted(hits, key=lambda hit: (-(hit.rerank_score or 0.0), -hit.rrf_score))


class CohereReranker:
    """Cross-encoder reranking through Cohere's `/v2/rerank` (relevance scores in [0, 1])."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 6,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._max_retries = max_retries
        self._sleep = sleep
        self._model = model
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(base_url="https://api.cohere.com", timeout=30)

    @property
    def model_name(self) -> str:
        return self._model

    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        if not hits:
            return []
        payload = {
            "model": self._model,
            "query": query,
            "documents": [hit.chunk.embedding_text for hit in hits],
            "top_n": len(hits),
        }
        for attempt in range(self._max_retries + 1):
            response = await self._client.post("/v2/rerank", json=payload, headers=self._headers)
            if response.status_code != 429 or attempt == self._max_retries:
                break
            # Rate limited: a read with no side effects, so waiting and retrying is safe.
            await self._sleep(_retry_after(response, attempt))
        response.raise_for_status()
        results = response.json()["results"]
        indices = [int(item["index"]) for item in results]
        if sorted(indices) != list(range(len(hits))):
            raise ValueError("El reranker devolvió un conjunto de documentos inesperado")
        return _by_relevance(
            [
                hits[int(item["index"])].model_copy(
                    update={"rerank_score": min(max(float(item["relevance_score"]), 0.0), 1.0)}
                )
                for item in results
            ]
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


class RerankCacheError(RuntimeError):
    """The committed rerank cache cannot serve this request; it is never silently rebuilt."""


class CachedReranker:
    """Relevance scores cached per (model, query, document).

    A cross-encoder scores each pair independently, so a cached score is valid whatever
    candidate list the pair came from (memory or Postgres ordering). With ``delegate=None`` the
    reranker is offline and a miss is an error. ``data/rerank_cache.json`` makes reranked
    evaluations reproducible without network or credentials.
    """

    def __init__(
        self,
        delegate: Reranker | None,
        path: Path,
        *,
        model_name: str | None = None,
        precision: int = 6,
    ) -> None:
        if delegate is None and model_name is None:
            raise ValueError("Un reranker offline necesita model_name")
        self._delegate = delegate
        self._path = path
        self._model_name = delegate.model_name if delegate is not None else str(model_name)
        self._precision = precision
        self._lock = asyncio.Lock()
        self._entries: dict[str, float] | None = None
        self._used: set[str] = set()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_offline(self) -> bool:
        return self._delegate is None

    def _key(self, query: str, hit: SearchHit) -> str:
        return hashlib.sha256(
            f"{self._model_name}\0{query}\0{hit.chunk.embedding_text}".encode()
        ).hexdigest()

    async def rerank(self, query: str, hits: Sequence[SearchHit]) -> list[SearchHit]:
        async with self._lock:
            if self._entries is None:
                self._entries = await asyncio.to_thread(self._load)
            keys = [self._key(query, hit) for hit in hits]
            self._used.update(keys)
            missing = [hit for hit, key in zip(hits, keys, strict=True) if key not in self._entries]
            if missing:
                if self._delegate is None:
                    raise RerankCacheError(
                        f"Faltan {len(missing)} scores de {self.model_name} en {self._path}; "
                        "ejecutá make rerank-cache (requiere COHERE_API_KEY)"
                    )
                for scored in await self._delegate.rerank(query, missing):
                    self._entries[self._key(query, scored)] = round(
                        scored.rerank_score or 0.0, self._precision
                    )
                await asyncio.to_thread(self._save, self._entries)
            return _by_relevance(
                [
                    hit.model_copy(update={"rerank_score": self._entries[key]})
                    for hit, key in zip(hits, keys, strict=True)
                ]
            )

    async def retain_used(self) -> int:
        """Drop scores not requested since construction (stale corpus or queries)."""
        async with self._lock:
            if self._entries is None:
                self._entries = await asyncio.to_thread(self._load)
            removed = len(self._entries.keys() - self._used)
            self._entries = {k: v for k, v in self._entries.items() if k in self._used}
            await asyncio.to_thread(self._save, self._entries)
            return removed

    def _load(self) -> dict[str, float]:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise RerankCacheError(f"{self._path} no es JSON válido") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
            raise RerankCacheError(f"{self._path} no tiene el formato esperado")
        if payload.get("model") != self.model_name:
            raise RerankCacheError(
                f"{self._path} fue generado con {payload.get('model')}, no con {self.model_name}"
            )
        entries: dict[str, float] = payload["entries"]
        return entries

    def _save(self, entries: dict[str, float]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        payload = {"model": self.model_name, "entries": dict(sorted(entries.items()))}
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self._path)
