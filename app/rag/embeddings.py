from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import httpx

from app.rag.text import hashing_embedding


class EmbeddingClient(Protocol):
    @property
    def model_name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]: ...


class HashingEmbeddingClient:
    """Deterministic, network-free embedding used by the unit/evaluation suite."""

    def __init__(self, dimensions: int = 256) -> None:
        self._dimensions = dimensions

    @property
    def model_name(self) -> str:
        return "local-hashing-v1"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [hashing_embedding(text, self.dimensions) for text in texts]


class EmbeddingCacheError(RuntimeError):
    """The committed cache cannot serve this request; it is never silently rebuilt."""


class CachedEmbeddingClient:
    """Persistent embedding cache keyed by sha256(text), bound to one model and dimension.

    ``data/query_cache.json`` is committed so the evaluation runs without network: with
    ``delegate=None`` the client is offline and a miss is an error, not a silent fallback.
    Vectors are rounded so fresh and cached results are bit-identical.
    """

    def __init__(
        self,
        delegate: EmbeddingClient | None,
        path: Path,
        *,
        model_name: str | None = None,
        dimensions: int | None = None,
        precision: int = 6,
    ) -> None:
        if delegate is None and (model_name is None or dimensions is None):
            raise ValueError("Un caché offline necesita model_name y dimensions")
        self._delegate = delegate
        self._path = path
        self._model_name = delegate.model_name if delegate is not None else str(model_name)
        self._dimensions = delegate.dimensions if delegate is not None else int(dimensions or 0)
        self._precision = precision
        self._lock = asyncio.Lock()
        self._entries: dict[str, list[float]] | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        async with self._lock:
            if self._entries is None:
                self._entries = await asyncio.to_thread(self._load)
            keys = [hashlib.sha256(text.encode()).hexdigest() for text in texts]
            unique = dict(zip(keys, texts, strict=True))
            missing = [(key, text) for key, text in unique.items() if key not in self._entries]
            if missing:
                if self._delegate is None:
                    raise EmbeddingCacheError(
                        f"Faltan {len(missing)} embeddings de {self.model_name} en "
                        f"{self._path}; ejecutá make embeddings-cache"
                    )
                fresh = await self._delegate.embed([text for _, text in missing])
                for (key, _), vector in zip(missing, fresh, strict=True):
                    self._entries[key] = [round(value, self._precision) for value in vector]
                await asyncio.to_thread(self._save, self._entries)
            return [tuple(self._entries[key]) for key in keys]

    async def retain(self, texts: Sequence[str]) -> int:
        """Drop entries for texts no longer in use (e.g. an edited chunk) and persist."""
        async with self._lock:
            if self._entries is None:
                self._entries = await asyncio.to_thread(self._load)
            keep = {hashlib.sha256(text.encode()).hexdigest() for text in texts}
            removed = len(self._entries.keys() - keep)
            self._entries = {key: value for key, value in self._entries.items() if key in keep}
            await asyncio.to_thread(self._save, self._entries)
            return removed

    def _load(self) -> dict[str, list[float]]:
        if not self._path.exists():
            return {}
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
        except ValueError as exc:
            raise EmbeddingCacheError(f"{self._path} no es JSON válido") from exc
        if not isinstance(payload, dict) or not isinstance(payload.get("entries"), dict):
            raise EmbeddingCacheError(f"{self._path} no tiene el formato esperado")
        if payload.get("model") != self.model_name or payload.get("dimensions") != self.dimensions:
            raise EmbeddingCacheError(
                f"{self._path} fue generado con {payload.get('model')}/"
                f"{payload.get('dimensions')}, no con {self.model_name}/{self.dimensions}"
            )
        entries: dict[str, list[float]] = payload["entries"]
        if any(len(vector) != self.dimensions for vector in entries.values()):
            raise EmbeddingCacheError(f"{self._path} contiene vectores de otra dimensión")
        return entries

    def _save(self, entries: dict[str, list[float]]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(f"{self._path.suffix}.tmp")
        payload = {
            "model": self.model_name,
            "dimensions": self.dimensions,
            "entries": dict(sorted(entries.items())),
        }
        temporary.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        temporary.replace(self._path)


class OpenAIEmbeddingClient:
    """Minimal OpenAI-compatible embedding adapter kept behind the local Protocol."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        dimensions: int,
        client: httpx.AsyncClient | None = None,
        batch_size: int = 64,
    ) -> None:
        self._batch_size = batch_size
        self._model = model
        self._dimensions = dimensions
        self._headers = {"Authorization": f"Bearer {api_key}"}
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            timeout=30,
        )

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def dimensions(self) -> int:
        return self._dimensions

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        vectors: list[tuple[float, ...]] = []
        for start in range(0, len(texts), self._batch_size):
            batch = list(texts[start : start + self._batch_size])
            response = await self._client.post(
                "/embeddings",
                json={"model": self.model_name, "dimensions": self.dimensions, "input": batch},
                headers=self._headers,
            )
            response.raise_for_status()
            ordered = sorted(response.json()["data"], key=lambda item: item["index"])
            batch_vectors = [tuple(float(value) for value in item["embedding"]) for item in ordered]
            if len(batch_vectors) != len(batch) or any(
                len(vector) != self.dimensions for vector in batch_vectors
            ):
                raise ValueError("El proveedor devolvió embeddings con forma inesperada")
            vectors.extend(batch_vectors)
        return vectors

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
