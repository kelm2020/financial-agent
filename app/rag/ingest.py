from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path

from app.rag.corpus import KB_PATH, corpus_sha256, load_corpus
from app.rag.embeddings import EmbeddingClient
from app.rag.models import IndexMetadata, KnowledgeChunk
from app.rag.store import HybridStore


async def ingest_corpus(
    store: HybridStore,
    embeddings: EmbeddingClient,
    *,
    effective_on: date,
    root: Path = KB_PATH,
) -> IndexMetadata:
    chunks, corpus_hash = await asyncio.to_thread(_load_snapshot, root, effective_on)
    vectors = await embeddings.embed([chunk.embedding_text for chunk in chunks])
    versions = sorted({chunk.policy_version for chunk in chunks})
    metadata = IndexMetadata(
        kb_version="+".join(versions),
        embedding_model=embeddings.model_name,
        embedding_dimensions=embeddings.dimensions,
        corpus_sha256=corpus_hash,
        chunk_count=len(chunks),
    )
    await store.replace_index(chunks, vectors, metadata)
    return metadata


async def assert_index_current(
    store: HybridStore,
    embeddings: EmbeddingClient,
    *,
    root: Path = KB_PATH,
) -> IndexMetadata:
    metadata = await store.metadata()
    if metadata is None:
        raise RuntimeError("La base de conocimiento todavía no fue indexada; ejecutá make ingest")
    current_hash = await asyncio.to_thread(_hash_corpus, root)
    if metadata.corpus_sha256 != current_hash:
        raise RuntimeError("El corpus en disco no coincide con el índice; ejecutá make ingest")
    if metadata.embedding_model != embeddings.model_name:
        raise RuntimeError("El modelo de embeddings configurado no coincide con el índice")
    if metadata.embedding_dimensions != embeddings.dimensions:
        raise RuntimeError("La dimensión configurada no coincide con el índice")
    return metadata


def _hash_corpus(root: Path) -> str:
    return corpus_sha256(root.glob("*.md"))


def _load_snapshot(root: Path, effective_on: date) -> tuple[list[KnowledgeChunk], str]:
    return load_corpus(root, effective_on=effective_on), _hash_corpus(root)
