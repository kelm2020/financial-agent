"""Embed the current corpus and both evaluation splits into data/query_cache.json.

This is the only retrieval command that needs network. Everything else (unit tests,
calibration, evaluation) reads the committed cache offline and fails loudly on a miss.
"""

from __future__ import annotations

import asyncio

from app.rag.corpus import load_corpus
from app.rag.evaluation import load_retrieval_dataset
from app.rag.factory import embedding_client
from config.settings import Settings, get_settings


def cache_texts() -> list[str]:
    dev = load_retrieval_dataset("dev")
    test = load_retrieval_dataset("test")
    dates = {dev.effective_on, test.effective_on}
    texts = [
        chunk.embedding_text
        for effective_on in sorted(dates)
        for chunk in load_corpus(effective_on=effective_on)
    ]
    for dataset in (dev, test):
        texts += [case.query for case in dataset.positive]
        texts += [case.query for case in dataset.negative]
    return list(dict.fromkeys(texts))


async def run(settings: Settings | None = None) -> None:
    resolved = settings or get_settings()
    if resolved.openai_api_key is None or not resolved.openai_api_key.get_secret_value().strip():
        raise SystemExit("OPENAI_API_KEY es necesaria para generar el caché de embeddings")
    texts = cache_texts()
    async with embedding_client(resolved, allow_network=True) as embeddings:
        await embeddings.embed(texts)
        removed = await embeddings.retain(texts)
    print(f"Cached {len(texts)} embeddings with {embeddings.model_name}; pruned {removed}")


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
