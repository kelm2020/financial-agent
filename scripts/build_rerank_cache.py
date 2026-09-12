"""Score every dev/test candidate with the Cohere cross-encoder into data/rerank_cache.json.

Needs COHERE_API_KEY. Candidates come from the offline store and, with ``--postgres``, also
from the pgvector index, so evaluations of both stores can run offline afterwards.
"""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack

from app.rag.evaluation import load_retrieval_dataset
from app.rag.factory import build_retriever, embedding_client, reranker_client
from app.rag.ingest import assert_index_current, ingest_corpus
from app.rag.store import HybridStore, InMemoryHybridStore, PgVectorHybridStore
from config.settings import Settings, get_settings


async def run(settings: Settings | None = None, *, include_postgres: bool = False) -> None:
    resolved = settings or get_settings()
    if resolved.cohere_api_key is None or not resolved.cohere_api_key.get_secret_value().strip():
        raise SystemExit("COHERE_API_KEY es necesaria para generar el caché del reranker")
    datasets = [load_retrieval_dataset("dev"), load_retrieval_dataset("test")]
    async with AsyncExitStack() as stack:
        embeddings = await stack.enter_async_context(
            embedding_client(resolved, allow_network=False)
        )
        reranker = await stack.enter_async_context(reranker_client(resolved, allow_network=True))
        if reranker is None:
            raise SystemExit("No se pudo construir el reranker")
        memory = InMemoryHybridStore()
        await ingest_corpus(memory, embeddings, effective_on=datasets[0].effective_on)
        stores: list[HybridStore] = [memory]
        if include_postgres:
            postgres = await stack.enter_async_context(PgVectorHybridStore(resolved.database_url))
            await assert_index_current(postgres, embeddings)
            stores.append(postgres)
        queries = 0
        for store in stores:
            retriever = build_retriever(store, embeddings, resolved, reranker=reranker)
            for dataset in datasets:
                cases = [(c.query, c.topic) for c in dataset.positive]
                cases += [(c.query, c.topic) for c in dataset.negative]
                for query, topic in cases:
                    await retriever.retrieve(query, topic=topic, effective_on=dataset.effective_on)
                    queries += 1
        removed = await reranker.retain_used()
    print(f"Reranked {queries} candidate lists with {reranker.model_name}; pruned {removed}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--postgres", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(run(include_postgres=args.postgres))


if __name__ == "__main__":  # pragma: no cover
    main()
