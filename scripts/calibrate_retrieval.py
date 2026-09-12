"""Calibrate the retrieval evidence gate on the dev split. It never reads the test split."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence

from app.rag.evaluation import (
    Calibration,
    calibrate_min_dense_score,
    calibrate_min_rerank_score,
    load_retrieval_dataset,
)
from app.rag.factory import build_retriever, embedding_client, reranker_client
from app.rag.ingest import ingest_corpus
from app.rag.store import InMemoryHybridStore
from config.settings import Settings, get_settings


async def run(settings: Settings | None = None, *, use_reranker: bool = False) -> Calibration:
    resolved = settings or get_settings()
    dataset = load_retrieval_dataset("dev")
    async with (
        embedding_client(resolved, allow_network=False) as embeddings,
        reranker_client(resolved, allow_network=False) as reranker,
    ):
        store = InMemoryHybridStore()
        await ingest_corpus(store, embeddings, effective_on=dataset.effective_on)
        if use_reranker:
            if reranker is None:
                raise SystemExit("No hay data/rerank_cache.json: ejecutá make rerank-cache")
            retriever = build_retriever(
                store, embeddings, resolved, reranker=reranker, evidence_gate="rerank"
            )
            calibration = await calibrate_min_rerank_score(retriever, dataset)
            variable = "RAG_MIN_RERANK_SCORE"
        else:
            retriever = build_retriever(store, embeddings, resolved, min_dense_score=-1.0)
            calibration = await calibrate_min_dense_score(retriever, dataset)
            variable = "RAG_MIN_DENSE_SCORE"
    print(
        f"{variable}={calibration.threshold:.3f} "
        f"(dev: {calibration.positives_answered}/{calibration.positive_cases} positivas "
        f"con evidencia, {calibration.negatives_abstained}/{calibration.negative_cases} "
        "negativas abstenidas)"
    )
    return calibration


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reranker", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    asyncio.run(run(use_reranker=args.reranker))


if __name__ == "__main__":  # pragma: no cover
    main()
