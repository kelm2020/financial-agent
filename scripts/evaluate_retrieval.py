"""Evaluate retrieval on one split and store, printing metrics and every failing case."""

from __future__ import annotations

import argparse
import asyncio
from collections.abc import Sequence
from contextlib import AsyncExitStack
from typing import Literal

from app.rag.evaluation import RetrievalMetrics, Split, evaluate_retriever, load_retrieval_dataset
from app.rag.factory import build_retriever, embedding_client, reranker_client
from app.rag.ingest import assert_index_current, ingest_corpus
from app.rag.store import HybridStore, InMemoryHybridStore, PgVectorHybridStore
from config.settings import Settings, get_settings

type StoreKind = Literal["memory", "postgres"]


def _format(value: float) -> str:
    return f"{value:.2f}".replace(".", ",")


def render(metrics: RetrievalMetrics, store: StoreKind, reranker: str) -> str:
    lines = [
        "| Split | Store | Reranker | recall@3 | MRR | Abstención |",
        "|---|---|---|---:|---:|---:|",
        f"| {metrics.split} | {store} | {reranker} | {_format(metrics.recall_at_3)} | "
        f"{_format(metrics.mrr)} | {metrics.abstentions}/{metrics.negative_cases} |",
    ]
    if metrics.mode == "ranking":
        lines = [
            "| Split | Store | Reranker | recall@1 | recall@3 | MRR |",
            "|---|---|---|---:|---:|---:|",
            f"| {metrics.split} | {store} | {reranker} | {_format(metrics.recall_at_1)} | "
            f"{_format(metrics.recall_at_3)} | {_format(metrics.mrr)} |",
            "Ranking antes del gate; no mide respondibilidad ni abstención.",
        ]
    for case in metrics.failures:
        evidence = "-" if case.dense_evidence is None else f"{case.dense_evidence:.3f}"
        lines.append(
            f"FAIL {case.id} [{case.kind}] status={case.status} dense={evidence} "
            f"ranked={list(case.ranked_sections)} :: {case.query}"
        )
    lines.append(f"mode={metrics.mode} recall@1={metrics.recall_at_1:.3f}")
    return "\n".join(lines)


async def run(
    split: Split,
    store_kind: StoreKind,
    *,
    use_reranker: bool = False,
    before_gate: bool = False,
    settings: Settings | None = None,
) -> RetrievalMetrics:
    resolved = settings or get_settings()
    dataset = load_retrieval_dataset(split)
    async with AsyncExitStack() as stack:
        embeddings = await stack.enter_async_context(
            embedding_client(resolved, allow_network=False)
        )
        reranker = (
            await stack.enter_async_context(reranker_client(resolved, allow_network=False))
            if use_reranker
            else None
        )
        if use_reranker and reranker is None:
            raise SystemExit(
                "No hay data/rerank_cache.json: el reranker no se puede medir "
                "(make rerank-cache, requiere COHERE_API_KEY)"
            )
        store: HybridStore
        if store_kind == "memory":
            store = InMemoryHybridStore()
            await ingest_corpus(store, embeddings, effective_on=dataset.effective_on)
        else:
            store = await stack.enter_async_context(PgVectorHybridStore(resolved.database_url))
            await assert_index_current(store, embeddings)
        retriever = build_retriever(store, embeddings, resolved, reranker=reranker)
        metrics = await evaluate_retriever(retriever, dataset, before_gate=before_gate)
    label = reranker.model_name if reranker is not None else "no"
    print(render(metrics, store_kind, label))
    return metrics


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--store", choices=("memory", "postgres"), default="memory")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--before-gate", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metrics = asyncio.run(
        run(args.split, args.store, use_reranker=args.reranker, before_gate=args.before_gate)
    )
    if metrics is not None and metrics.failures:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
