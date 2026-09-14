"""Measure end-to-end answerability (evals/answerability.yaml) with the real model and retriever.

Calls the providers: OpenAI for the agent model and query embeddings, Cohere for the reranker.
"""

from __future__ import annotations

import asyncio
from contextlib import AsyncExitStack
from datetime import date

from app.graph.context import Retriever
from app.llm.openai_responses import OpenAIResponsesLLM
from app.llm.protocol import LLMClient
from app.rag.factory import build_retriever, embedding_client, reranker_client
from app.rag.ingest import ingest_corpus
from app.rag.store import InMemoryHybridStore
from config.settings import Settings, get_settings
from evals.answerability import (
    AnswerabilityReport,
    evaluate_answerability,
    load_answerability_dataset,
)


def render(report: AnswerabilityReport) -> str:
    lines = [
        "| Positivas con cita correcta | Negativas sin respuesta citada | Citas incorrectas |",
        "|---:|---:|---:|",
        f"| {report.answered_correctly}/{report.positive_cases} | "
        f"{report.abstained_negatives}/{report.negative_cases} | {report.wrong_citations} |",
    ]
    for result in report.failures:
        lines.append(
            f"FAIL {result.id} [{result.kind}] intent={result.intent} outcome={result.outcome} "
            f"cited={list(result.cited)} retrieved={list(result.retrieved)} "
            f"flags={list(result.flags)} :: {result.query} -> {result.text}"
        )
    return "\n".join(lines)


async def _live_dependencies(
    stack: AsyncExitStack, settings: Settings, effective_on: date
) -> tuple[LLMClient, Retriever]:  # pragma: no cover - needs provider keys and network
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    if not key:
        raise SystemExit("OPENAI_API_KEY es necesaria para medir respuestas reales")
    embeddings = await stack.enter_async_context(embedding_client(settings, allow_network=True))
    reranker = await stack.enter_async_context(reranker_client(settings, allow_network=True))
    store = InMemoryHybridStore()
    await ingest_corpus(store, embeddings, effective_on=effective_on)
    model = settings.openai_agent_model
    llm = OpenAIResponsesLLM(
        api_key=key,
        model=model,
        max_output_tokens=2000,
        reasoning_effort="low" if model.startswith("gpt-5") else None,
    )
    return llm, build_retriever(store, embeddings, settings, reranker=reranker)


async def run(settings: Settings | None = None) -> AnswerabilityReport:
    dataset = load_answerability_dataset()
    async with AsyncExitStack() as stack:
        llm, retriever = await _live_dependencies(
            stack, settings or get_settings(), dataset.effective_on
        )
        report = await evaluate_answerability(dataset, llm=llm, retriever=retriever)
    print(render(report), flush=True)
    return report


def main() -> None:
    asyncio.run(run())


if __name__ == "__main__":  # pragma: no cover
    main()
