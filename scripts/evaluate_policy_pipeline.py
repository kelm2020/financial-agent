"""End-to-end evaluation of policy questions through the agent: routing, answerability and safety.

Every row runs one real turn (graph, guards, mock backend and the hybrid retriever). Offline, the
agent answers from the calibrated retrieval gate and nothing judges the text; live, the model
answers and a judge of a different model checks that the visible answer responds. A row whose
measurement failed (retriever or model down) is reported as unavailable and left out of the
confusion matrix, never as a negative.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import shutil
import tempfile
from collections.abc import Sequence
from contextlib import AsyncExitStack
from datetime import date
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from app.guards.grounding import CITATION_LABEL
from app.llm.openai_responses import OpenAIResponsesLLM
from app.rag.factory import (
    EMBEDDING_CACHE_PATH,
    RERANK_CACHE_PATH,
    build_retriever,
    embedding_client,
    reranker_client,
)
from app.rag.ingest import ingest_corpus
from app.rag.store import InMemoryHybridStore
from config.settings import Settings
from evals.environment import agent_session
from evals.judge import JudgeSample, judge_response

# Sources whose turn must not search the policies: the answer lives elsewhere or nowhere.
_NO_RETRIEVAL_SOURCES = frozenset(
    {"escalation", "deflection", "backend", "action", "clarification"}
)
# Events that mean the turn could not be measured: a dependency failed, not the agent's judgment.
_INFRASTRUCTURE_EVENTS = frozenset(
    {"retriever_unavailable", "response_model_unavailable", "answer_check_unavailable"}
)
# The judge reads a neutral situation, never the labeled expectation (§9.4).
_SITUATION = "Cliente con deuda en mora que hace una consulta sobre las políticas de pago."


def confusion(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    available = [row for row in rows if row.get("available", True)]
    tp = sum(row["expected"] and row["answerable"] for row in available)
    fn = sum(row["expected"] and not row["answerable"] for row in available)
    fp = sum(not row["expected"] and row["answerable"] for row in available)
    tn = sum(not row["expected"] and not row["answerable"] for row in available)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "unavailable": len(rows) - len(available),
    }


def answered(events: Sequence[dict[str, Any]]) -> bool:
    """The customer read a policy answer (model, verbatim quotes or extract), not an abstention."""
    return any(
        event["type"] == "policy_answer" and event.get("outcome") != "abstained" for event in events
    )


def score_row(case: dict[str, Any], observed: dict[str, Any]) -> bool:
    """Routing, safety, answerability and, for an answer, a labeled source section. The judge
    verdict is added by the caller when live."""
    tools = observed["tools"]
    safe = not observed["writes"]
    if case["source"] in _NO_RETRIEVAL_SOURCES:
        safe = safe and "search_policies" not in tools
    if case["source"] == "escalation":
        safe = safe and "request_human" in tools
    answer_ok = case["source"] != "rag" or observed["answerable"] == case["answerable"]
    if answer_ok and case["source"] == "rag" and case["answerable"]:
        # An answer from another section is not the answer: offline, "¿cuánto tarda la
        # transferencia?" was answered with FAQ-010, about a relative paying.
        cited = {label.strip("[]").upper() for label in CITATION_LABEL.findall(observed["text"])}
        answer_ok = bool(cited & {section.upper() for section in case["sections"]})
    return bool(observed["source"] == case["source"] and safe and answer_ok)


async def run(args: argparse.Namespace) -> dict[str, Any]:
    dataset_path = Path(args.dataset)
    dataset_bytes = await asyncio.to_thread(dataset_path.read_bytes)
    dataset = yaml.safe_load(dataset_bytes)
    settings = Settings()
    cache = Path(args.cache_dir)
    await asyncio.to_thread(cache.mkdir, parents=True, exist_ok=True)
    for source in (EMBEDDING_CACHE_PATH, RERANK_CACHE_PATH):
        target = cache / source.name
        if not await asyncio.to_thread(target.exists):
            await asyncio.to_thread(shutil.copyfile, source, target)
    report: dict[str, Any] = {
        "dataset": str(dataset_path),
        "dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(),
        "mode": "live" if args.live else "offline",
        "reranker": args.reranker,
        "agent_model": settings.openai_agent_model if args.live else None,
        "judge_model": args.judge_model if args.live else None,
        "cases": [],
    }
    async with AsyncExitStack() as stack:
        embeddings = await stack.enter_async_context(
            embedding_client(
                settings, allow_network=args.live, cache_path=cache / EMBEDDING_CACHE_PATH.name
            )
        )
        reranker = None
        if args.reranker:
            reranker = await stack.enter_async_context(
                reranker_client(
                    settings, allow_network=args.live, cache_path=cache / RERANK_CACHE_PATH.name
                )
            )
            if reranker is None:
                raise RuntimeError("Reranker unavailable; do not report a reranked evaluation")
        llm = judge = None
        if args.live:
            if settings.openai_api_key is None:
                raise RuntimeError("OPENAI_API_KEY required for --live")
            if args.judge_model == settings.openai_agent_model:
                raise ValueError("The judge must use a different model than the agent")
            key = settings.openai_api_key.get_secret_value()
            llm = OpenAIResponsesLLM(
                api_key=key,
                model=settings.openai_agent_model,
                max_output_tokens=4000,
                reasoning_effort="low",
                timeout_seconds=45,
            )
            stack.push_async_callback(llm.aclose)
            judge = OpenAIResponsesLLM(
                api_key=key, model=args.judge_model, max_output_tokens=1500, timeout_seconds=45
            )
            stack.push_async_callback(judge.aclose)
        store = InMemoryHybridStore()
        effective_on = dataset["effective_on"]
        if isinstance(effective_on, str):
            effective_on = date.fromisoformat(effective_on)
        await ingest_corpus(store, embeddings, effective_on=effective_on)
        retriever = build_retriever(store, embeddings, settings, reranker=reranker)
        output = Path(args.output)
        for case in dataset["cases"]:
            row = await _evaluate_case(case, llm=llm, judge=judge, retriever=retriever)
            report["cases"].append(row)
            report["metrics"] = confusion(
                [item for item in report["cases"] if item["expected_source"] == "rag"]
            )
            report["passed"] = sum(item["passed"] for item in report["cases"])
            report["total"] = len(report["cases"])
            await asyncio.to_thread(output.parent.mkdir, parents=True, exist_ok=True)
            await asyncio.to_thread(
                output.write_text, json.dumps(report, ensure_ascii=False, indent=2)
            )
            print(
                f"{case['id']} {'PASS' if row['passed'] else 'FAIL'} "
                f"available={row['available']} answerable={row['answerable']}",
                flush=True,
            )
    return report


async def _evaluate_case(
    case: dict[str, Any], *, llm: Any, judge: Any, retriever: Any
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": case["id"],
        "query": case["query"],
        "expected": case["answerable"],
        "expected_source": case["source"],
        "available": True,
        "answerable": False,
        "passed": False,
    }
    try:
        async with agent_session("CUST-00125", llm=llm, retriever=retriever) as session:
            turn = await session.send(case["query"])
            events = list(session.recorder.events)
            row.update(
                source=turn.state.get("selected_source", ""),
                answerable=answered(events),
                text=turn.text,
                tools=[call.name for call in session.recorder.tool_calls],
                llm_tasks=list(turn.llm_tasks),
                writes=len(session.recorder.agreement_writes),
                policy_answer=[event for event in events if event["type"] == "policy_answer"],
            )
        failed = sorted({event["type"] for event in events} & _INFRASTRUCTURE_EVENTS)
        if failed:
            row.update(available=False, error=",".join(failed))
            return row
        passed = score_row(case, row)
        if judge is not None:
            judgment = await judge_response(
                judge,
                JudgeSample(
                    id=case["id"], situation=_SITUATION, user=case["query"], response=row["text"]
                ),
            )
            row["judge"] = judgment.model_dump()
            passed = passed and judgment.verdict("responde_lo_pedido") == "pass"
        row["semantic_checked"] = judge is not None
        row["passed"] = passed
    except Exception as exc:
        row.update(available=False, passed=False, error=type(exc).__name__)
    return row


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="evals/policy_regression.yaml")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--reranker", action="store_true")
    parser.add_argument("--judge-model", default="gpt-4.1-mini-2025-04-14")
    parser.add_argument(
        "--cache-dir", default=str(Path(tempfile.gettempdir()) / "financial-agent-policy-cache")
    )
    parser.add_argument("--output", default="evals/reports/policy_pipeline.json")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    report = asyncio.run(run(parse_args(argv)))
    print(json.dumps({key: value for key, value in report.items() if key != "cases"}, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":  # pragma: no cover
    main()
