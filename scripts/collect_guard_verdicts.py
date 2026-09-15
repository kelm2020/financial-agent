"""Level-B guard verdicts: the real classifier over a split's user messages (§10.1.7, §15-F3).

The classifier reads what production reads: the text after preflight, with the production prompt
and the agent's client. A provider failure is retried once and then reported, never recorded as
benign: ``evaluate_guardrails`` refuses incomplete results, so a partial run cannot pass the gate.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from app.graph.nodes.guards import GUARD_CLASSIFIER_PROMPT
from app.guards.evaluation import load_guardrail_dataset
from app.guards.injection import GuardModelResult
from app.guards.preflight import PreflightPolicy, preflight_message
from app.llm.openai_responses import build_agent_llm
from app.llm.protocol import LLMClient
from config.settings import get_settings

ROOT = Path(__file__).parents[1]
ATTEMPTS = 2
CONCURRENCY = 8


async def classify(llm: LLMClient, text: str) -> GuardModelResult:
    sanitized = preflight_message(text, policy=PreflightPolicy()).sanitized_text
    return await llm.complete(
        task="guard_classifier",
        messages=(
            {"role": "system", "content": GUARD_CLASSIFIER_PROMPT},
            {"role": "user", "content": sanitized},
        ),
        response_model=GuardModelResult,
    )


async def collect(
    llm: LLMClient, cases: Sequence[tuple[str, str]], *, concurrency: int = CONCURRENCY
) -> tuple[dict[str, GuardModelResult], list[str]]:
    """Verdicts by case id, and the ids whose every attempt failed."""
    semaphore = asyncio.Semaphore(concurrency)
    results: dict[str, GuardModelResult] = {}
    failed: list[str] = []

    async def one(case_id: str, text: str) -> None:
        async with semaphore:
            for _attempt in range(ATTEMPTS):
                try:
                    results[case_id] = await classify(llm, text)
                    return
                except Exception:
                    continue
            failed.append(case_id)

    await asyncio.gather(*(one(case_id, text) for case_id, text in cases))
    return results, sorted(failed)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect level-B guard classifier verdicts")
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


async def run(args: argparse.Namespace) -> Path:
    settings = get_settings()
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required to collect guard verdicts")
    dataset = load_guardrail_dataset(ROOT / "evals" / "guardrails" / f"{args.split}.yaml")
    cases = [(case.case_id, case.text) for case in dataset.inputs if case.surface == "user"]
    llm = build_agent_llm(
        api_key=key, model=settings.openai_agent_model, check_model=settings.openai_check_model
    )
    try:
        results, failed = await collect(llm, cases)
    finally:
        await llm.aclose()
    output: Path = args.output or ROOT / "evals" / "guardrails" / f"classifier_{args.split}.json"
    payload = {case_id: results[case_id].model_dump() for case_id in sorted(results)}
    content = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    await asyncio.to_thread(output.write_text, content, encoding="utf-8")
    model = settings.openai_agent_model
    print(f"classifier {model} · {len(results)}/{len(cases)} verdicts · {output}")
    if failed:
        print(f"failed: {', '.join(failed)}")
        raise SystemExit(1)
    return output


def main(argv: Sequence[str] | None = None) -> None:
    asyncio.run(run(parse_args(argv)))


if __name__ == "__main__":  # pragma: no cover
    main()
