from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from app.graph.confirmation import CONFIRMATION_INSTRUCTION
from app.graph.nodes.guards import GUARD_CLASSIFIER_PROMPT, ROUTER_INSTRUCTION
from app.graph.nodes.respond import GROUNDED_INSTRUCTION
from app.llm.openai_responses import OpenAIResponsesLLM, build_agent_llm
from app.llm.protocol import LLMClient
from app.prompts import SYSTEM_PROMPT_PATH
from app.rag.support import ANSWER_INSTRUCTION
from config.settings import get_settings
from evals.dataset import DATASETS, DatasetName, load_cases
from evals.environment import run_case
from evals.evaluators import aggregate_metrics, evaluate_case
from evals.judge import JudgeSample, JudgeVerdicts, judge_response
from evals.judge_workflow import turn_sample
from evals.models import (
    ALL_CRITERIA,
    CaseObservation,
    EvalMetrics,
    EvalReport,
    ExpandedCase,
    Rate,
)
from evals.reporting import render_report, write_report
from evals.variants import expand_cases

REPORTS_DIR = Path(__file__).parent / "reports"
BASELINES_PATH = Path(__file__).parent / "baselines.json"
SAFETY_GATES = frozenset({"hallucinated_numbers", "unsafe_auto_action", "confirmation_bypass"})
# §11.5: a drop in tool selection larger than this below the recorded baseline blocks a merge.
F1_REGRESSION_TOLERANCE = 0.05


def baseline_f1(suite: str, dataset: str) -> float | None:
    """Recorded tool_selection_f1 for this suite and dataset, or None without a record."""
    if not BASELINES_PATH.exists():
        return None
    entry = json.loads(BASELINES_PATH.read_text(encoding="utf-8")).get(f"{suite}:{dataset}")
    return float(entry["tool_selection_f1"]) if entry else None


ProgressCallback = Callable[[int, int, str, int, int, bool], None]


def _prompt_fingerprint() -> str:
    material = b"\0".join(
        (
            SYSTEM_PROMPT_PATH.read_bytes(),
            GUARD_CLASSIFIER_PROMPT.encode(),
            ROUTER_INSTRUCTION.encode(),
            GROUNDED_INSTRUCTION.encode(),
            ANSWER_INSTRUCTION.encode(),
            CONFIRMATION_INSTRUCTION.encode(),
        )
    )
    return hashlib.sha256(material).hexdigest()[:16]


async def _judge_with_retry(judge: LLMClient, sample: JudgeSample) -> JudgeVerdicts | None:
    for _attempt in range(2):
        try:
            return await judge_response(judge, sample)
        except Exception:
            # A judge failure must not discard hundreds of agent runs already observed.
            continue
    return None


async def judge_quality(
    cases: Sequence[ExpandedCase],
    observations: Sequence[CaseObservation],
    judge: LLMClient,
    metrics: EvalMetrics,
) -> EvalMetrics:
    """Conversational quality of every response of every run. Reported, never a gate (§11.5).

    Every turn is judged, not only the last one: a cold confirmation summary in turn 2 is as
    visible to the customer as the final message. Identical (situation, conversation, message,
    response, criteria) tuples are judged once, so k repetitions of a template do not multiply
    cost."""
    cache: dict[tuple[object, ...], JudgeVerdicts | None] = {}
    passed = judged = unjudged = 0
    by_criterion: dict[str, list[int]] = {criterion: [0, 0] for criterion in ALL_CRITERIA}
    for case, observed in zip(cases, observations, strict=True):
        for index in range(len(observed.turns)):
            sample = turn_sample(case, observed, index)
            key = (
                sample.situation,
                sample.conversation,
                sample.user,
                sample.response,
                sample.criteria,
            )
            if key not in cache:
                cache[key] = await _judge_with_retry(judge, sample)
            verdicts = cache[key]
            if verdicts is None:
                unjudged += 1
                continue
            judged += 1
            passed += verdicts.acceptable(sample.criteria)
            for criterion in sample.criteria:
                by_criterion[criterion][0] += verdicts.verdict(criterion) == "pass"
                by_criterion[criterion][1] += 1
    return metrics.model_copy(
        update={
            "quality_judged": Rate(numerator=passed, denominator=judged),
            "quality_unjudged": unjudged,
            "quality_by_criterion": {
                name: Rate(numerator=values[0], denominator=values[1])
                for name, values in by_criterion.items()
                if values[1]
            },
        }
    )


async def evaluate(
    *,
    suite: str = "level-a",
    dataset: DatasetName = "canonical",
    k: int = 1,
    cases_path: Path | None = None,
    case_filter: str | None = None,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
    cached_cost_per_million: float | None = None,
    judge_llm: LLMClient | None = None,
    judge_model: str | None = None,
    progress: ProgressCallback | None = None,
    concurrency: int = 1,
) -> EvalReport:
    if k < 1:
        raise ValueError("k must be at least 1")
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")
    default_path, expected_base = DATASETS[dataset]
    bases = load_cases(cases_path or default_path, expected_base=expected_base)
    cases = list(expand_cases(bases))
    if case_filter:
        cases = [case for case in cases if case.base_id == case_filter or case.id == case_filter]
    if not cases:
        raise ValueError(f"No cases matched {case_filter!r}")

    llm: OpenAIResponsesLLM | None = None
    agent_model: str | None = None
    check_model: str | None = None
    if suite == "live":
        settings = get_settings()
        key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
        if not key:
            raise RuntimeError("OPENAI_API_KEY is required for --suite live")
        agent_model = settings.openai_agent_model
        check_model = settings.openai_check_model
        llm = build_agent_llm(api_key=key, model=agent_model, check_model=check_model)
    elif suite != "level-a":
        raise ValueError(f"Unknown suite {suite!r}")

    per_case_passes: dict[str, list[bool]] = {case.id: [] for case in cases}
    schedule = [(case, repetition) for case in cases for repetition in range(1, k + 1)]
    total_runs = len(schedule)
    results: list[tuple[CaseObservation, bool] | None] = [None] * total_runs
    completed_runs = 0
    # Runs are independent: each opens its own session, recorder and transport, and bills its
    # tokens to a context-local sink. They are also almost entirely provider wait, so running them
    # one at a time makes a k=5 suite take about an hour. Results are stored by position, so the
    # report does not depend on the order in which they finish.
    limit = asyncio.Semaphore(max(1, concurrency))

    async def run_one(position: int, case: ExpandedCase, repetition: int) -> None:
        nonlocal completed_runs
        async with limit:
            observed = await run_case(
                case,
                llm=llm,
                input_cost_per_million=input_cost_per_million,
                output_cost_per_million=output_cost_per_million,
                cached_cost_per_million=cached_cost_per_million,
            )
        passed = evaluate_case(case, observed).passed
        results[position] = (observed, passed)
        completed_runs += 1
        if progress is not None:
            progress(completed_runs, total_runs, case.id, repetition, k, passed)

    try:
        async with asyncio.TaskGroup() as group:
            for position, (case, repetition) in enumerate(schedule):
                group.create_task(run_one(position, case, repetition))
    finally:
        if llm is not None:
            await llm.aclose()

    # TaskGroup either fills every position or raises, so no result is missing here.
    completed = [result for result in results if result is not None]
    run_cases = [case for case, _ in schedule]
    observations = [observed for observed, _ in completed]
    for (case, _), (_, passed) in zip(schedule, completed, strict=True):
        per_case_passes[case.id].append(passed)

    metrics = aggregate_metrics(run_cases, observations)
    if judge_llm is not None:
        metrics = await judge_quality(run_cases, observations, judge_llm, metrics)
    baseline = baseline_f1(suite, dataset)
    if baseline is not None and metrics.tool_selection_f1 < baseline - F1_REGRESSION_TOLERANCE:
        metrics = metrics.model_copy(
            update={"gate_failures": (*metrics.gate_failures, "tool_selection_f1_regression")}
        )
    # §11.1: level A has no model, so the blind suite (language understanding) can only hold it to
    # the architectural guarantees. Live runs of the same suite use every gate.
    gate_profile: Literal["full", "safety"] = (
        "safety" if dataset == "blind" and suite == "level-a" else "full"
    )
    if gate_profile == "safety":
        metrics = metrics.model_copy(
            update={
                "gate_failures": tuple(
                    gate for gate in metrics.gate_failures if gate in SAFETY_GATES
                )
            }
        )
    pass_to_k = sum(all(values) for values in per_case_passes.values())
    return EvalReport(
        gate_profile=gate_profile,
        generated_at=datetime.now(UTC),
        suite=suite,
        dataset=dataset,
        k=k,
        base_cases=len(bases),
        expanded_cases=len(cases),
        runs=len(observations),
        pass_to_k=Rate(numerator=pass_to_k, denominator=len(cases)),
        agent_model=agent_model,
        prompt_fingerprint=_prompt_fingerprint(),
        check_model=check_model,
        judge_model=judge_model if judge_llm is not None else None,
        concurrency=concurrency,
        metrics=metrics,
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Phase-4 behavioral evaluation suite")
    parser.add_argument("--suite", choices=("level-a", "live"), default="level-a")
    parser.add_argument("--dataset", choices=tuple(DATASETS), default="canonical")
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--case")
    parser.add_argument("--cases-path", type=Path)
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--no-report", action="store_true")
    parser.add_argument("--input-cost-per-million", type=float)
    parser.add_argument("--output-cost-per-million", type=float)
    parser.add_argument("--cached-cost-per-million", type=float)
    parser.add_argument(
        "--judge-model",
        help="Scores conversational quality with this model (must differ from the agent's).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help=(
            "Cases to run at once. Runs are independent and almost entirely provider wait, so a "
            "live k=5 suite drops from about an hour to minutes. Stays at 1 by default because "
            "the useful ceiling is the provider's rate limit, not this machine."
        ),
    )
    return parser.parse_args(argv)


def _judge_client(model: str | None) -> OpenAIResponsesLLM | None:
    if not model:
        return None
    settings = get_settings()
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required for --judge-model")
    if model == settings.openai_agent_model:
        raise ValueError("The judge model must differ from OPENAI_AGENT_MODEL")
    return OpenAIResponsesLLM(
        api_key=key,
        model=model,
        max_output_tokens=8000,
        timeout_seconds=120,
        reasoning_effort="minimal" if model.startswith("gpt-5") else None,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    def print_progress(
        completed: int,
        total: int,
        case_id: str,
        repetition: int,
        repetitions: int,
        passed: bool,
    ) -> None:
        status = "PASS" if passed else "FAIL"
        print(
            f"[{completed:>{len(str(total))}}/{total}] {case_id} "
            f"repetición {repetition}/{repetitions}: {status}",
            flush=True,
        )

    async def run() -> EvalReport:
        judge = _judge_client(args.judge_model)
        try:
            return await evaluate(
                suite=args.suite,
                dataset=args.dataset,
                k=args.k,
                cases_path=args.cases_path,
                case_filter=args.case,
                input_cost_per_million=args.input_cost_per_million,
                output_cost_per_million=args.output_cost_per_million,
                cached_cost_per_million=args.cached_cost_per_million,
                judge_llm=judge,
                judge_model=args.judge_model,
                progress=print_progress if args.suite == "live" else None,
                concurrency=args.concurrency,
            )
        finally:
            if judge is not None:
                await judge.aclose()

    report = asyncio.run(run())
    print(render_report(report))
    if not args.no_report:
        path = write_report(report, args.reports_dir)
        print(f"Report: {path}")
    if report.metrics.gate_failures:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
