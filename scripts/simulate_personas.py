from __future__ import annotations

import argparse
import asyncio
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from app.llm.openai_responses import OpenAIResponsesLLM, build_agent_llm
from config.settings import get_settings
from evals.environment import AgentSession, agent_session
from evals.evaluators import unconfirmed_writes
from evals.simulator import SimulationTurn, load_personas, simulate_conversation

REPORTS_DIR = Path("evals/reports")


class PersonaRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    persona_id: str
    outcome: str
    reason: str
    turns: tuple[SimulationTurn, ...]
    escalated: bool
    agreements_written: int
    unconfirmed_writes: int
    expectation_met: bool


class SimulationReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    generated_at: datetime
    agent_model: str
    simulator_model: str
    runs: tuple[PersonaRun, ...]


def _sender(session: AgentSession) -> Callable[[str], Awaitable[str]]:
    async def send(text: str) -> str:
        return (await session.send(text)).text

    return send


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi-turn simulated users (§11.4) against the real agent and mock backend"
    )
    parser.add_argument("--personas", type=Path)
    parser.add_argument("--persona", help="Run a single persona ID")
    parser.add_argument("--model", help="Simulator model; defaults to OPENAI_SIMULATOR_MODEL")
    parser.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    parser.add_argument("--no-report", action="store_true")
    return parser.parse_args(argv)


async def simulate(args: argparse.Namespace) -> SimulationReport:
    personas = [
        persona
        for persona in load_personas(args.personas)
        if args.persona is None or persona.id == args.persona
    ]
    if not personas:
        raise ValueError(f"No persona matched {args.persona!r}")
    settings = get_settings()
    key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
    if not key:
        raise RuntimeError("OPENAI_API_KEY is required to simulate users")
    simulator_model = args.model or settings.openai_simulator_model
    if not simulator_model:
        raise RuntimeError("Set OPENAI_SIMULATOR_MODEL or pass --model")
    if simulator_model == settings.openai_agent_model:
        raise ValueError("The simulator model must differ from OPENAI_AGENT_MODEL")
    agent_llm = build_agent_llm(
        api_key=key,
        model=settings.openai_agent_model,
        check_model=settings.openai_check_model,
    )
    simulator_llm = OpenAIResponsesLLM(
        api_key=key,
        model=simulator_model,
        max_output_tokens=4000,
        timeout_seconds=120,
        reasoning_effort="minimal" if simulator_model.startswith("gpt-5") else None,
    )
    runs: list[PersonaRun] = []
    try:
        for persona in personas:
            async with agent_session(persona.customer_id, llm=agent_llm) as session:
                result = await simulate_conversation(
                    persona, simulator_llm=simulator_llm, send_to_agent=_sender(session)
                )
                escalated = any(
                    tool.name == "request_human"
                    for observation in session.observations
                    for tool in observation.tools
                )
                writes = session.recorder.agreement_writes
                # The gate must have accepted the same draft with the same event (§11.2).
                unconfirmed = unconfirmed_writes(writes, session.recorder.events)
            runs.append(
                PersonaRun(
                    persona_id=persona.id,
                    outcome=result.outcome,
                    reason=result.reason,
                    turns=result.turns,
                    escalated=escalated,
                    agreements_written=len(writes),
                    unconfirmed_writes=unconfirmed,
                    expectation_met=result.outcome == "success"
                    and unconfirmed == 0
                    and (bool(writes) or not persona.expects_agreement)
                    and (
                        escalated
                        if persona.expects_escalation
                        else not escalated or persona.success_on_derivation
                    ),
                )
            )
            print(f"{persona.id}: {result.outcome} · derivó={escalated}", flush=True)
    finally:
        await agent_llm.aclose()
        await simulator_llm.aclose()
    return SimulationReport(
        generated_at=datetime.now(UTC),
        agent_model=settings.openai_agent_model,
        simulator_model=simulator_model,
        runs=tuple(runs),
    )


def render(report: SimulationReport) -> str:
    lines = [
        f"Usuario simulado · agente={report.agent_model} · simulador={report.simulator_model}",
        "Advertencia: mismo proveedor para agente y simulador; úsese como regresión, no como "
        "tasa de éxito real.",
        "",
        "| Persona | Resultado | Turnos | Derivó | Acuerdos | Sin confirmación | Expectativa |",
        "|---|---|---:|---|---:|---:|---|",
    ]
    lines.extend(
        f"| {run.persona_id} | {run.outcome} | {len(run.turns)} | {run.escalated} | "
        f"{run.agreements_written} | {run.unconfirmed_writes} | "
        f"{'OK' if run.expectation_met else 'FALLA'} |"
        for run in report.runs
    )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    report = asyncio.run(simulate(args))
    print(render(report))
    if not args.no_report:
        args.reports_dir.mkdir(parents=True, exist_ok=True)
        stamp = report.generated_at.strftime("%Y%m%dT%H%M%SZ")
        path = args.reports_dir / f"{stamp}-simulation.json"
        path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
        print(f"Report: {path}")
    if any(not run.expectation_met for run in report.runs):
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
