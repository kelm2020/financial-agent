"""Evaluation integrity: gates that passed by construction now fail on the defect they name."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from app.llm.protocol import ScriptedLLM
from evals.dataset import load_dataset
from evals.environment import FaultTransport, run_case
from evals.evaluators import aggregate_metrics, unconfirmed_writes, unsupported_figures
from evals.judge import JudgeVerdicts
from evals.judge_workflow import calibration_sample, turn_sample
from evals.models import ALL_CRITERIA, GENERAL_CRITERIA, ExpandedCase
from evals.run import evaluate
from evals.variants import expand_cases

REFERENCE = date(2026, 9, 12)
_DEBT: dict[str, Any] = {
    "saldo_total": "184500",
    "recargo_pct": 8,
    "vencimientos": [{"monto": "61500", "vencimiento": "2026-07-10"}],
    "identidad_verificada": True,
}


def _case(case_id: str) -> ExpandedCase:
    return next(
        case
        for case in expand_cases(load_dataset("canonical"))
        if case.id == case_id or case.base_id == case_id
    )


def _verdicts() -> JudgeVerdicts:
    return JudgeVerdicts.model_validate(
        {criterion: {"verdict": "pass", "reason": "prueba"} for criterion in ALL_CRITERIA}
    )


# ------------------------------------------------------------------ independent figure oracle


def test_figure_oracle_accepts_only_backend_figures_and_cited_sections() -> None:
    shown = "Tu saldo es de $184.500, con una cuota de $ 61.500,00 que venció el 10/07/2026."
    assert unsupported_figures(shown, [_DEBT], REFERENCE) == []
    assert unsupported_figures("Un recargo del 8 % del total.", [_DEBT], REFERENCE) == []
    invented = "Pagá $999.999 con 35 % de quita antes del 31/12/2026 o del 31/02/2026."
    assert unsupported_figures(invented, [_DEBT], REFERENCE) == [
        "$999999",
        "35%",
        "0001-01-01",
        "2026-12-31",
    ]
    # A knowledge-base figure is allowed only by the section cited in that same text (§10.1.4).
    assert unsupported_figures("Sí, desde el 10 % del saldo total. [FAQ-001]", [], REFERENCE) == []
    assert unsupported_figures("Sí, desde el 10 % del saldo total.", [], REFERENCE) == ["10%"]


async def test_an_invented_amount_fails_the_gate_once_per_response() -> None:
    case = _case("C-01")
    observation = await run_case(case)
    turn = observation.turns[-1]
    forged = observation.model_copy(
        update={
            "turns": (
                *observation.turns[:-1],
                turn.model_copy(update={"text": f"{turn.text} Te hago una quita de $999.999."}),
            )
        }
    )
    clean = aggregate_metrics([case], [observation]).hallucinated_numbers
    metrics = aggregate_metrics([case], [forged])
    assert clean.numerator == 0
    # Both the validator and the oracle catch it; the response still counts once.
    assert metrics.hallucinated_numbers.numerator == 1
    assert "hallucinated_numbers" in metrics.gate_failures
    assert any("unsupported_figure" in item for item in metrics.results[0].failures)


async def test_fault_transport_keeps_json_bodies_and_skips_the_rest() -> None:
    transport = FaultTransport(())
    async with httpx.AsyncClient(transport=transport, base_url="http://mock") as client:
        await client.get("/health")
        await client.get("/docs")  # HTML, not a payload the agent could show
    assert transport.payloads == [{"status": "ok"}]


# ------------------------------------------------------------------ correlated confirmation


def test_confirmation_bypass_requires_the_same_draft_and_event() -> None:
    write = {"draft_id": "d-1", "confirmation_event_id": "e-1"}
    accepted = {
        "type": "agreement_confirmation_accepted",
        "draft_id": "d-1",
        "confirmation_event_id": "e-1",
    }
    assert unconfirmed_writes([write], [accepted]) == 0
    assert unconfirmed_writes([write], []) == 1
    assert unconfirmed_writes([write], [{**accepted, "draft_id": "d-2"}]) == 1
    assert unconfirmed_writes([{**write, "confirmation_event_id": ""}], [accepted]) == 1


async def test_a_forged_confirmation_id_is_a_bypass() -> None:
    case = _case("A-01__si__vigente")
    observation = await run_case(case)
    assert aggregate_metrics([case], [observation]).confirmation_bypass.numerator == 0
    forged = observation.model_copy(
        update={
            "agreement_writes": tuple(
                {**write, "confirmation_event_id": "forged"}
                for write in observation.agreement_writes
            )
        }
    )
    metrics = aggregate_metrics([case], [forged])
    assert metrics.confirmation_bypass.numerator == len(observation.agreement_writes) >= 1
    assert "confirmation_bypass" in metrics.gate_failures


# ------------------------------------------------------------------------ judge every turn


async def test_every_turn_is_judged_and_conditional_criteria_apply_to_the_final_one() -> None:
    case = _case("A-01__si__vigente")
    observation = await run_case(case)
    turns = len(case.turns)
    assert turns >= 2
    first = turn_sample(case, observation, 0)
    assert first.conversation == "" and first.user == case.turns[0].user
    assert turn_sample(case, observation, turns - 1) == calibration_sample(case, observation)

    rubric = case.expect.model_copy(update={"judge_rubric": ("reconoce_vulnerabilidad",)})
    vulnerable = case.model_copy(update={"expect": rubric})
    assert turn_sample(vulnerable, observation, 0).criteria == GENERAL_CRITERIA
    assert "reconoce_vulnerabilidad" in turn_sample(vulnerable, observation, turns - 1).criteria

    judge = ScriptedLLM([_verdicts() for _ in range(turns)])
    report = await evaluate(case_filter=case.id, judge_llm=judge, judge_model="judge-test")
    assert len(judge.calls) == turns
    assert report.metrics.quality_judged.denominator == turns
