"""Level-A guardrail evaluation (§10.1.7): deterministic layers only, numerator/denominator.

Inputs run through the real preflight and deterministic rules with NO classifier verdict: the
probabilistic classifier is measured at level B (F4), never simulated here. Outputs run through
``validate_candidate``, the same full check ``render_and_validate`` applies (validator, citations
and high-risk grounding), against the CUST-00125 fixture state.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from app.graph.nodes.respond import validate_candidate
from app.graph.state import AgentState, ResponsePlan
from app.guards.injection import GuardModelResult, evaluate_rules, resolve_guard
from app.guards.output import OutputValidator
from app.guards.preflight import PreflightPolicy, preflight_message
from app.guards.untrusted import spotlight, summary_is_safe
from app.policy.engine import opciones_permitidas
from app.rag.corpus import load_corpus
from app.rag.models import SearchHit
from app.tools.schemas import Customer, Debt, PaymentOption

ROOT = Path(__file__).parents[2]
FIXTURES = ROOT / "mock_api" / "fixtures"
EVALUATION_NOW = datetime(2026, 9, 12, 15, 0, tzinfo=UTC)
SESSION_CUSTOMER = "CUST-00125"

InputCategory = Literal["direct", "obfuscated", "indirect", "exfiltration", "benign"]
OutputCategory = Literal["violation", "correct"]


class InputCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: InputCategory
    surface: Literal["user", "kb_chunk", "backend_field", "summary"] = "user"
    text: str


class OutputCase(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    case_id: str
    category: OutputCategory
    kind: str
    text: str
    plan_kind: Literal["direct", "policy"] = "direct"
    risk: Literal["low", "high"] = "low"
    retrieved: tuple[str, ...] = ()
    claims: tuple[dict[str, str], ...] = ()
    allowlisted_contacts: tuple[str, ...] = ()


class GuardrailDataset(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    split: Literal["dev", "test"]
    inputs: tuple[InputCase, ...]
    outputs: tuple[OutputCase, ...]


class Rate(BaseModel):
    model_config = ConfigDict(frozen=True)

    numerator: int
    denominator: int
    upper_95: float = Field(description="One-sided 95% Clopper-Pearson upper bound")

    @property
    def value(self) -> float:
        return self.numerator / self.denominator if self.denominator else 0.0


class GuardrailMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    split: str
    injection_detection: Rate
    benign_deflect: Rate
    benign_restrict: Rate
    output_violation_escape: Rate
    output_false_block: Rate
    missed_attacks: tuple[str, ...]
    escaped_outputs: tuple[str, ...]
    blocked_correct_outputs: tuple[str, ...]
    input_categories: dict[str, int]
    output_kinds: dict[str, int]


def clopper_pearson_upper(numerator: int, denominator: int, confidence: float = 0.95) -> float:
    """Exact one-sided upper bound: the largest p with P(X <= k | n, p) >= 1 - confidence."""
    if denominator == 0:
        return 1.0
    if numerator >= denominator:
        return 1.0
    alpha = 1 - confidence

    def cdf(p: float) -> float:
        return sum(
            math.comb(denominator, k) * p**k * (1 - p) ** (denominator - k)
            for k in range(numerator + 1)
        )

    low, high = numerator / denominator, 1.0
    for _ in range(60):
        middle = (low + high) / 2
        if cdf(middle) > alpha:
            low = middle
        else:
            high = middle
    return high


def _rate(numerator: int, denominator: int) -> Rate:
    return Rate(
        numerator=numerator,
        denominator=denominator,
        upper_95=clopper_pearson_upper(numerator, denominator),
    )


def load_guardrail_dataset(path: Path) -> GuardrailDataset:
    return GuardrailDataset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def input_verdict(case: InputCase) -> str:
    """Deterministic verdict for one attack surface (no classifier verdict)."""
    if case.surface == "summary":
        return "allow" if summary_is_safe(case.text) else "restrict"
    if case.surface in {"kb_chunk", "backend_field"}:
        # Retrieved text is data: it counts as contained when the rules flag it or its
        # delimiters had to be neutralized before reaching the prompt.
        wrapped = spotlight("DATOS", "case", case.text)
        preflight = preflight_message(case.text, policy=PreflightPolicy())
        flagged = evaluate_rules(preflight.detection_text).restrict
        return "restrict" if flagged or "[DELIMITADOR_NEUTRALIZADO]" in wrapped else "allow"
    preflight = preflight_message(case.text, policy=PreflightPolicy())
    rules = evaluate_rules(
        preflight.detection_text, preflight.result.flags, session_customer_id=SESSION_CUSTOMER
    )
    return resolve_guard(rules, GuardModelResult()).verdict


def evaluation_state(retrieved_sections: tuple[str, ...] = ()) -> AgentState:
    customers = json.loads((FIXTURES / "customers.json").read_text(encoding="utf-8"))
    customer = Customer.model_validate_json(
        json.dumps(next(item for item in customers if item["customer_id"] == SESSION_CUSTOMER))
    )
    debts = json.loads((FIXTURES / "debts.json").read_text(encoding="utf-8"))
    debt = Debt.model_validate_json(
        json.dumps({"customer_id": SESSION_CUSTOMER, **debts[SESSION_CUSTOMER]})
    )
    raw_options = json.loads((FIXTURES / "options.json").read_text(encoding="utf-8"))
    backend = TypeAdapter(list[PaymentOption]).validate_json(
        json.dumps(raw_options[SESSION_CUSTOMER])
    )
    chunks = {chunk.section_id: chunk for chunk in load_corpus(effective_on=date(2026, 9, 12))}
    hits = [
        SearchHit(
            chunk=chunks[section],
            lexical_score=1.0,
            dense_score=0.9,
            lexical_rank=1,
            dense_rank=1,
            rrf_score=1.0,
        )
        for section in retrieved_sections
    ]
    return {
        "customer_id": SESSION_CUSTOMER,
        "customer": customer,
        "debt": debt,
        "offered_options": opciones_permitidas(customer, debt, backend, as_of=EVALUATION_NOW),
        "retrieved": hits,
    }


def output_flags(case: OutputCase) -> tuple[str, ...]:
    from app.guards.grounding import GroundedReply

    state = evaluation_state(case.retrieved)
    plan = ResponsePlan(
        kind=case.plan_kind,
        template_id=None,
        generation="grounded_policy_reply"
        if case.risk == "high"
        else ("policy_reply" if case.plan_kind == "policy" else "debt_reply"),
        risk=case.risk,
    )
    claims: tuple[Any, ...] = (
        GroundedReply.model_validate({"text": case.text, "claims": list(case.claims)}).claims
        if case.claims
        else ()
    )
    validator = OutputValidator(contact_allowlist=case.allowlisted_contacts)
    return validate_candidate(
        case.text,
        plan,
        state,
        validator,
        claims=claims,
        policy_content=case.plan_kind == "policy",
    )


def evaluate_guardrails(dataset: GuardrailDataset) -> GuardrailMetrics:
    attacks = [case for case in dataset.inputs if case.category != "benign"]
    benign = [case for case in dataset.inputs if case.category == "benign"]
    attack_verdicts = {case.case_id: input_verdict(case) for case in attacks}
    benign_verdicts = [input_verdict(case) for case in benign]
    violations = [case for case in dataset.outputs if case.category == "violation"]
    correct = [case for case in dataset.outputs if case.category == "correct"]
    escaped = tuple(case.case_id for case in violations if not output_flags(case))
    blocked = tuple(case.case_id for case in correct if output_flags(case))
    missed = tuple(case_id for case_id, verdict in attack_verdicts.items() if verdict == "allow")
    return GuardrailMetrics(
        split=dataset.split,
        injection_detection=_rate(len(attacks) - len(missed), len(attacks)),
        benign_deflect=_rate(benign_verdicts.count("deflect"), len(benign)),
        benign_restrict=_rate(benign_verdicts.count("restrict"), len(benign)),
        output_violation_escape=_rate(len(escaped), len(violations)),
        output_false_block=_rate(len(blocked), len(correct)),
        missed_attacks=missed,
        escaped_outputs=escaped,
        blocked_correct_outputs=blocked,
        input_categories=dict(Counter(case.category for case in dataset.inputs)),
        output_kinds=dict(Counter(case.kind for case in dataset.outputs)),
    )
