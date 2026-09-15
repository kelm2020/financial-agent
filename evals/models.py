from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

# Binary judge criteria (ADR-010). The four general ones apply to every response; the
# conditional ones apply only when a case lists them in `expect.judge_rubric`.
JudgeCriterion = Literal[
    "responde_lo_pedido",
    "proximo_paso",
    "tono_adecuado",
    "claridad",
    "reconoce_vulnerabilidad",
]
GENERAL_CRITERIA: tuple[JudgeCriterion, ...] = (
    "responde_lo_pedido",
    "proximo_paso",
    "tono_adecuado",
    "claridad",
)
CONDITIONAL_CRITERIA: tuple[JudgeCriterion, ...] = ("reconoce_vulnerabilidad",)
ALL_CRITERIA: tuple[JudgeCriterion, ...] = GENERAL_CRITERIA + CONDITIONAL_CRITERIA


class EvalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class TurnSpec(EvalModel):
    user: str = Field(min_length=1)
    advance_seconds: int = Field(default=0, ge=0)
    expect_contains: tuple[str, ...] = ()
    expect_not_contains: tuple[str, ...] = ()


class FaultSpec(EvalModel):
    method: Literal["GET", "POST"]
    path_prefix: str
    mode: Literal["timeout", "timeout_after_commit", "500", "partial", "malformed"]


class SetupSpec(EvalModel):
    now: datetime = datetime(2026, 9, 12, 15, 0, tzinfo=UTC)
    faults: tuple[FaultSpec, ...] = ()
    max_tool_calls: int = Field(default=4, ge=0)
    # The production budget (app/main.py, ADR-011).
    max_llm_calls: int = Field(default=4, ge=0)

    @model_validator(mode="after")
    def aware_clock(self) -> SetupSpec:
        if self.now.tzinfo is None or self.now.utcoffset() is None:
            raise ValueError("setup.now must be timezone-aware")
        return self


class ExpectSpec(EvalModel):
    expected_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    must_escalate: bool | None = None
    escalation_reason: str | None = None
    must_create_agreement: bool | None = None
    agreement_option_id: str | None = None
    response_contains: tuple[str, ...] = ()
    response_not_contains: tuple[str, ...] = ()
    requires_citation: bool = False
    policy_case: bool = False
    unsafe_action_opportunity: bool = False
    expected_trajectory: tuple[str, ...] = ()
    trajectory_mode: Literal["unordered", "strict"] = "unordered"
    judge_rubric: tuple[JudgeCriterion, ...] = ()
    # False when several tool paths are equally safe (e.g. clarify vs. list options): only
    # forbidden tools are asserted and the case does not count towards tool_selection_f1.
    check_tool_set: bool = True

    @property
    def judge_criteria(self) -> tuple[JudgeCriterion, ...]:
        extra = tuple(item for item in self.judge_rubric if item not in GENERAL_CRITERIA)
        return GENERAL_CRITERIA + tuple(dict.fromkeys(extra))


class ExpectPatch(EvalModel):
    expected_tools: tuple[str, ...] | None = None
    forbidden_tools: tuple[str, ...] | None = None
    must_escalate: bool | None = None
    escalation_reason: str | None = None
    must_create_agreement: bool | None = None
    agreement_option_id: str | None = None
    response_contains: tuple[str, ...] | None = None
    response_not_contains: tuple[str, ...] | None = None
    requires_citation: bool | None = None
    policy_case: bool | None = None
    unsafe_action_opportunity: bool | None = None
    expected_trajectory: tuple[str, ...] | None = None
    trajectory_mode: Literal["unordered", "strict"] | None = None


class VariantValue(EvalModel):
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    turn_text: dict[int, str] = Field(default_factory=dict)
    advance_before: dict[int, int] = Field(default_factory=dict)
    drop_turns: tuple[int, ...] = ()
    situation: str | None = None
    expect: ExpectPatch | None = None


class VariantAxis(EvalModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    values: tuple[VariantValue, ...] = Field(min_length=1)


class CaseSpec(EvalModel):
    id: str = Field(pattern=r"^[A-Z]-\d{2}$")
    title: str
    category: Literal[
        "consulta",
        "negociacion",
        "accion",
        "ambiguedad",
        "fuera",
        "escalamiento",
        "robustez",
    ]
    customer_id: str = Field(pattern=r"^CUST-\d{5}$")
    # Neutral description for judges and labelers: what is going on, never the expected text.
    situation: str = ""
    turns: tuple[TurnSpec, ...] = Field(min_length=1)
    evidence: tuple[str, ...] = ()
    setup: SetupSpec = SetupSpec()
    expect: ExpectSpec
    variant_axes: tuple[VariantAxis, ...] = ()


class CaseFile(EvalModel):
    cases: tuple[CaseSpec, ...]


class ExpandedCase(EvalModel):
    id: str
    base_id: str
    title: str
    category: str
    customer_id: str
    situation: str = ""
    turns: tuple[TurnSpec, ...]
    evidence: tuple[str, ...]
    setup: SetupSpec
    expect: ExpectSpec


class ObservedTool(EvalModel):
    name: str
    arguments: dict[str, Any]


class TurnObservation(EvalModel):
    text: str
    http_status: int
    state: dict[str, Any]
    tools: tuple[ObservedTool, ...]
    events: tuple[dict[str, Any], ...]
    trajectory: tuple[str, ...]
    latency_ms: float = Field(ge=0)
    llm_latency_ms: float = Field(ge=0)
    llm_tasks: tuple[str, ...] = ()

    @property
    def model_authored(self) -> bool:
        """The visible text was written by the model (and validated), not by a template."""
        return any(task in {"response", "grounded_response"} for task in self.llm_tasks)


class CaseObservation(EvalModel):
    case_id: str
    turns: tuple[TurnObservation, ...]
    agreement_writes: tuple[dict[str, Any], ...]
    final_agreement: dict[str, Any] | None
    total_latency_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    cost_usd: float | None = Field(default=None, ge=0)
    # Every JSON body the backend returned in the conversation, for the independent figure oracle.
    backend_payloads: tuple[Any, ...] = ()


class Rate(EvalModel):
    numerator: int = Field(ge=0)
    denominator: int = Field(ge=0)

    @property
    def value(self) -> float:
        return self.numerator / self.denominator if self.denominator else 0.0


class CaseResult(EvalModel):
    case_id: str
    passed: bool
    failures: tuple[str, ...]
    # Per turn, only for failed runs: guard verdict, route, classifier signal, model calls and
    # notable events, so a flaky live failure can be diagnosed from the report alone.
    diagnostics: tuple[str, ...] = ()
    # Responses showing a figure outside the allowed set, counted once each whether the output
    # validator, the independent figure oracle or both caught it.
    hallucinated_turns: int = Field(default=0, ge=0)


class EvalMetrics(EvalModel):
    cases_passed: Rate
    tool_selection_f1: float = Field(ge=0, le=1)
    valid_tool_args: Rate
    grounded_answers: Rate
    # Turns where the model wrote a policy answer that passed validation, over turns where it
    # tried. A low rate means the answers customers see are fallback extracts, not the model.
    model_answers_accepted: Rate = Field(default_factory=lambda: Rate(numerator=0, denominator=0))
    hallucinated_numbers: Rate
    policy_compliance: Rate
    unsafe_auto_action: Rate
    confirmation_bypass: Rate
    escalation_recall: Rate
    escalation_precision: Rate
    trajectory_match: Rate
    quality_judged: Rate
    # Responses the judge could not score after one retry (provider error or truncated output).
    # They are left out of quality_judged instead of aborting a long live run.
    quality_unjudged: int = Field(default=0, ge=0)
    quality_by_criterion: dict[str, Rate] = Field(default_factory=dict)
    p95_turn_latency_ms: float = Field(ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    total_cost_usd: float | None = Field(default=None, ge=0)
    results: tuple[CaseResult, ...]
    gate_failures: tuple[str, ...]


class EvalReport(EvalModel):
    generated_at: datetime
    suite: str
    dataset: str = "canonical"
    k: int = Field(ge=1)
    base_cases: int = Field(ge=0)
    expanded_cases: int = Field(ge=0)
    runs: int = Field(ge=0)
    pass_to_k: Rate
    agent_model: str | None = None
    check_model: str | None = None
    prompt_fingerprint: str | None = None
    judge_model: str | None = None
    gate_profile: Literal["full", "safety"] = "full"
    metrics: EvalMetrics
