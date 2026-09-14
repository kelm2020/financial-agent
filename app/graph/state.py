from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

from app.guards.codes import GuardFlag
from app.guards.injection import GuardModelResult, GuardRuleResult, GuardVerdict
from app.guards.preflight import PreflightResult
from app.rag.models import SearchHit, Topic
from app.tools.schemas import (
    AgreementDraft,
    Customer,
    Debt,
    EscalationMotivo,
    OptionsSnapshot,
    PaymentOption,
)

Intent = Literal[
    "consulta_deuda",
    "consulta_general",
    "negociacion",
    "aceptar_opcion",
    "fuera_de_dominio",
    "pedido_humano",
    "saludo_despedida",
    # A short "no" to the next step the last reply offered ("¿Querés que veamos alternativas?").
    "rechaza_oferta",
    "ambiguo",
]


class RouteResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: Intent
    option_id: str | None = None
    installments: int | None = None
    topic: Topic = "any"
    escalation_motivo: EscalationMotivo | None = None
    # Monthly amount the customer said they can pay, in answer to "¿cuánto podrías pagar?".
    monthly_amount: int | None = None


class ConfirmationVerdict(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: Literal["yes", "no", "other"]


class GeneratedReply(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str


# Debt figures and low-risk policy answers are deterministic (templates and verified extracts, D7);
# the model only writes high-risk policy answers, and every sentence carries a verified quote.
Generation = Literal["grounded_policy_reply"]


class ResponsePlan(BaseModel):
    """Response intent and allowed facts. Never user-facing text (§4.1, §10.1.4).

    Model text is produced, validated and discarded inside ``render_and_validate``; only the
    validated result reaches ``messages``. Keeping text out of this model keeps an unvalidated
    candidate out of the state and of every checkpoint.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal[
        "direct",
        "policy",
        "negotiation",
        "confirmation",
        "result",
        "deflect",
        "escalate",
        "error",
    ]
    template_id: str | None = None
    generation: Generation | None = None
    facts: dict[str, Any] = Field(default_factory=dict)
    cited_section_ids: tuple[str, ...] = ()
    risk: Literal["low", "high"] = "low"
    followup: Literal["confirmation"] | None = None


AgreementStatus = Literal["none", "active", "unknown"]


class AgentState(TypedDict, total=False):
    conversation_id: str
    customer_id: str
    channel: Literal["chat", "voice"]
    messages: Annotated[list[AnyMessage], add_messages]
    turn_index: int
    conversation_summary: str
    summary_pending: list[str]
    turns_since_summary: int
    last_user_text: str
    detection_text: str
    preflight_result: PreflightResult

    customer: Customer
    debt: Debt
    debt_status: Literal["ok", "not_found", "unavailable"]
    debt_fingerprint: str
    debt_fetched_at: datetime
    options_snapshot: OptionsSnapshot
    offered_options: list[PaymentOption]
    pending_draft: AgreementDraft | None
    confirmation_other_count: int
    confirmation_event_id: str

    guard_rule_result: GuardRuleResult | None
    guard_model_result: GuardModelResult | None
    route_result: RouteResult | None
    confirmation_candidate: ConfirmationVerdict | None
    guard_verdict: GuardVerdict | None
    guard_flags: list[GuardFlag]
    deflect_count: int

    retrieved: list[SearchHit]
    response_plan: ResponsePlan | None
    agreement_status: AgreementStatus
    agreement_id: str
    agreement_fingerprint: str
    # Terms of the agreement registered in this conversation, to answer questions about the plan.
    active_agreement: AgreementDraft | None
    # Reason of the last successful transfer. While a person owns the case the agent keeps
    # answering questions but no longer negotiates (ESC-001).
    handoff_motivo: str
    # The single option the last reply proposed ("¿Te sirve esa?"), so a bare "sí, me sirve" or
    # "mejor no" on the next turn refers to it. Cleared on every other turn.
    proposed_option_id: str
    # The next step the last reply offered: "options" or "human". Written, like the proposal, only
    # by render_and_validate and only when the customer actually saw that reply.
    offered_next_step: str
    unknown_write_key: str
    unknown_draft: AgreementDraft | None
    http_status: int
