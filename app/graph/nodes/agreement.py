from __future__ import annotations

from dataclasses import dataclass
from uuid import uuid4

from langgraph.runtime import Runtime

from app.graph.context import GraphContext
from app.graph.nodes.hydrate import BusinessRead, read_business_data
from app.graph.state import AgentState, ConfirmationVerdict, ResponsePlan
from app.guards.config import guardrail_config
from app.policy.engine import (
    PAYMENT_METHODS,
    NegotiationProposal,
    evaluar_propuesta,
    medio_pago_permitido,
    opciones_permitidas,
    vencimiento_oferta,
)
from app.tools.client import agreement_idempotency_key
from app.tools.schemas import AgreementDraft, MedioPago, PaymentOption


@dataclass(frozen=True, slots=True)
class FreshOffer:
    read: BusinessRead
    allowed: list[PaymentOption]


async def _fresh_offer(state: AgentState, runtime: Runtime[GraphContext]) -> FreshOffer | None:
    """Re-read customer, debt and options (never the turn cache) and apply the policy (§8.3)."""
    now = runtime.context.clock.now()
    read = await read_business_data(
        runtime.context, customer=True, debt=True, options=True, now=now
    )
    if not read.complete:
        return None
    assert read.customer is not None and read.debt is not None and read.options is not None
    allowed = opciones_permitidas(read.customer, read.debt, read.options, as_of=now)
    return FreshOffer(read=read, allowed=allowed)


def _default_payment_method(option: PaymentOption) -> MedioPago | None:
    """First method in rules.yaml order that admits the option; shown in the confirmation."""
    return next(
        (method for method in PAYMENT_METHODS if medio_pago_permitido(option, method)), None
    )


def _same_terms(draft: AgreementDraft, option: PaymentOption) -> bool:
    return (
        option.opcion_id == draft.opcion_id
        and option.monto_total == draft.monto_total
        and option.cuotas == draft.cuotas
        and option.monto_cuota == draft.monto_cuota
        and option.primer_vencimiento == draft.fecha_primer_vencimiento
    )


def _requested_option(state: AgentState, allowed: list[PaymentOption]) -> PaymentOption | None:
    route = state.get("route_result")
    option_id = route.option_id if route is not None else None
    installments = route.installments if route is not None else None
    return next(
        (
            option
            for option in allowed
            if (option_id is not None and option.opcion_id == option_id)
            or (option_id is None and installments is not None and option.cuotas == installments)
        ),
        None,
    )


async def _new_draft(
    state: AgentState,
    runtime: Runtime[GraphContext],
    offer: FreshOffer,
    option: PaymentOption,
) -> AgreementDraft | None:
    """Freeze a draft only for an option the policy engine accepts right now."""
    assert offer.read.customer is not None and offer.read.debt is not None
    now = runtime.context.clock.now()
    method = _default_payment_method(option)
    expires_at = vencimiento_oferta(option, now)
    if method is None or expires_at <= now:
        return None
    decision = evaluar_propuesta(
        offer.read.customer,
        offer.read.debt,
        NegotiationProposal(opcion_elegida_id=option.opcion_id, medio_pago=method),
        offer.read.options or [],
        as_of=now,
    )
    if decision.decision != "aceptable":
        return None
    return AgreementDraft(
        draft_id=str(uuid4()),
        opcion_id=option.opcion_id,
        monto_total=option.monto_total,
        cuotas=option.cuotas,
        monto_cuota=option.monto_cuota,
        fecha_primer_vencimiento=option.primer_vencimiento,
        medio_pago=method,
        debt_fingerprint=offer.read.fingerprint,
        # min(valid_until, now + policy window): never extended beyond the backend validity.
        expires_at=expires_at,
        policy_refs=option.policy_refs,
    )


def _offer_again(
    offer: FreshOffer, *, template_id: str, extra: dict[str, object] | None = None
) -> dict[str, object]:
    template = template_id if offer.allowed else "no_options"
    return {
        **offer.read.updates,
        **(extra or {}),
        "offered_options": offer.allowed,
        "pending_draft": None,
        "confirmation_other_count": 0,
        "response_plan": ResponsePlan(kind="negotiation", template_id=template),
    }


async def build_draft(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, object]:
    if state.get("agreement_status") == "unknown":
        return {"response_plan": ResponsePlan(kind="error", template_id="write_unknown_pending")}
    offer = await _fresh_offer(state, runtime)
    if offer is None:
        return {"response_plan": ResponsePlan(kind="error", template_id="data_unavailable")}
    if (
        state.get("agreement_status") == "active"
        and state.get("agreement_fingerprint") == offer.read.fingerprint
    ):
        return {
            **offer.read.updates,
            "response_plan": ResponsePlan(kind="result", template_id="agreement_exists"),
        }
    route = state.get("route_result")
    requested = _requested_option(state, offer.allowed)
    if requested is None:
        if route is not None and route.option_id:
            return {
                **offer.read.updates,
                "offered_options": offer.allowed,
                "http_status": 404,
                "response_plan": ResponsePlan(kind="error", template_id="option_not_found"),
            }
        return _offer_again(offer, template_id="options")
    draft = await _new_draft(state, runtime, offer, requested)
    if draft is None:
        return _offer_again(offer, template_id="option_not_allowed")
    runtime.context.recorder.record_event("agreement_draft_created", draft_id=draft.draft_id)
    return {
        **offer.read.updates,
        "offered_options": offer.allowed,
        "pending_draft": draft,
        "confirmation_other_count": 0,
        "response_plan": ResponsePlan(kind="confirmation", template_id="confirmation_question"),
    }


async def confirm_gate(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, object]:
    draft = state.get("pending_draft")
    if not isinstance(draft, AgreementDraft):
        # Anything but a frozen, typed draft is invalid. It is discarded, never rebuilt (INV-4).
        return {
            "pending_draft": None,
            "confirmation_other_count": 0,
            "response_plan": ResponsePlan(kind="error", template_id="draft_invalid"),
        }
    candidate = state.get("confirmation_candidate") or ConfirmationVerdict(verdict="other")
    if state.get("guard_verdict") == "restrict" and candidate.verdict == "yes":
        candidate = ConfirmationVerdict(verdict="other")

    if candidate.verdict == "no":
        runtime.context.recorder.record_event("agreement_draft_cancelled", draft_id=draft.draft_id)
        return {
            "pending_draft": None,
            "confirmation_other_count": 0,
            "response_plan": ResponsePlan(kind="direct", template_id="draft_cancelled"),
        }

    if candidate.verdict == "other":
        count = state.get("confirmation_other_count", 0) + 1
        if count >= guardrail_config().confirmation_other_cancel_after:
            runtime.context.recorder.record_event(
                "agreement_draft_cancelled", draft_id=draft.draft_id, reason="loop_sin_avance"
            )
            return {
                "pending_draft": None,
                "confirmation_other_count": 0,
                "response_plan": ResponsePlan(kind="direct", template_id="draft_cancelled_loop"),
            }
        # The question is answered (read-only) and the same draft is asked again.
        return {"confirmation_other_count": count}

    now = runtime.context.clock.now()
    if draft.expires_at > now:
        return {"confirmation_other_count": 0}

    # yes + expired draft: discard it, re-read and offer again. Never "no", never executed.
    runtime.context.recorder.record_event("agreement_draft_expired", draft_id=draft.draft_id)
    offer = await _fresh_offer(state, runtime)
    if offer is None:
        return {
            "pending_draft": None,
            "confirmation_other_count": 0,
            "response_plan": ResponsePlan(kind="error", template_id="draft_expired"),
        }
    option = next((item for item in offer.allowed if item.opcion_id == draft.opcion_id), None)
    refreshed = await _new_draft(state, runtime, offer, option) if option is not None else None
    if refreshed is None:
        return _offer_again(offer, template_id="draft_expired_options")
    return {
        **offer.read.updates,
        "offered_options": offer.allowed,
        "pending_draft": refreshed,
        "confirmation_other_count": 0,
        "response_plan": ResponsePlan(
            kind="confirmation",
            template_id="confirmation_question",
            facts={"refreshed": True},
        ),
    }


async def execute_agreement(state: AgentState, runtime: Runtime[GraphContext]) -> dict[str, object]:
    context = runtime.context
    draft = state.get("pending_draft")
    if not isinstance(draft, AgreementDraft):
        return {"response_plan": ResponsePlan(kind="error", template_id="draft_invalid")}
    now = context.clock.now()
    if draft.expires_at <= now:
        return {
            "pending_draft": None,
            "response_plan": ResponsePlan(kind="error", template_id="draft_expired"),
        }

    # Fase 3.1 — revalidate the SAME frozen draft against fresh data; never rebuild it.
    offer = await _fresh_offer(state, runtime)
    if offer is None:
        return {"response_plan": ResponsePlan(kind="error", template_id="data_unavailable")}
    assert offer.read.customer is not None and offer.read.debt is not None
    option = next((item for item in offer.allowed if item.opcion_id == draft.opcion_id), None)
    decision = (
        evaluar_propuesta(
            offer.read.customer,
            offer.read.debt,
            NegotiationProposal(opcion_elegida_id=draft.opcion_id, medio_pago=draft.medio_pago),
            offer.read.options or [],
            as_of=now,
        )
        if option is not None
        else None
    )
    if (
        offer.read.fingerprint != draft.debt_fingerprint
        or option is None
        or not _same_terms(draft, option)
        or decision is None
        or decision.decision != "aceptable"
    ):
        context.recorder.record_event("agreement_draft_invalidated", draft_id=draft.draft_id)
        return _offer_again(offer, template_id="draft_invalidated")

    key = agreement_idempotency_key(context.scope.customer_id, draft.draft_id)
    context.recorder.record_tool(
        "create_payment_agreement", draft_id=draft.draft_id, opcion_id=draft.opcion_id
    )
    result = await context.gateway.create_payment_agreement(
        context.scope,
        draft_id=draft.draft_id,
        opcion_id=draft.opcion_id,
        debt_fingerprint=draft.debt_fingerprint,
        medio_pago=draft.medio_pago,
        idempotency_key=key,
    )
    if result.status == "ok" and result.data is not None:
        context.recorder.agreement_writes.append(
            {
                "draft_id": draft.draft_id,
                "opcion_id": draft.opcion_id,
                "monto_total": str(draft.monto_total),
                "agreement_id": result.data.agreement_id,
                "replayed": result.data.replayed,
            }
        )
        context.recorder.record_event(
            "agreement_created", draft_id=draft.draft_id, agreement_id=result.data.agreement_id
        )
        return {
            **offer.read.updates,
            "pending_draft": None,
            "agreement_status": "active",
            "agreement_id": result.data.agreement_id,
            "agreement_fingerprint": draft.debt_fingerprint,
            "response_plan": ResponsePlan(
                kind="result",
                template_id="agreement_created",
                facts={"agreement_id": result.data.agreement_id},
            ),
        }
    if result.status in {"timeout", "upstream_error", "partial"}:
        # The write may have committed. Never report success or failure (INV-9): audit with the
        # key and derive to an operator who can reconcile (§8.3 Fase 3).
        context.recorder.record_event(
            "agreement_outcome_unknown", draft_id=draft.draft_id, idempotency_key=key
        )
        context.recorder.record_tool("request_human", motivo="outcome_de_escritura_desconocido")
        transfer = await context.gateway.transfer_to_human(
            context.scope,
            conversation_id=state.get("conversation_id", "unknown"),
            motivo="outcome_de_escritura_desconocido",
            resumen=(
                "Confirmación de acuerdo con resultado desconocido. "
                f"Draft {draft.draft_id}. Idempotency-Key {key}."
            ),
        )
        return {
            "pending_draft": None,
            "agreement_status": "unknown",
            "unknown_write_key": key,
            "response_plan": ResponsePlan(
                kind="escalate",
                template_id="write_unknown" if transfer.status == "ok" else "write_unknown_offer",
            ),
        }
    return {
        "pending_draft": None,
        "response_plan": ResponsePlan(kind="error", template_id="write_rejected"),
    }
