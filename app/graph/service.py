from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any, Protocol

from langchain_core.messages import AIMessage, HumanMessage

from app.conversations.store import ConversationRecord
from app.graph.context import GraphContext, RecordedLLM
from app.graph.recorder import TurnBudgetExceeded, TurnRecorder
from app.guards.output import ValidationContext
from app.guards.preflight import PreflightPolicy, preflight_message
from app.runtime.conversation_coordinator import ConversationRunCoordinator
from app.runtime.rate_limit import SlidingWindowRateLimiter

_LOGGER = logging.getLogger("app.turns")


class ConversationStore(Protocol):
    async def create(
        self,
        customer_id: str,
        *,
        channel: str = "chat",
        conversation_id: str | None = None,
    ) -> ConversationRecord: ...

    async def get_owned(
        self, conversation_id: str, customer_id: str
    ) -> ConversationRecord | None: ...


class ConversationNotFoundError(LookupError):
    pass


@dataclass(frozen=True, slots=True)
class TurnResult:
    text: str
    state: dict[str, Any]
    events: tuple[dict[str, str], ...]
    http_status: int = 200


class ConversationAgentService:
    def __init__(
        self,
        *,
        graph: Any,
        conversations: ConversationStore,
        coordinator: ConversationRunCoordinator,
        preflight_policy: PreflightPolicy | None = None,
        rate_limiter: SlidingWindowRateLimiter | None = None,
    ) -> None:
        self._graph = graph
        self._conversations = conversations
        self._coordinator = coordinator
        self._preflight_policy = preflight_policy or PreflightPolicy()
        self._rate_limiter = rate_limiter

    async def create_conversation(
        self, customer_id: str, *, channel: str = "chat", conversation_id: str | None = None
    ) -> ConversationRecord:
        return await self._conversations.create(
            customer_id, channel=channel, conversation_id=conversation_id
        )

    async def _owned(
        self, conversation_id: str, customer_id: str, recorder: TurnRecorder
    ) -> ConversationRecord:
        record = await self._conversations.get_owned(conversation_id, customer_id)
        recorder.record_event("ownership_checked")
        if record is None:
            raise ConversationNotFoundError(conversation_id)
        return record

    async def send_message(
        self,
        conversation_id: str,
        customer_id: str,
        text: str,
        *,
        context: GraphContext,
    ) -> TurnResult:
        context.recorder.start_turn()
        turn_context = replace(
            context,
            llm=(RecordedLLM(context.llm, context.recorder) if context.llm is not None else None),
            guard_classifier=(
                RecordedLLM(context.guard_classifier, context.recorder)
                if context.guard_classifier is not None
                else None
            ),
        )
        if context.scope.customer_id != customer_id:
            raise ConversationNotFoundError(conversation_id)
        record = await self._owned(conversation_id, customer_id, context.recorder)
        if self._rate_limiter is not None and not await self._rate_limiter.allow(conversation_id):
            return TurnResult(
                text="Hay demasiados mensajes seguidos. Esperá un momento y volvé a intentar.",
                state={},
                events=(),
                http_status=429,
            )
        outcome = preflight_message(text, policy=self._preflight_policy)
        if outcome.result.rejected:
            response = "El mensaje supera el límite permitido. Enviá una versión más breve."
            return TurnResult(text=response, state={}, events=(), http_status=413)
        # Operational log line: identifiers and codes only, never the message text (INV-14).
        log_line = (
            f"turn conversation={conversation_id} characters={len(outcome.sanitized_text)} "
            f"preflight_flags={','.join(outcome.result.flags) or '-'}"
        )
        _LOGGER.info(log_line)
        context.recorder.record_log(log_line)
        events: list[dict[str, str]] = []
        result: dict[str, Any] = {}
        async with self._coordinator.hold(conversation_id):
            context.recorder.record_event("lock_acquired")
            graph_input: dict[str, Any] = {
                "conversation_id": conversation_id,
                "customer_id": customer_id,
                "channel": record.channel,
                # Only the sanitized text ever becomes a HumanMessage or reaches a checkpoint.
                "messages": [HumanMessage(content=outcome.sanitized_text)],
                "last_user_text": outcome.sanitized_text,
                "detection_text": outcome.detection_text,
                "preflight_result": outcome.result,
                # Turn-scoped outputs must not leak forward from the previous checkpoint.
                "guard_rule_result": None,
                "guard_model_result": None,
                "route_result": None,
                "confirmation_candidate": None,
                "guard_verdict": None,
                "guard_flags": [],
                "response_plan": None,
                "retrieved": [],
                "selected_source": "",
                "http_status": 200,
            }
            try:
                async for mode, chunk in self._graph.astream(
                    graph_input,
                    {"configurable": {"thread_id": record.thread_id}},
                    context=turn_context,
                    stream_mode=["custom", "values"],
                ):
                    if mode == "values":
                        result = chunk
                    elif _is_validated_event(chunk):
                        events.append({"event": chunk["event"], "data": chunk["data"]})
            except TurnBudgetExceeded as exc:
                return await self._budget_exhausted(
                    conversation_id, context, kind=exc.kind, limit=exc.limit
                )
        response = next(
            (
                str(message.content)
                for message in reversed(result.get("messages", []))
                if isinstance(message, AIMessage)
            ),
            "",
        )
        return TurnResult(
            text=response,
            state=result,
            events=tuple(events),
            http_status=int(result.get("http_status", 200)),
        )

    @staticmethod
    async def _budget_exhausted(
        conversation_id: str,
        context: GraphContext,
        *,
        kind: str,
        limit: int,
    ) -> TurnResult:
        context.recorder.record_event("turn_budget_exhausted", kind=kind, limit=limit)
        context.recorder.record_tool("request_human", motivo="loop_sin_avance")
        transfer = await context.gateway.transfer_to_human(
            context.scope,
            conversation_id=conversation_id,
            motivo="loop_sin_avance",
            resumen=f"Se agotó el presupuesto seguro del turno ({kind}).",
        )
        text = "Alcancé el límite seguro de operaciones para este turno. " + (
            "Te derivé con un asesor para continuar."
            if transfer.status == "ok"
            else "No pude completar la derivación; probá nuevamente en unos minutos."
        )
        validation = context.output_validator.validate(
            text, ValidationContext(allowed_customer_id=context.scope.customer_id)
        )
        if not validation.valid:
            text = "No pude continuar de forma segura. Te puedo derivar con un asesor."
        return TurnResult(
            text=text,
            state={"http_status": 200},
            events=({"event": "validated_clause", "data": text},),
        )


def _is_validated_event(chunk: object) -> bool:
    """Only custom events emitted by render_and_validate are forwarded (§10.1.5)."""
    return (
        isinstance(chunk, dict)
        and chunk.get("event") in {"validated_clause", "filler"}
        and isinstance(chunk.get("data"), str)
    )
