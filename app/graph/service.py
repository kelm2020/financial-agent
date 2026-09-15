from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, replace
from typing import Any, Protocol

from langchain_core.messages import AIMessage, HumanMessage

from app.conversations.store import ConversationRecord
from app.graph.context import GraphContext, RecordedLLM
from app.graph.effects import transfer_to_human
from app.graph.recorder import TurnBudgetExceeded, TurnRecorder
from app.guards.output import ValidationContext
from app.guards.preflight import PreflightPolicy, preflight_message
from app.runtime.conversation_coordinator import ConversationBusyError, ConversationRunCoordinator
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


type TurnItem = tuple[str, Any]


@dataclass(slots=True)
class TurnStream:
    """A started turn: its HTTP status and the validated events it emits while it runs (§10.1.5).

    ``finished`` is a turn that ended before emitting anything (rate limit, preflight, a graph
    without output); otherwise events come from the running ``task`` through ``queue``.
    """

    http_status: int
    finished: TurnResult | None = None
    task: asyncio.Task[TurnResult] | None = None
    queue: asyncio.Queue[TurnItem] | None = None
    first: dict[str, str] | None = None

    async def events(self) -> AsyncIterator[dict[str, str]]:
        if self.finished is not None:
            return
        assert self.queue is not None
        if self.first is not None:
            yield self.first
        while True:
            kind, payload = await self.queue.get()
            if kind == "end":
                return
            if kind == "event":
                yield payload

    async def result(self) -> TurnResult:
        if self.finished is not None:
            return self.finished
        assert self.task is not None
        # Shielded: a caller that is cancelled (a client that disconnects) never cancels the turn.
        return await asyncio.shield(self.task)


@dataclass(frozen=True, slots=True)
class _PreparedTurn:
    conversation_id: str
    customer_id: str
    record: ConversationRecord
    context: GraphContext
    turn_context: GraphContext
    outcome: Any


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
        # Strong references: a turn whose reader went away must still run to its end.
        self._running: set[asyncio.Task[TurnResult]] = set()

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
        """Run a turn to its end. The HTTP API streams the same turn through ``start_turn``."""
        turn = await self.start_turn(conversation_id, customer_id, text, context=context)
        return await turn.result()

    async def start_turn(
        self,
        conversation_id: str,
        customer_id: str,
        text: str,
        *,
        context: GraphContext,
        finalizer: Callable[[], Awaitable[None]] | None = None,
    ) -> TurnStream:
        """Start a turn and return as soon as its HTTP status is known (§10.1.5).

        Ownership, the rate limit, preflight and the conversation lock are resolved before the first
        byte, and so is a status the graph sets before rendering (an unknown option). The graph runs
        as a task of its own: validated events reach the caller as they are emitted, and a caller
        that stops reading never cuts a turn, a write or its checkpoint in half. ``finalizer``
        releases per-turn resources exactly once, when the turn has ended.
        """
        try:
            prepared = await self._prepare_turn(conversation_id, customer_id, text, context)
        except BaseException:
            if finalizer is not None:
                await finalizer()
            raise
        if isinstance(prepared, TurnResult):
            if finalizer is not None:
                await finalizer()
            return TurnStream(http_status=prepared.http_status, finished=prepared)
        queue: asyncio.Queue[TurnItem] = asyncio.Queue()
        task = asyncio.create_task(self._run_turn(prepared, queue, finalizer))
        self._running.add(task)
        task.add_done_callback(self._forget_turn)
        kind, payload = await queue.get()
        if kind == "end":
            finished = await task
            return TurnStream(http_status=finished.http_status, finished=finished)
        if kind == "status":
            return TurnStream(http_status=int(payload), task=task, queue=queue)
        return TurnStream(http_status=200, task=task, queue=queue, first=payload)

    async def drain(self) -> None:
        """Wait for the turns still running, so a shutdown never interrupts one."""
        await asyncio.gather(*self._running, return_exceptions=True)

    def _forget_turn(self, task: asyncio.Task[TurnResult]) -> None:
        self._running.discard(task)
        error = None if task.cancelled() else task.exception()
        if error is not None and not isinstance(error, ConversationBusyError):
            # The type only: an exception message can carry customer text (INV-14).
            _LOGGER.error("turn failed error=%s", type(error).__name__)

    async def _prepare_turn(
        self, conversation_id: str, customer_id: str, text: str, context: GraphContext
    ) -> TurnResult | _PreparedTurn:
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
        return _PreparedTurn(
            conversation_id=conversation_id,
            customer_id=customer_id,
            record=record,
            context=context,
            turn_context=turn_context,
            outcome=outcome,
        )

    async def _run_turn(
        self,
        turn: _PreparedTurn,
        queue: asyncio.Queue[TurnItem],
        finalizer: Callable[[], Awaitable[None]] | None,
    ) -> TurnResult:
        try:
            return await self._execute_turn(turn, queue)
        finally:
            try:
                if finalizer is not None:
                    await finalizer()
            finally:
                queue.put_nowait(("end", None))

    async def _execute_turn(
        self, turn: _PreparedTurn, queue: asyncio.Queue[TurnItem]
    ) -> TurnResult:
        context = turn.context
        outcome = turn.outcome
        events: list[dict[str, str]] = []
        result: dict[str, Any] = {}
        status_sent = False
        async with self._coordinator.hold(turn.conversation_id):
            context.recorder.record_event("lock_acquired")
            graph_input: dict[str, Any] = {
                "conversation_id": turn.conversation_id,
                "customer_id": turn.customer_id,
                "channel": turn.record.channel,
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
                    {"configurable": {"thread_id": turn.record.thread_id}},
                    context=turn.turn_context,
                    stream_mode=["custom", "values"],
                ):
                    if mode == "values":
                        result = chunk
                        status = int(chunk.get("http_status", 200))
                        if status != 200 and not events and not status_sent:
                            # Nothing was emitted yet, so the response headers can still carry it.
                            queue.put_nowait(("status", status))
                            status_sent = True
                    elif _is_validated_event(chunk):
                        event = {"event": chunk["event"], "data": chunk["data"]}
                        events.append(event)
                        queue.put_nowait(("event", event))
            except TurnBudgetExceeded as exc:
                exhausted = await self._budget_exhausted(
                    turn.conversation_id, context, kind=exc.kind, limit=exc.limit
                )
                for event in exhausted.events:
                    queue.put_nowait(("event", event))
                return exhausted
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
        transfer = await transfer_to_human(
            context,
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
