"""The HTTP turn streams validated events while the graph runs (§10.1.5).

A turn is a task of its own: its status is known before the first byte, its events are forwarded as
render_and_validate emits them, and a reader that goes away never cuts the turn in half.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Mapping, Sequence
from typing import Any

import pytest
from pydantic import BaseModel

from app.graph.service import ConversationAgentService, ConversationNotFoundError
from app.guards.grounding import GroundedClaim, GroundedReply
from app.llm.protocol import ScriptedLLM
from app.runtime.conversation_coordinator import (
    ConversationBusyError,
    InMemoryConversationRunCoordinator,
)
from tests.agent_support import StaticRetriever, agent_runtime, corpus_chunk, supported_check

_NO_DATES = "El canal automático no cambia fechas."
_QUESTION = "¿Se puede correr el vencimiento de una cuota?"


class GatedLLM:
    """Holds the policy answer until the test releases it."""

    def __init__(self, responses: Sequence[BaseModel]) -> None:
        self._inner = ScriptedLLM(responses)
        self.generating = asyncio.Event()
        self.release = asyncio.Event()

    async def complete[T: BaseModel](
        self,
        *,
        task: str,
        messages: Sequence[Mapping[str, str]],
        response_model: type[T],
    ) -> T:
        if task == "grounded_response":
            self.generating.set()
            await self.release.wait()
        return await self._inner.complete(
            task=task, messages=messages, response_model=response_model
        )


def _policy_llm() -> GatedLLM:
    claim = GroundedClaim(sentence=_NO_DATES, section_id="FAQ-003", quote=_NO_DATES)
    return GatedLLM([GroundedReply(text=_NO_DATES, claims=(claim,)), supported_check()])


class ValuesOnlyGraph:
    async def astream(self, graph_input: dict[str, Any], *_: Any, **__: Any) -> Any:
        yield "values", graph_input


class FailingGraph:
    async def astream(self, *_: Any, **__: Any) -> Any:
        raise RuntimeError("provider down with customer text")
        yield  # pragma: no cover - makes this an async generator


async def test_the_filler_reaches_the_client_before_the_answer_is_generated() -> None:
    llm = _policy_llm()
    retriever = StaticRetriever([corpus_chunk("FAQ-003")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        turn = await runtime.service.start_turn(
            conversation.conversation_id, "CUST-00125", _QUESTION, context=runtime.context
        )
        events = turn.events()
        first = await anext(events)
        # The filler is out while the answer model is still generating.
        await asyncio.wait_for(llm.generating.wait(), timeout=5)
        assert (turn.http_status, first["event"], llm.release.is_set()) == (200, "filler", False)
        llm.release.set()
        rest = [event async for event in events]
        result = await turn.result()
    assert result.text.endswith("[FAQ-003]")
    assert " ".join(event["data"] for event in rest) == result.text


async def test_a_turn_completes_when_its_reader_stops_reading() -> None:
    llm = _policy_llm()
    retriever = StaticRetriever([corpus_chunk("FAQ-003")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        turn = await runtime.service.start_turn(
            conversation.conversation_id, "CUST-00125", _QUESTION, context=runtime.context
        )
        await anext(turn.events())  # the client reads the filler and disconnects
        llm.release.set()
        await runtime.service.drain()
        result = await turn.result()
        # The lock was released and the answer reached the checkpoint: the next turn runs.
        following = await runtime.service.send_message(
            conversation.conversation_id, "CUST-00125", "¿Cuánto debo?", context=runtime.context
        )
    assert result.text.endswith("[FAQ-003]")
    assert following.http_status == 200 and "$184.500" in following.text


async def test_a_status_the_graph_sets_is_known_before_the_first_event() -> None:
    async with agent_runtime() as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        turn = await runtime.service.start_turn(
            conversation.conversation_id,
            "CUST-00125",
            "Quiero la opción OPT-ZZ99",
            context=runtime.context,
        )
        events = [event async for event in turn.events()]
    assert turn.http_status == 404 and events


async def test_turns_that_never_reach_the_graph_release_their_resources() -> None:
    async with agent_runtime() as runtime:
        coordinator = InMemoryConversationRunCoordinator(timeout_seconds=0.01)
        service = ConversationAgentService(
            graph=runtime.graph, conversations=runtime.store, coordinator=coordinator
        )
        conversation = await service.create_conversation("CUST-00125")
        closed: list[str] = []

        async def finalizer() -> None:
            closed.append("gateway")

        with pytest.raises(ConversationNotFoundError):
            await service.start_turn(
                "missing", "CUST-00125", "hola", context=runtime.context, finalizer=finalizer
            )
        too_long = await service.start_turn(
            conversation.conversation_id,
            "CUST-00125",
            "a" * 50_000,
            context=runtime.context,
            finalizer=finalizer,
        )
        assert too_long.http_status == 413
        assert [event async for event in too_long.events()] == []
        async with coordinator.hold(conversation.conversation_id):
            with pytest.raises(ConversationBusyError):
                await service.start_turn(
                    conversation.conversation_id,
                    "CUST-00125",
                    "hola",
                    context=runtime.context,
                    finalizer=finalizer,
                )
    assert closed == ["gateway", "gateway", "gateway"]


async def test_a_turn_without_output_or_with_a_failure_ends_before_streaming(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async with agent_runtime() as runtime:
        for graph in (ValuesOnlyGraph(), FailingGraph()):
            service = ConversationAgentService(
                graph=graph,
                conversations=runtime.store,
                coordinator=InMemoryConversationRunCoordinator(),
            )
            conversation = await service.create_conversation("CUST-00125")
            if isinstance(graph, ValuesOnlyGraph):
                turn = await service.start_turn(
                    conversation.conversation_id, "CUST-00125", "hola", context=runtime.context
                )
                assert (turn.http_status, (await turn.result()).text) == (200, "")
                continue
            with caplog.at_level(logging.ERROR, logger="app.turns"):
                with pytest.raises(RuntimeError):
                    await service.start_turn(
                        conversation.conversation_id, "CUST-00125", "hola", context=runtime.context
                    )
                await asyncio.sleep(0)
    assert "turn failed error=RuntimeError" in caplog.text
    assert "customer text" not in caplog.text
