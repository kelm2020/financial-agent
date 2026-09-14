from __future__ import annotations

import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta

import httpx
from langgraph.checkpoint.memory import InMemorySaver

from app.conversations.store import InMemoryConversationStore
from app.graph.build import build_graph
from app.graph.context import GraphContext
from app.graph.persistence import checkpoint_serializer
from app.graph.recorder import TurnRecorder
from app.graph.service import ConversationAgentService
from app.guards.output import OutputValidator
from app.llm.openai_responses import OpenAIResponsesLLM
from app.llm.protocol import LLMClient
from app.prompts import load_system_prompt
from app.rag.corpus import load_corpus
from app.rag.models import RetrievalResult, SearchHit, Topic
from app.runtime.conversation_coordinator import InMemoryConversationRunCoordinator
from app.security.scope import CustomerScope, session_from_token_claims
from app.tools.client import CollectionsGateway
from config.settings import Settings
from evals.models import (
    CaseObservation,
    ExpandedCase,
    FaultSpec,
    ObservedTool,
    SetupSpec,
    TurnObservation,
)
from mock_api.auth import issue_token
from mock_api.idempotency_store import idempotency_store
from mock_api.main import app as mock_app


class MutableClock:
    def __init__(self, value: datetime) -> None:
        self._value = value

    def now(self) -> datetime:
        return self._value

    def advance(self, seconds: int) -> None:
        self._value += timedelta(seconds=seconds)


class EvidenceRetriever:
    def __init__(self, section_ids: Sequence[str], effective_on: datetime) -> None:
        corpus = load_corpus(effective_on=effective_on.date())
        by_section = {chunk.section_id: chunk for chunk in corpus}
        missing = set(section_ids) - set(by_section)
        if missing:
            raise ValueError(f"Unknown evidence sections: {sorted(missing)}")
        self._hits = tuple(
            SearchHit(
                chunk=by_section[section_id],
                lexical_score=1,
                dense_score=1,
                lexical_rank=index,
                dense_rank=index,
                rrf_score=1,
            )
            for index, section_id in enumerate(section_ids, 1)
        )

    async def search(
        self, query: str, *, topic: Topic, effective_on: object, limit: int = 4
    ) -> RetrievalResult:
        del query, topic, effective_on, limit
        return self._result()

    async def search_for_generation(
        self, query: str, *, topic: Topic, effective_on: object, limit: int = 4
    ) -> RetrievalResult:
        del query, topic, effective_on, limit
        return self._result()

    def _result(self) -> RetrievalResult:
        if not self._hits:
            return RetrievalResult(status="no_evidence", on_no_evidence="ofrecer_derivacion")
        return RetrievalResult(
            status="ok",
            hits=self._hits,
            source_chunk_ids=tuple(hit.chunk.section_id for hit in self._hits),
        )


class FaultTransport(httpx.AsyncBaseTransport):
    def __init__(self, faults: Sequence[FaultSpec]) -> None:
        self._inner = httpx.ASGITransport(app=mock_app)
        self._faults = tuple(faults)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        fault = next(
            (
                item
                for item in self._faults
                if item.method == request.method and request.url.path.startswith(item.path_prefix)
            ),
            None,
        )
        if fault is None:
            return await self._inner.handle_async_request(request)
        if fault.mode == "timeout_after_commit":
            response = await self._inner.handle_async_request(request)
            await response.aread()
            raise httpx.ReadTimeout("eval timeout after commit", request=request)
        request.headers["X-Mock-Fail"] = fault.mode
        return await self._inner.handle_async_request(request)


def _settings() -> Settings:
    return Settings(
        mock_api_url="http://mock",
        openai_api_key=None,
        cohere_api_key=None,
    )


def _scope(customer_id: str, settings: Settings) -> CustomerScope:
    token = issue_token(customer_id, settings).access_token
    session = session_from_token_claims(customer_id, token, customer_id)
    return CustomerScope.from_session(session)


@dataclass(slots=True)
class AgentSession:
    """One real conversation against the graph, policies and mock backend of the evaluation."""

    service: ConversationAgentService
    context: GraphContext
    recorder: TurnRecorder
    clock: MutableClock
    conversation_id: str
    customer_id: str
    observations: list[TurnObservation]

    async def send(self, text: str, *, advance_seconds: int = 0) -> TurnObservation:
        self.clock.advance(advance_seconds)
        tool_start = len(self.recorder.tool_calls)
        event_start = len(self.recorder.events)
        trajectory_start = len(self.recorder.trajectory)
        llm_start = len(self.recorder.llm_calls)
        started = time.perf_counter()
        result = await self.service.send_message(
            self.conversation_id, self.customer_id, text, context=self.context
        )
        latency = (time.perf_counter() - started) * 1000
        llm_calls = self.recorder.llm_calls[llm_start:]
        observation = TurnObservation(
            text=result.text,
            http_status=result.http_status,
            state=result.state,
            tools=tuple(
                ObservedTool(name=item.name, arguments=item.arguments)
                for item in self.recorder.tool_calls[tool_start:]
            ),
            events=tuple(self.recorder.events[event_start:]),
            trajectory=tuple(self.recorder.trajectory[trajectory_start:]),
            latency_ms=latency,
            llm_latency_ms=sum(item.latency_ms for item in llm_calls),
            llm_tasks=tuple(item.task for item in llm_calls),
        )
        self.observations.append(observation)
        return observation


@asynccontextmanager
async def agent_session(
    customer_id: str,
    *,
    llm: LLMClient | None = None,
    evidence: Sequence[str] = (),
    setup: SetupSpec | None = None,
) -> AsyncIterator[AgentSession]:
    resolved = setup or SetupSpec()
    await idempotency_store.reset()
    settings = _settings()
    client = httpx.AsyncClient(transport=FaultTransport(resolved.faults), base_url="http://mock")
    recorder = TurnRecorder(
        max_tool_calls=resolved.max_tool_calls, max_llm_calls=resolved.max_llm_calls
    )
    clock = MutableClock(resolved.now)
    service = ConversationAgentService(
        graph=build_graph(InMemorySaver(serde=checkpoint_serializer())),
        conversations=InMemoryConversationStore(),
        coordinator=InMemoryConversationRunCoordinator(timeout_seconds=1),
    )
    context = GraphContext(
        scope=_scope(customer_id, settings),
        gateway=CollectionsGateway(client=client, settings=settings),
        clock=clock,
        recorder=recorder,
        output_validator=OutputValidator(contact_allowlist=()),
        llm=llm,
        guard_classifier=llm,
        retriever=EvidenceRetriever(evidence, resolved.now),
        # The production system prompt: the published fingerprint must match what actually ran.
        system_prompt=load_system_prompt(),
    )
    try:
        conversation = await service.create_conversation(customer_id)
        yield AgentSession(
            service=service,
            context=context,
            recorder=recorder,
            clock=clock,
            conversation_id=conversation.conversation_id,
            customer_id=customer_id,
            observations=[],
        )
    finally:
        await client.aclose()


def final_agreement(session: AgentSession) -> dict[str, object] | None:
    final_state = session.observations[-1].state if session.observations else {}
    fingerprint = final_state.get("agreement_fingerprint") or final_state.get("debt_fingerprint")
    if not fingerprint:
        return None
    return idempotency_store.active_agreement(session.customer_id, str(fingerprint))


async def run_case(
    case: ExpandedCase,
    *,
    llm: LLMClient | None = None,
    input_cost_per_million: float | None = None,
    output_cost_per_million: float | None = None,
    cached_cost_per_million: float | None = None,
) -> CaseObservation:
    usage_start = len(llm.usage_records) if isinstance(llm, OpenAIResponsesLLM) else 0
    async with agent_session(
        case.customer_id, llm=llm, evidence=case.evidence, setup=case.setup
    ) as session:
        for turn in case.turns:
            await session.send(turn.user, advance_seconds=turn.advance_seconds)
        agreement = final_agreement(session)
        observations = tuple(session.observations)
        writes = tuple(session.recorder.agreement_writes)
    usage = llm.usage_records[usage_start:] if isinstance(llm, OpenAIResponsesLLM) else ()
    input_tokens = sum(item.input_tokens for item in usage)
    output_tokens = sum(item.output_tokens for item in usage)
    cached_tokens = sum(item.cached_tokens for item in usage)
    cost = None
    if input_cost_per_million is not None and output_cost_per_million is not None:
        cached_rate = (
            cached_cost_per_million
            if cached_cost_per_million is not None
            else input_cost_per_million
        )
        cost = (
            (input_tokens - cached_tokens) * input_cost_per_million
            + cached_tokens * cached_rate
            + output_tokens * output_cost_per_million
        ) / 1_000_000
    return CaseObservation(
        case_id=case.id,
        turns=observations,
        agreement_writes=writes,
        final_agreement=agreement,
        total_latency_ms=sum(item.latency_ms for item in observations),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cached_tokens=cached_tokens,
        cost_usd=cost,
    )
