from __future__ import annotations

import contextlib
import json
import time
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import httpx
from langgraph.checkpoint.memory import InMemorySaver

from app.conversations.store import InMemoryConversationStore
from app.graph.build import build_graph
from app.graph.context import GraphContext, Retriever
from app.graph.persistence import checkpoint_serializer
from app.graph.recorder import TurnRecorder
from app.graph.service import ConversationAgentService
from app.guards.output import OutputValidator
from app.llm.openai_responses import OpenAIResponsesLLM, ProviderUsage
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

# List prices per million tokens (input, cached input, output) of models a run uses besides the
# agent model, which the CLI prices. A usage record of a model without a price leaves the cost
# unreported instead of billing it at the agent's price.
MODEL_PRICES_PER_MILLION: dict[str, tuple[float, float, float]] = {
    "gpt-5-mini": (0.25, 0.025, 2.00),
}


def usage_cost(
    usage: Sequence[ProviderUsage],
    agent_model: str,
    input_cost_per_million: float | None,
    output_cost_per_million: float | None,
    cached_cost_per_million: float | None,
) -> float | None:
    """USD of a run's provider calls. The CLI prices the agent model and MODEL_PRICES_PER_MILLION
    every other one, such as the policy answer check (ADR-011)."""
    if input_cost_per_million is None or output_cost_per_million is None:
        return None
    cached_rate = (
        input_cost_per_million if cached_cost_per_million is None else cached_cost_per_million
    )
    rates = {
        **MODEL_PRICES_PER_MILLION,
        agent_model: (input_cost_per_million, cached_rate, output_cost_per_million),
    }
    if any(item.model not in rates for item in usage):
        return None
    return (
        sum(
            (item.input_tokens - item.cached_tokens) * rates[item.model][0]
            + item.cached_tokens * rates[item.model][1]
            + item.output_tokens * rates[item.model][2]
            for item in usage
        )
        / 1_000_000
    )


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
    """The mock backend in-process, with injected faults.

    Every JSON body the agent received is kept: the independent figure oracle checks the visible
    text against what the backend returned, never against the agent's own state.
    """

    def __init__(self, faults: Sequence[FaultSpec]) -> None:
        self._inner = httpx.ASGITransport(app=mock_app)
        self._faults = tuple(faults)
        self.payloads: list[Any] = []

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
            return await self._forward(request)
        if fault.mode == "timeout_after_commit":
            response = await self._inner.handle_async_request(request)
            await response.aread()
            raise httpx.ReadTimeout("eval timeout after commit", request=request)
        request.headers["X-Mock-Fail"] = fault.mode
        return await self._forward(request)

    async def _forward(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        body = await response.aread()
        # An empty or malformed body (an injected fault) carries nothing the agent could show.
        with contextlib.suppress(ValueError):
            self.payloads.append(json.loads(body))
        return response


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
    transport: FaultTransport

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
    retriever: Retriever | None = None,
    offline_policy_allowed: bool | None = None,
) -> AsyncIterator[AgentSession]:
    # With a model, policy is answered as in production: only what the model judged answerable.
    # Without one (level A), the calibrated extract is the only policy path there is.
    if offline_policy_allowed is None:
        offline_policy_allowed = llm is None
    resolved = setup or SetupSpec()
    await idempotency_store.reset()
    settings = _settings()
    transport = FaultTransport(resolved.faults)
    client = httpx.AsyncClient(transport=transport, base_url="http://mock")
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
        # Case evidence by default; the answerability eval passes the real hybrid retriever.
        retriever=retriever if retriever is not None else EvidenceRetriever(evidence, resolved.now),
        # The production system prompt: the published fingerprint must match what actually ran.
        system_prompt=load_system_prompt(),
        offline_policy_allowed=offline_policy_allowed,
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
            transport=transport,
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
        payloads = tuple(session.transport.payloads)
    usage = llm.usage_records[usage_start:] if isinstance(llm, OpenAIResponsesLLM) else ()
    input_tokens = sum(item.input_tokens for item in usage)
    output_tokens = sum(item.output_tokens for item in usage)
    cached_tokens = sum(item.cached_tokens for item in usage)
    cost = usage_cost(
        usage,
        llm.model if isinstance(llm, OpenAIResponsesLLM) else "",
        input_cost_per_million,
        output_cost_per_million,
        cached_cost_per_million,
    )
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
        backend_payloads=payloads,
    )
