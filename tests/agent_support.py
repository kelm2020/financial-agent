"""Shared offline fixtures for graph, invariant, guardrail and acceptance tests."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from typing import Any

import httpx
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import ChannelVersions, Checkpoint, CheckpointMetadata
from langgraph.checkpoint.memory import InMemorySaver

from app.conversations.store import ConversationRecord, InMemoryConversationStore
from app.graph.build import build_graph
from app.graph.context import GraphContext
from app.graph.nodes.hydrate import debt_fingerprint
from app.graph.persistence import checkpoint_serializer
from app.graph.recorder import TurnRecorder
from app.graph.service import ConversationAgentService
from app.guards.output import OutputValidator
from app.llm.protocol import LLMClient
from app.policy.engine import vencimiento_oferta
from app.rag.corpus import load_corpus
from app.rag.models import RetrievalResult, SearchHit, Topic
from app.rag.support import AnswerSupportDecision
from app.runtime.clock import FixedClock
from app.runtime.conversation_coordinator import InMemoryConversationRunCoordinator
from app.security.scope import CustomerScope, session_from_token_claims
from app.tools.client import CollectionsGateway
from app.tools.schemas import AgreementDraft, MedioPago
from config.settings import Settings
from mock_api.auth import issue_token
from mock_api.idempotency_store import idempotency_store
from mock_api.main import app as mock_app

# Inside the fixtures' offer window: options are valid until 2026-09-13T23:59-03:00 and the
# first installment (2026-09-20) is 8 days away, within the 5-15 day policy window.
REFERENCE_NOW = datetime(2026, 9, 12, 15, 0, tzinfo=UTC)
AFTER_OFFER_VALIDITY = datetime(2026, 9, 14, 12, 0, tzinfo=UTC)


def offline_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "mock_api_url": "http://mock",
        "openai_api_key": None,
        "cohere_api_key": None,
        **overrides,
    }
    return Settings(**values)


def scope_for(customer_id: str, settings: Settings) -> CustomerScope:
    token = issue_token(customer_id, settings).access_token
    return CustomerScope.from_session(session_from_token_claims(customer_id, token, customer_id))


def auth_headers(customer_id: str, settings: Settings) -> dict[str, str]:
    return {"Authorization": f"Bearer {issue_token(customer_id, settings).access_token}"}


# ------------------------------------------------------------------------ backend faults


@dataclass(frozen=True, slots=True)
class BackendFault:
    """``mode`` is a mock header mode, or ``timeout_after_commit`` (request reaches the
    backend and commits, but the client only ever sees a read timeout)."""

    method: str
    path_prefix: str
    mode: str


class FaultInjectingTransport(httpx.AsyncBaseTransport):
    def __init__(self, faults: Sequence[BackendFault] = ()) -> None:
        self._inner = httpx.ASGITransport(app=mock_app)
        self._faults = tuple(faults)
        self.requests: list[tuple[str, str]] = []

    def _fault_for(self, request: httpx.Request) -> BackendFault | None:
        return next(
            (
                fault
                for fault in self._faults
                if fault.method == request.method and request.url.path.startswith(fault.path_prefix)
            ),
            None,
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append((request.method, request.url.path))
        fault = self._fault_for(request)
        if fault is None:
            return await self._inner.handle_async_request(request)
        if fault.mode == "timeout_after_commit":
            response = await self._inner.handle_async_request(request)
            await response.aread()
            raise httpx.ReadTimeout(
                "simulated timeout after the backend committed", request=request
            )
        request.headers["X-Mock-Fail"] = fault.mode
        return await self._inner.handle_async_request(request)


# ------------------------------------------------------------------------------ retrieval


def corpus_chunk(section_id: str, *, content: str | None = None) -> SearchHit:
    chunk = next(
        chunk
        for chunk in load_corpus(effective_on=date(2026, 9, 12))
        if chunk.section_id == section_id
    )
    if content is not None:
        chunk = chunk.model_copy(update={"content": content})
    return SearchHit(
        chunk=chunk,
        lexical_score=1.0,
        dense_score=0.9,
        lexical_rank=1,
        dense_rank=1,
        rrf_score=1.0,
    )


class StaticRetriever:
    """Deterministic retriever: returns the configured hits for every query."""

    def __init__(self, hits: Sequence[SearchHit]) -> None:
        self._hits = tuple(hits)
        self.calls: list[tuple[str, str]] = []

    def _result(self) -> RetrievalResult:
        if not self._hits:
            return RetrievalResult(status="no_evidence", on_no_evidence="ofrecer_derivacion")
        return RetrievalResult(
            status="ok",
            hits=self._hits,
            source_chunk_ids=tuple(dict.fromkeys(hit.chunk.section_id for hit in self._hits)),
        )

    async def search(
        self, query: str, *, topic: Topic, effective_on: date, limit: int = 4
    ) -> RetrievalResult:
        self.calls.append(("search", topic))
        return self._result()

    async def search_for_generation(
        self, query: str, *, topic: Topic, effective_on: date, limit: int = 4
    ) -> RetrievalResult:
        self.calls.append(("search_for_generation", topic))
        return self._result()


# ---------------------------------------------------------------------------- persistence


class SpyCheckpointer(InMemorySaver):
    """Records every checkpoint access in an ordered log shared with the ownership store."""

    def __init__(self, log: list[tuple[str, str]]) -> None:
        super().__init__(serde=checkpoint_serializer())
        self.log = log

    @staticmethod
    def _thread(config: RunnableConfig) -> str:
        return str(config.get("configurable", {}).get("thread_id", ""))

    async def aget_tuple(self, config: RunnableConfig) -> Any:
        self.log.append(("checkpoint_read", self._thread(config)))
        return await super().aget_tuple(config)

    def get_tuple(self, config: RunnableConfig) -> Any:
        self.log.append(("checkpoint_read", self._thread(config)))
        return super().get_tuple(config)

    async def alist(self, config: RunnableConfig | None, **kwargs: Any) -> AsyncIterator[Any]:
        self.log.append(("checkpoint_read", self._thread(config or {})))
        async for item in super().alist(config, **kwargs):
            yield item

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        self.log.append(("checkpoint_write", self._thread(config)))
        return await super().aput(config, checkpoint, metadata, new_versions)


class SpyConversationStore(InMemoryConversationStore):
    def __init__(self, log: list[tuple[str, str]]) -> None:
        super().__init__()
        self.log = log

    async def get_owned(self, conversation_id: str, customer_id: str) -> ConversationRecord | None:
        record = await super().get_owned(conversation_id, customer_id)
        self.log.append(("ownership_checked", f"conversation:{conversation_id}"))
        return record


# -------------------------------------------------------------------------------- runtime


@dataclass(slots=True)
class AgentRuntime:
    service: ConversationAgentService
    graph: Any
    saver: SpyCheckpointer
    store: SpyConversationStore
    context: GraphContext
    client: httpx.AsyncClient
    transport: FaultInjectingTransport
    log: list[tuple[str, str]] = field(default_factory=list)

    @property
    def recorder(self) -> TurnRecorder:
        return self.context.recorder

    async def seed(self, conversation: ConversationRecord, values: dict[str, Any]) -> None:
        """Test-only state setup, straight through the graph API (never via the service)."""
        await self.graph.aupdate_state(
            {"configurable": {"thread_id": conversation.thread_id}},
            {
                **values,
                "conversation_id": conversation.conversation_id,
                "customer_id": conversation.customer_id,
            },
        )

    async def history(self, conversation: ConversationRecord) -> list[dict[str, Any]]:
        return [
            snapshot.values
            async for snapshot in self.graph.aget_state_history(
                {"configurable": {"thread_id": conversation.thread_id}}
            )
        ]


@asynccontextmanager
async def agent_runtime(
    customer_id: str = "CUST-00125",
    *,
    llm: LLMClient | None = None,
    guard_classifier: LLMClient | None = None,
    retriever: Any | None = None,
    faults: Sequence[BackendFault] = (),
    now: datetime = REFERENCE_NOW,
    validator: OutputValidator | None = None,
    system_prompt: str = "",
) -> AsyncIterator[AgentRuntime]:
    # The in-process mock keeps registered agreements, and they now make the account ineligible.
    await idempotency_store.reset()
    settings = offline_settings()
    log: list[tuple[str, str]] = []
    transport = FaultInjectingTransport(faults)
    client = httpx.AsyncClient(transport=transport, base_url="http://mock")
    saver = SpyCheckpointer(log)
    store = SpyConversationStore(log)
    graph = build_graph(saver)
    service = ConversationAgentService(
        graph=graph,
        conversations=store,
        coordinator=InMemoryConversationRunCoordinator(timeout_seconds=1),
    )
    context = GraphContext(
        scope=scope_for(customer_id, settings),
        gateway=CollectionsGateway(client=client, settings=settings),
        clock=FixedClock(now),
        recorder=TurnRecorder(),
        output_validator=validator or OutputValidator(contact_allowlist=()),
        llm=llm,
        guard_classifier=guard_classifier,
        retriever=retriever,
        system_prompt=system_prompt,
    )
    try:
        yield AgentRuntime(
            service=service,
            graph=graph,
            saver=saver,
            store=store,
            context=context,
            client=client,
            transport=transport,
            log=log,
        )
    finally:
        await client.aclose()


async def fixture_draft(
    runtime: AgentRuntime,
    *,
    option_id: str = "OPT-3C",
    draft_id: str = "draft-fixture",
    expires_at: datetime | None = None,
    medio_pago: MedioPago = "debito_automatico",
    monto_total: Decimal | None = None,
) -> AgreementDraft:
    """A frozen draft built from the backend's CURRENT debt and options, like build_draft."""
    context = runtime.context
    debt = await context.gateway.get_debt(context.scope)
    options = await context.gateway.get_payment_options(context.scope)
    assert debt.data is not None and options.data is not None
    option = next(item for item in options.data.opciones if item.opcion_id == option_id)
    now = context.clock.now()
    return AgreementDraft(
        draft_id=draft_id,
        opcion_id=option.opcion_id,
        monto_total=monto_total or option.monto_total,
        cuotas=option.cuotas,
        monto_cuota=option.monto_cuota,
        anticipo=option.anticipo,
        fecha_primer_vencimiento=option.primer_vencimiento,
        medio_pago=medio_pago,
        debt_fingerprint=debt_fingerprint(debt.data),
        expires_at=expires_at or vencimiento_oferta(option, now),
        policy_refs=option.policy_refs,
    )


def expired(now: datetime = REFERENCE_NOW) -> datetime:
    return now - timedelta(seconds=1)


def supported_check(claims: int = 1) -> AnswerSupportDecision:
    """The semantic check approving every claim of a model policy answer (ADR-011)."""
    return AnswerSupportDecision(
        question_asks="la pregunta",
        reply_answers="la pregunta",
        off_topic_claim_indices=(),
        redundant_claim_indices=(),
        answers_question=True,
        supported_claim_indices=tuple(range(claims)),
        unsupported_claim_indices=(),
        unresolved_aspects=(),
        reason="supported",
    )
