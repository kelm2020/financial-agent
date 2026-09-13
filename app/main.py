from __future__ import annotations

import hashlib
import json
import secrets
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from langgraph.checkpoint.memory import InMemorySaver
from psycopg_pool import AsyncConnectionPool
from pydantic import BaseModel, ConfigDict, Field

from app.conversations.store import InMemoryConversationStore, PostgresConversationStore
from app.graph.build import build_graph
from app.graph.context import GraphContext, Retriever
from app.graph.persistence import checkpoint_serializer, postgres_checkpointer
from app.graph.recorder import TurnRecorder
from app.graph.service import ConversationAgentService, ConversationNotFoundError
from app.guards.config import guardrail_config, load_contact_allowlist
from app.guards.output import OutputValidator
from app.guards.preflight import PreflightPolicy
from app.llm.openai_responses import OpenAIResponsesLLM
from app.llm.protocol import LLMClient
from app.rag.factory import build_retriever, embedding_client, reranker_client
from app.rag.store import PgVectorHybridStore
from app.runtime.clock import Clock, SystemClock
from app.runtime.conversation_coordinator import (
    ConversationBusyError,
    InMemoryConversationRunCoordinator,
    PostgresConversationRunCoordinator,
    release_session_advisory_locks,
)
from app.runtime.rate_limit import SlidingWindowRateLimiter
from app.security.scope import CustomerScope, session_from_token_claims
from app.tools.client import CollectionsGateway
from config.settings import Settings, get_settings
from mock_api.auth import TokenClaims, require_claims

SYSTEM_PROMPT_PATH = Path(__file__).parent / "prompts" / "system_v1.md"


def _prompt_canary(settings: Settings) -> str:
    configured = (
        settings.system_prompt_canary.get_secret_value().strip()
        if settings.system_prompt_canary is not None
        else ""
    )
    if configured:
        return configured
    if settings.app_env == "production":
        # Stable across replicas without publishing a default. Deployments should still provide
        # SYSTEM_PROMPT_CANARY explicitly; this derives a deterministic fallback from an existing
        # deployment secret so prompt caching is not fragmented during rollout.
        secret = settings.mock_token_secret.get_secret_value()
        digest = hashlib.sha256(f"prompt-canary|{secret}".encode()).hexdigest()[:24]
        return f"ref-{digest}"
    return f"ref-{secrets.token_hex(12)}"


class CreateConversationBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Voice needs the T0/T1/T2 ladder and DTMF confirmation (INV-15..17, F7). Until then the API
    # refuses the channel instead of running voice turns through chat controls.
    channel: Literal["chat"] = "chat"


class MessageBody(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1)


def _bearer_token(authorization: str) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.casefold() != "bearer" or not token:
        raise HTTPException(status_code=401, detail={"code": "MISSING_TOKEN"})
    return token


def _load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def create_app(
    *,
    settings: Settings | None = None,
    backend_app: Any | None = None,
    backend_transport: httpx.AsyncBaseTransport | None = None,
    use_postgres: bool | None = None,
    llm: LLMClient | None = None,
    retriever: Retriever | None = None,
    clock: Clock | None = None,
) -> FastAPI:
    """Build the API. ``llm``, ``retriever``, ``clock`` and transports exist for offline tests;
    production resolves them from settings."""
    resolved = settings or get_settings()
    guards = guardrail_config()
    postgres_enabled = resolved.app_env == "production" if use_postgres is None else use_postgres
    api_key = resolved.openai_api_key.get_secret_value() if resolved.openai_api_key else ""
    owned_llm = (
        OpenAIResponsesLLM(api_key=api_key, model=resolved.openai_agent_model)
        if llm is None and api_key.strip()
        else None
    )
    api_llm: LLMClient | None = llm or owned_llm
    api_retriever: Retriever | None = retriever
    # Static per deployment (prompt caching, §12.5). Without explicit configuration each process
    # draws its own secret instead of using a value published in the repository.
    canary = _prompt_canary(resolved)
    base_prompt = _load_system_prompt()
    system_prompt = f"{base_prompt}\nReferencia interna de versión: {canary}"
    output_validator = OutputValidator(
        contact_allowlist=load_contact_allowlist(),
        prompt_canary=canary,
        protected_prompt=base_prompt,
    )

    def install_runtime(api: FastAPI, *, saver: Any, conversations: Any, coordinator: Any) -> None:
        graph = build_graph(saver)
        api.state.graph = graph
        api.state.agent_service = ConversationAgentService(
            graph=graph,
            conversations=conversations,
            coordinator=coordinator,
            preflight_policy=PreflightPolicy(max_characters=guards.message_max_characters),
            rate_limiter=SlidingWindowRateLimiter(
                limit=guards.conversation_rate_limit,
                window_seconds=guards.conversation_rate_window_seconds,
            ),
        )

    @asynccontextmanager
    async def lifespan(api: FastAPI) -> AsyncIterator[None]:
        nonlocal api_retriever
        async with AsyncExitStack() as stack:
            if owned_llm is not None:
                stack.push_async_callback(owned_llm.aclose)
            if not postgres_enabled:
                yield
                return
            pool = AsyncConnectionPool(
                resolved.database_url,
                open=False,
                kwargs={"autocommit": True},
                # A pooled connection never goes back holding a session advisory lock.
                reset=release_session_advisory_locks,
            )
            await pool.open(wait=True)
            stack.push_async_callback(pool.close)
            saver = await stack.enter_async_context(postgres_checkpointer(resolved.database_url))
            if api_retriever is None and api_key.strip():
                rag_store = await stack.enter_async_context(
                    PgVectorHybridStore(resolved.database_url)
                )
                embeddings = await stack.enter_async_context(
                    embedding_client(resolved, allow_network=True)
                )
                reranker = None
                if (
                    resolved.cohere_api_key is not None
                    and resolved.cohere_api_key.get_secret_value().strip()
                ):
                    reranker = await stack.enter_async_context(
                        reranker_client(resolved, allow_network=True)
                    )
                api_retriever = build_retriever(rag_store, embeddings, resolved, reranker=reranker)
            install_runtime(
                api,
                saver=saver,
                conversations=PostgresConversationStore(pool),
                coordinator=PostgresConversationRunCoordinator(
                    pool, timeout_seconds=resolved.conversation_lock_timeout_seconds
                ),
            )
            yield

    api = FastAPI(title="Collections Agent", version="0.3.0", lifespan=lifespan)
    if not postgres_enabled:
        install_runtime(
            api,
            saver=InMemorySaver(serde=checkpoint_serializer()),
            conversations=InMemoryConversationStore(),
            coordinator=InMemoryConversationRunCoordinator(
                timeout_seconds=resolved.conversation_lock_timeout_seconds
            ),
        )

    @asynccontextmanager
    async def gateway_for(token: str) -> AsyncIterator[CollectionsGateway]:
        transport = backend_transport or (
            httpx.ASGITransport(app=backend_app) if backend_app is not None else None
        )
        in_process = transport is not None
        client = httpx.AsyncClient(
            transport=transport,
            base_url=("http://mock" if in_process else str(resolved.mock_api_url)),
            timeout=resolved.tool_timeout_seconds,
        )
        try:
            yield CollectionsGateway(client=client, settings=resolved)
        finally:
            await client.aclose()

    @api.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @api.post("/conversations", status_code=201)
    async def create_conversation(
        body: CreateConversationBody,
        claims: Annotated[TokenClaims, Depends(require_claims)],
    ) -> dict[str, str]:
        service: ConversationAgentService = api.state.agent_service
        record = await service.create_conversation(claims.sub, channel=body.channel)
        return {
            "conversation_id": record.conversation_id,
            "message": (
                "Soy un asistente virtual de cobranzas. La conversación se registra para "
                "gestionar tu consulta."
            ),
        }

    @api.post("/conversations/{conversation_id}/messages")
    async def send_message(
        conversation_id: str,
        body: MessageBody,
        authorization: Annotated[str, Header(alias="Authorization")],
        claims: Annotated[TokenClaims, Depends(require_claims)],
    ) -> StreamingResponse:
        token = _bearer_token(authorization)
        scope = CustomerScope.from_session(session_from_token_claims(claims.sub, token, claims.sub))
        service: ConversationAgentService = api.state.agent_service
        async with gateway_for(token) as gateway:
            context = GraphContext(
                scope=scope,
                gateway=gateway,
                clock=clock or SystemClock(),
                recorder=TurnRecorder(),
                output_validator=output_validator,
                llm=api_llm,
                guard_classifier=api_llm,
                retriever=api_retriever,
                system_prompt=system_prompt,
            )
            try:
                result = await service.send_message(
                    conversation_id, claims.sub, body.message, context=context
                )
            except ConversationNotFoundError as exc:
                raise HTTPException(status_code=404, detail={"code": "NOT_FOUND"}) from exc
            except ConversationBusyError as exc:
                raise HTTPException(
                    status_code=409,
                    detail={"code": "CONVERSATION_BUSY"},
                    headers={"Retry-After": "1"},
                ) from exc

        async def event_source() -> AsyncIterator[str]:
            # The service already filtered to render_and_validate's validated custom events.
            for event in result.events:
                yield (
                    f"event: {event['event']}\n"
                    f"data: {json.dumps(event['data'], ensure_ascii=False)}\n\n"
                )
            yield "event: done\ndata: {}\n\n"

        return StreamingResponse(
            event_source(),
            status_code=result.http_status,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return api


app = create_app()
