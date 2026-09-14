from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import httpx
import pytest
from pydantic import SecretStr

from app.llm.protocol import ScriptedLLM
from app.main import create_app
from app.rag.ingest import ingest_corpus
from app.runtime.clock import FixedClock
from config.settings import Settings
from mock_api.auth import issue_token
from mock_api.idempotency_store import idempotency_store
from mock_api.main import app as mock_app
from tests.agent_support import REFERENCE_NOW
from tests.rag_support import NamedHashingEmbeddingClient


@pytest.fixture(autouse=True)
async def reset_writes() -> None:
    await idempotency_store.reset()


@pytest.fixture
def settings() -> Settings:
    return Settings(mock_api_url="http://mock")


def _headers(customer_id: str, settings: Settings) -> dict[str, str]:
    token = issue_token(customer_id, settings).access_token
    return {"Authorization": f"Bearer {token}"}


async def test_graph_contains_guard_join_and_single_output_boundary(settings: Settings) -> None:
    api = create_app(settings=settings, backend_app=mock_app, clock=FixedClock(REFERENCE_NOW))
    graph = api.state.graph.get_graph()

    assert {
        "guard_rules",
        "guard_classifier",
        "route_or_confirm",
        "resolve_guard",
        "render_and_validate",
    } <= set(graph.nodes)
    assert sum(edge.target == "render_and_validate" for edge in graph.edges) >= 4


async def test_http_two_phase_agreement_and_validated_sse(settings: Settings) -> None:
    api = create_app(settings=settings, backend_app=mock_app, clock=FixedClock(REFERENCE_NOW))
    headers = _headers("CUST-00125", settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api), base_url="http://agent"
    ) as client:
        created = await client.post("/conversations", json={"channel": "chat"}, headers=headers)
        conversation_id = created.json()["conversation_id"]

        proposed = await client.post(
            f"/conversations/{conversation_id}/messages",
            json={"message": "Quiero la opción de 3 cuotas"},
            headers=headers,
        )
        assert proposed.status_code == 200
        assert "event: validated_clause" in proposed.text
        assert "3 cuotas" in proposed.text

        confirmed = await client.post(
            f"/conversations/{conversation_id}/messages",
            json={"message": "sí, confirmo"},
            headers=headers,
        )
        assert confirmed.status_code == 200
        assert "quedó registrado" in confirmed.text
        assert "event: done" in confirmed.text
        assert "event: token" not in confirmed.text


async def test_foreign_conversation_is_404_before_checkpoint(settings: Settings) -> None:
    api = create_app(settings=settings, backend_app=mock_app)
    owner = _headers("CUST-00125", settings)
    foreign = _headers("CUST-00212", settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api), base_url="http://agent"
    ) as client:
        created = await client.post("/conversations", json={}, headers=owner)
        conversation_id = created.json()["conversation_id"]
        response = await client.post(
            f"/conversations/{conversation_id}/messages",
            json={"message": "¿Cuánto debo?"},
            headers=foreign,
        )

    assert response.status_code == 404


async def test_zero_debt_is_not_not_found(settings: Settings) -> None:
    api = create_app(settings=settings, backend_app=mock_app)
    headers = _headers("CUST-00450", settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api), base_url="http://agent"
    ) as client:
        created = await client.post("/conversations", json={}, headers=headers)
        response = await client.post(
            f"/conversations/{created.json()['conversation_id']}/messages",
            json={"message": "¿Cuánto debo?"},
            headers=headers,
        )

    assert "no registrás deuda vigente" in response.text.lower()


async def test_local_mode_builds_an_in_memory_retriever(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    ingested: list[int] = []

    @asynccontextmanager
    async def hashing_embeddings(*args: object, **kwargs: object) -> AsyncIterator[object]:
        yield NamedHashingEmbeddingClient("local-hashing", 8)

    @asynccontextmanager
    async def no_reranker(*args: object, **kwargs: object) -> AsyncIterator[None]:
        yield None

    async def recording_ingest(store: Any, embeddings: Any, *, effective_on: date) -> Any:
        metadata = await ingest_corpus(store, embeddings, effective_on=effective_on)
        ingested.append(metadata.chunk_count)
        return metadata

    monkeypatch.setattr("app.main.embedding_client", hashing_embeddings)
    monkeypatch.setattr("app.main.reranker_client", no_reranker)
    monkeypatch.setattr("app.main.ingest_corpus", recording_ingest)
    keyed = settings.model_copy(
        update={"openai_api_key": SecretStr("sk-local"), "cohere_api_key": SecretStr("co-local")}
    )
    api = create_app(
        settings=keyed,
        backend_app=mock_app,
        llm=ScriptedLLM([]),
        use_postgres=False,
        clock=FixedClock(REFERENCE_NOW),
    )
    async with api.router.lifespan_context(api):
        assert ingested and ingested[0] > 0
