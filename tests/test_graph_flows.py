from __future__ import annotations

import httpx
import pytest

from app.main import create_app
from config.settings import Settings
from mock_api.auth import issue_token
from mock_api.idempotency_store import idempotency_store
from mock_api.main import app as mock_app


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
    api = create_app(settings=settings, backend_app=mock_app)
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
    api = create_app(settings=settings, backend_app=mock_app)
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
