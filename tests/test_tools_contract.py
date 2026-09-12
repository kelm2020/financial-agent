from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from fastapi import status
from pydantic import BaseModel

from app.security.scope import (
    AuthenticatedSession,
    CustomerScope,
    session_from_token_claims,
)
from app.tools.client import CollectionsGateway, _safe_error_detail, agreement_idempotency_key
from app.tools.registry import ToolPhase, model_tools_for_phase
from app.tools.schemas import MODEL_TOOL_SCHEMAS, AgreementResponse, Customer, Debt
from config.settings import Settings
from mock_api.auth import issue_token
from mock_api.idempotency_store import IdempotencyStore, idempotency_store
from mock_api.main import app
from scripts import initialize_database


@pytest.fixture(autouse=True)
async def reset_idempotency() -> None:
    await idempotency_store.reset()


@pytest.fixture
def settings() -> Settings:
    return Settings(
        mock_api_url="http://test",
        tool_retry_attempts=3,
        tool_timeout_seconds=0.1,
        circuit_breaker_threshold=3,
        circuit_breaker_reset_seconds=30,
    )


def scope_for(customer_id: str, settings: Settings) -> CustomerScope:
    token = issue_token(customer_id, settings).access_token
    session = AuthenticatedSession(
        customer_id=customer_id,
        subject=customer_id,
        downstream_token=token,
        authenticated_at=datetime.now(UTC),
    )
    return CustomerScope.from_session(session)


@pytest.fixture
async def gateway(settings: Settings) -> AsyncIterator[CollectionsGateway]:
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    instance = CollectionsGateway(client=client, settings=settings)
    yield instance
    await client.aclose()


class _HasCustomerId(Protocol):
    customer_id: str


def _assert_no_unsupported_constraints(schema: dict[str, Any]) -> None:
    unsupported = {
        "minimum",
        "maximum",
        "multipleOf",
        "pattern",
        "format",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
    }

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            assert unsupported.isdisjoint(value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)

    walk(schema)


def test_model_tool_schemas_are_flat_strict_and_customer_free() -> None:
    assert "create_payment_agreement" not in MODEL_TOOL_SCHEMAS
    for schema_model in MODEL_TOOL_SCHEMAS.values():
        schema = schema_model.model_json_schema()
        serialized = str(schema)
        assert "customer_id" not in serialized
        assert schema.get("additionalProperties") is False
        assert set(schema.get("required", [])) == set(schema.get("properties", {}))
        _assert_no_unsupported_constraints(schema)


def test_registry_never_exposes_the_write_capability() -> None:
    for phase in ToolPhase:
        tools = model_tools_for_phase(phase)
        assert "create_payment_agreement" not in tools
    assert model_tools_for_phase(ToolPhase.DRAFT_PENDING) == {}


def test_domain_constraints_are_separate_from_model_schemas() -> None:
    with pytest.raises(ValueError):
        Debt.model_validate(
            {
                "customer_id": "attacker-controlled",
                "moneda": "ARS",
                "saldo_total": 1,
                "capital": 1,
                "intereses": 0,
                "dias_mora": 1,
                "estado": "mora_temprana",
                "vencimientos": [],
                "as_of": datetime.now(UTC),
            }
        )


def test_customer_scope_cannot_be_created_from_an_arbitrary_id() -> None:
    with pytest.raises(TypeError, match="authenticated session"):
        CustomerScope(
            "CUST-99999",
            datetime.now(UTC),
            "attacker-token",
            _issuer=object(),
        )


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get_customer", "/customer/CUST-00125"),
        ("get_debt", "/debt/CUST-00125"),
        ("get_payment_options", "/payment-options/CUST-00125"),
    ],
)
async def test_read_contracts_return_typed_data(
    gateway: CollectionsGateway,
    settings: Settings,
    method: str,
    path: str,
) -> None:
    del path
    scope = scope_for("CUST-00125", settings)
    result = await getattr(gateway, method)(scope)
    assert result.status == "ok"
    assert isinstance(result.data, BaseModel)
    assert cast(_HasCustomerId, result.data).customer_id == "CUST-00125"


async def test_zero_debt_is_a_valid_domain_result(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    result = await gateway.get_debt(scope_for("CUST-00450", settings))
    assert result.status == "ok"
    assert isinstance(result.data, Debt)
    assert result.data.estado == "paid"
    assert result.data.saldo_total == 0


async def test_not_found_is_not_reported_as_no_debt(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    result = await gateway.get_debt(scope_for("CUST-99999", settings))
    assert result.status == "not_found"
    assert result.data is None
    assert "No se encontró" in result.message_for_model


async def test_backend_rejects_foreign_sub(settings: Settings) -> None:
    token = issue_token("CUST-00125", settings).access_token
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/debt/CUST-00212", headers={"Authorization": f"Bearer {token}"}
        )
    assert response.status_code == status.HTTP_403_FORBIDDEN
    assert response.json()["detail"]["code"] == "SUBJECT_MISMATCH"


@pytest.mark.parametrize(
    ("failure_mode", "expected_status", "retriable"),
    [
        ("timeout", "timeout", True),
        ("500", "upstream_error", True),
        ("partial", "partial", False),
        ("malformed", "upstream_error", False),
    ],
)
async def test_all_failure_modes_are_compacted_into_tool_results(
    gateway: CollectionsGateway,
    settings: Settings,
    failure_mode: str,
    expected_status: str,
    retriable: bool,
) -> None:
    result = await gateway.get_debt(scope_for("CUST-00125", settings), failure_mode=failure_mode)
    assert result.status == expected_status
    assert result.retriable is retriable
    assert result.correlation_id
    if failure_mode == "partial":
        # `data` can't hold a partially-populated Debt (the domain model requires every
        # field), but the fields that DID come back must still reach the caller so a
        # future node can say "tengo el saldo pero no el detalle de intereses".
        assert result.data is None
        assert result.partial_data is not None
        assert "intereses" not in result.partial_data
        assert result.partial_data["saldo_total"] == 184500.0


async def test_idempotency_store_covers_in_progress_replay_reuse_and_ttl() -> None:
    store = IdempotencyStore()
    fingerprint = store.request_fingerprint({"draft_id": "one"})
    assert (await store.reserve("key", "CUST-00125", fingerprint)).kind == "new"
    assert (await store.reserve("key", "CUST-00125", fingerprint)).kind == "in_progress"
    await store.complete("key", {"agreement_id": "AGR-1"})
    assert (await store.reserve("key", "CUST-00125", fingerprint)).kind == "replay"
    assert (await store.reserve("key", "CUST-00125", "other-fingerprint")).kind == "reuse"

    expiring = IdempotencyStore(ttl=timedelta(microseconds=-1))
    assert (await expiring.reserve("expired", "CUST-00125", fingerprint)).kind == "new"
    assert (await expiring.reserve("expired", "CUST-00125", fingerprint)).kind == "new"


async def test_repeated_failures_open_the_circuit(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    scope = scope_for("CUST-00125", settings)
    first = await gateway.get_customer(scope, failure_mode="500")
    second = await gateway.get_customer(scope)
    assert first.status == "upstream_error"
    assert second.status == "upstream_error"
    assert "Circuito" in second.message_for_model


async def test_agreement_replay_returns_the_same_response(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    scope = scope_for("CUST-00125", settings)
    kwargs = {
        "draft_id": "draft-001",
        "opcion_id": "OPT-3C",
        "debt_fingerprint": "a" * 64,
        "medio_pago": "debito_automatico",
        "idempotency_key": agreement_idempotency_key("CUST-00125", "draft-001"),
    }
    first = await gateway.create_payment_agreement(scope, **kwargs)
    replay = await gateway.create_payment_agreement(scope, **kwargs)
    assert first.status == replay.status == "ok"
    assert isinstance(first.data, AgreementResponse)
    assert isinstance(replay.data, AgreementResponse)
    assert first.data.agreement_id == replay.data.agreement_id
    assert first.data.replayed is False
    assert replay.data.replayed is True


async def test_two_concurrent_requests_create_one_agreement(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    scope = scope_for("CUST-00125", settings)
    kwargs = {
        "draft_id": "draft-concurrent",
        "opcion_id": "OPT-3C",
        "debt_fingerprint": "e" * 64,
        "medio_pago": "debito_automatico",
        "idempotency_key": agreement_idempotency_key("CUST-00125", "draft-concurrent"),
    }
    first, second = await asyncio.gather(
        gateway.create_payment_agreement(scope, **kwargs),
        gateway.create_payment_agreement(scope, **kwargs),
    )
    assert first.status == second.status == "ok"
    assert isinstance(first.data, AgreementResponse)
    assert isinstance(second.data, AgreementResponse)
    assert first.data.agreement_id == second.data.agreement_id


async def test_idempotency_key_reuse_with_other_payload_is_rejected(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    scope = scope_for("CUST-00125", settings)
    key = agreement_idempotency_key("CUST-00125", "draft-reuse")
    first = await gateway.create_payment_agreement(
        scope,
        draft_id="draft-reuse",
        opcion_id="OPT-3C",
        debt_fingerprint="b" * 64,
        medio_pago="debito_automatico",
        idempotency_key=key,
    )
    reused = await gateway.create_payment_agreement(
        scope,
        draft_id="draft-reuse-modified",
        opcion_id="OPT-6C",
        debt_fingerprint="c" * 64,
        medio_pago="transferencia",
        idempotency_key=key,
    )
    assert first.status == "ok"
    assert reused.status == "invalid_input"
    assert "clave ya fue usada" in reused.message_for_model


async def test_active_agreement_conflict_returns_existing_business_outcome(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    scope = scope_for("CUST-00125", settings)
    common = {
        "opcion_id": "OPT-3C",
        "debt_fingerprint": "d" * 64,
        "medio_pago": "debito_automatico",
    }
    first = await gateway.create_payment_agreement(
        scope,
        draft_id="draft-a",
        idempotency_key=agreement_idempotency_key("CUST-00125", "draft-a"),
        **common,
    )
    conflict = await gateway.create_payment_agreement(
        scope,
        draft_id="draft-b",
        idempotency_key=agreement_idempotency_key("CUST-00125", "draft-b"),
        **common,
    )
    assert first.status == "ok"
    assert conflict.status == "rejected_by_policy"
    assert "acuerdo activo" in conflict.message_for_model


async def test_transfer_contract_returns_a_typed_ticket(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    result = await gateway.transfer_to_human(
        scope_for("CUST-00212", settings),
        conversation_id="conversation-001",
        motivo="amenaza_legal",
        resumen="El cliente informó que ya interviene su abogado.",
    )
    assert result.status == "ok"
    assert result.data is not None
    assert not isinstance(result.data, dict)
    assert result.data.ticket_id.startswith("TKT-")


async def test_fixture_debt_payload_is_strictly_validated(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    result = await gateway.get_debt(scope_for("CUST-00125", settings))
    assert isinstance(result.data, Debt)
    assert result.data.capital + result.data.intereses == result.data.saldo_total


async def test_customer_contract_includes_previous_agreements(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    result = await gateway.get_customer(scope_for("CUST-00212", settings))
    assert isinstance(result.data, Customer)
    assert result.data.acuerdos_previos.incumplidos == 2


async def test_get_customer_not_found(gateway: CollectionsGateway, settings: Settings) -> None:
    result = await gateway.get_customer(scope_for("CUST-99999", settings))
    assert result.status == "not_found"


async def test_get_payment_options_not_found(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    result = await gateway.get_payment_options(scope_for("CUST-99999", settings))
    assert result.status == "not_found"


@pytest.mark.parametrize("failure_mode", ["partial", "malformed"])
async def test_payment_options_injection_branches(
    gateway: CollectionsGateway, settings: Settings, failure_mode: str
) -> None:
    result = await gateway.get_payment_options(
        scope_for("CUST-00125", settings), failure_mode=failure_mode
    )
    assert result.status in {"partial", "upstream_error"}


async def test_gateway_context_manager_closes_the_client_it_owns(settings: Settings) -> None:
    async with CollectionsGateway(settings=settings) as owned_gateway:
        assert owned_gateway._owns_client is True
    assert owned_gateway._client.is_closed


async def test_read_transport_timeout_is_compacted_into_tool_result(
    gateway: CollectionsGateway, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.TimeoutException("boom")

    monkeypatch.setattr(gateway._client, "request", boom)
    result = await gateway.get_debt(scope_for("CUST-00125", settings))
    assert result.status == "timeout"
    assert result.retriable is True


async def test_read_transport_connect_error_is_compacted_into_tool_result(
    gateway: CollectionsGateway, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def boom(*args: Any, **kwargs: Any) -> httpx.Response:
        raise httpx.ConnectError("boom")

    monkeypatch.setattr(gateway._client, "request", boom)
    result = await gateway.get_debt(scope_for("CUST-00125", settings))
    assert result.status == "upstream_error"
    assert result.retriable is True


async def test_create_agreement_rejects_invalid_domain_parameters_before_any_request(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    result = await gateway.create_payment_agreement(
        scope_for("CUST-00125", settings),
        draft_id="draft-x",
        opcion_id="OPT-3C",
        debt_fingerprint="not-a-valid-fingerprint",
        medio_pago="debito_automatico",
        idempotency_key="irrelevant",
    )
    assert result.status == "invalid_input"
    assert "inválidos" in result.message_for_model


async def test_transfer_rejects_invalid_domain_parameters_before_any_request(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    result = await gateway.transfer_to_human(
        scope_for("CUST-00125", settings),
        conversation_id="conv-1",
        motivo="falla_tecnica",
        resumen=12345,  # type: ignore[arg-type]
    )
    assert result.status == "invalid_input"
    assert "inválidos" in result.message_for_model


async def test_transfer_failure_is_never_marked_retriable(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    # transfer_to_human has no idempotency key, so a blind retry could file the
    # same escalation twice. It must never come back flagged as retriable.
    result = await gateway.transfer_to_human(
        scope_for("CUST-00125", settings),
        conversation_id="conv-1",
        motivo="falla_tecnica",
        resumen="test",
        failure_mode="500",
    )
    assert result.status == "upstream_error"
    assert result.retriable is False


async def test_circuit_breaker_closes_again_after_the_reset_window(settings: Settings) -> None:
    reset_settings = Settings(
        mock_api_url=settings.mock_api_url,
        tool_retry_attempts=1,
        tool_timeout_seconds=settings.tool_timeout_seconds,
        circuit_breaker_threshold=1,
        circuit_breaker_reset_seconds=0.01,
    )
    transport = httpx.ASGITransport(app=app)
    client = httpx.AsyncClient(transport=transport, base_url="http://test")
    gateway = CollectionsGateway(client=client, settings=reset_settings)
    try:
        scope = scope_for("CUST-00125", reset_settings)
        opened = await gateway.get_customer(scope, failure_mode="500")
        assert opened.status == "upstream_error"
        await asyncio.sleep(0.05)
        recovered = await gateway.get_customer(scope)
        assert recovered.status == "ok"
    finally:
        await client.aclose()


async def test_partial_response_with_invalid_json_body_is_tolerated(
    gateway: CollectionsGateway, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def fake_partial(*args: Any, **kwargs: Any) -> httpx.Response:
        return httpx.Response(status_code=206, content=b"not-json")

    monkeypatch.setattr(gateway._client, "request", fake_partial)
    result = await gateway.get_debt(scope_for("CUST-00125", settings))
    assert result.status == "partial"
    assert result.partial_data is None


async def test_write_circuit_breaker_opens_after_repeated_failures(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    scope = scope_for("CUST-00125", settings)
    first = await gateway.create_payment_agreement(
        scope,
        draft_id="draft-circuit",
        opcion_id="OPT-3C",
        debt_fingerprint="a" * 64,
        medio_pago="debito_automatico",
        idempotency_key=agreement_idempotency_key("CUST-00125", "draft-circuit"),
        failure_mode="500",
    )
    second = await gateway.create_payment_agreement(
        scope,
        draft_id="draft-circuit-2",
        opcion_id="OPT-3C",
        debt_fingerprint="a" * 64,
        medio_pago="debito_automatico",
        idempotency_key=agreement_idempotency_key("CUST-00125", "draft-circuit-2"),
    )
    assert first.status == "upstream_error"
    assert second.status == "upstream_error"
    assert "Circuito" in second.message_for_model


async def test_write_gives_up_after_exhausting_retries_on_persistent_conflict(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    scope = scope_for("CUST-00125", settings)
    key = agreement_idempotency_key("CUST-00125", "draft-stuck")
    fingerprint = idempotency_store.request_fingerprint(
        {
            "customer_id": "CUST-00125",
            "draft_id": "draft-stuck",
            "opcion_id": "OPT-3C",
            "debt_fingerprint": "f" * 64,
            "medio_pago": "debito_automatico",
        }
    )
    # Reserve the key and never complete it, simulating a write that is stuck
    # "in progress" for the whole duration of the caller's retry budget.
    reservation = await idempotency_store.reserve(key, "CUST-00125", fingerprint)
    assert reservation.kind == "new"

    result = await gateway.create_payment_agreement(
        scope,
        draft_id="draft-stuck",
        opcion_id="OPT-3C",
        debt_fingerprint="f" * 64,
        medio_pago="debito_automatico",
        idempotency_key=key,
    )
    assert result.status == "upstream_error"
    assert result.message_for_model == "El request sigue en proceso"


async def test_create_agreement_rejects_unknown_option(
    gateway: CollectionsGateway, settings: Settings
) -> None:
    result = await gateway.create_payment_agreement(
        scope_for("CUST-00125", settings),
        draft_id="draft-unknown-option",
        opcion_id="OPT-ZZZZZZ",
        debt_fingerprint="a" * 64,
        medio_pago="debito_automatico",
        idempotency_key=agreement_idempotency_key("CUST-00125", "draft-unknown-option"),
    )
    assert result.status == "rejected_by_policy"
    assert "no pertenece" in result.message_for_model


async def test_create_agreement_abandons_reservation_on_unexpected_error(
    gateway: CollectionsGateway, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(_agreement: dict[str, Any]) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(idempotency_store, "save_agreement", boom)
    key = agreement_idempotency_key("CUST-00125", "draft-boom")
    with pytest.raises(RuntimeError, match="boom"):
        await gateway.create_payment_agreement(
            scope_for("CUST-00125", settings),
            draft_id="draft-boom",
            opcion_id="OPT-3C",
            debt_fingerprint="a" * 64,
            medio_pago="debito_automatico",
            idempotency_key=key,
        )
    # The reservation must have been abandoned, not left dangling as in_progress,
    # so a retry after a genuine server bug can still get a clean "new" attempt.
    fingerprint = idempotency_store.request_fingerprint(
        {
            "customer_id": "CUST-00125",
            "draft_id": "draft-boom",
            "opcion_id": "OPT-3C",
            "debt_fingerprint": "a" * 64,
            "medio_pago": "debito_automatico",
        }
    )
    reservation = await idempotency_store.reserve(key, "CUST-00125", fingerprint)
    assert reservation.kind == "new"


def test_safe_error_detail_returns_empty_dict_for_non_json_body() -> None:
    response = httpx.Response(status_code=500, content=b"not-json")
    assert _safe_error_detail(response) == {}


async def test_health_endpoint() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_token_endpoint_issues_a_bearer_token() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/auth/token", json={"customer_id": "CUST-00125"})
    assert response.status_code == 200
    assert response.json()["token_type"] == "bearer"


async def test_token_endpoint_rejects_unknown_customer() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post("/auth/token", json={"customer_id": "CUST-99999"})
    assert response.status_code == 404


async def test_require_claims_rejects_missing_token() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get("/debt/CUST-00125")
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"]["code"] == "MISSING_TOKEN"


async def test_require_claims_rejects_an_invalid_token() -> None:
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/debt/CUST-00125", headers={"Authorization": "Bearer not-a-real-token"}
        )
    assert response.status_code == status.HTTP_401_UNAUTHORIZED
    assert response.json()["detail"]["code"] == "INVALID_TOKEN"


async def test_create_agreement_requires_idempotency_key_header(settings: Settings) -> None:
    token = issue_token("CUST-00125", settings).access_token
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/payment-agreement",
            json={
                "customer_id": "CUST-00125",
                "draft_id": "d1",
                "opcion_id": "OPT-3C",
                "debt_fingerprint": "a" * 64,
                "medio_pago": "debito_automatico",
            },
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == status.HTTP_400_BAD_REQUEST
    assert response.json()["detail"]["code"] == "INVALID_IDEMPOTENCY_KEY"


async def test_latency_injection_delays_the_response(settings: Settings) -> None:
    token = issue_token("CUST-00125", settings).access_token
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/customer/CUST-00125",
            headers={"Authorization": f"Bearer {token}", "X-Mock-Latency-Ms": "1"},
        )
    assert response.status_code == 200


async def test_unknown_failure_mode_header_is_rejected(settings: Settings) -> None:
    token = issue_token("CUST-00125", settings).access_token
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.get(
            "/customer/CUST-00125",
            headers={"Authorization": f"Bearer {token}", "X-Mock-Fail": "bogus"},
        )
    assert response.status_code == status.HTTP_400_BAD_REQUEST


def test_customer_scope_exposes_issued_at_and_cache_namespace() -> None:
    session = AuthenticatedSession(
        customer_id="CUST-00125",
        subject="CUST-00125",
        downstream_token="token",
        authenticated_at=datetime.now(UTC),
    )
    scope = CustomerScope.from_session(session)
    assert scope.issued_at == session.authenticated_at
    assert scope.cache_namespace() == "customer:CUST-00125"


def test_session_from_token_claims_builds_an_authenticated_session() -> None:
    session = session_from_token_claims("CUST-00125", "raw-token", "CUST-00125")
    assert session.customer_id == "CUST-00125"
    assert session.downstream_token == "raw-token"
    assert session.subject == "CUST-00125"


def test_initialize_database_runs_migrations_and_sets_up_the_checkpointer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_checkpointer = MagicMock()
    fake_context = MagicMock()
    fake_context.__enter__ = MagicMock(return_value=fake_checkpointer)
    fake_context.__exit__ = MagicMock(return_value=False)

    with (
        patch("scripts.initialize_database.command.upgrade") as fake_upgrade,
        patch(
            "scripts.initialize_database.PostgresSaver.from_conn_string",
            return_value=fake_context,
        ) as fake_from_conn,
    ):
        initialize_database.main()

    fake_upgrade.assert_called_once()
    fake_from_conn.assert_called_once()
    fake_checkpointer.setup.assert_called_once()
