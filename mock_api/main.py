from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException, Response, status
from fastapi.responses import JSONResponse
from pydantic import TypeAdapter

from app.tools.schemas import (
    AgreementResponse,
    CreateAgreementRequest,
    Customer,
    Debt,
    PaymentOption,
    PaymentOptions,
    TransferRequest,
    TransferResponse,
)
from mock_api.auth import (
    TokenClaims,
    TokenRequest,
    TokenResponse,
    assert_subject,
    issue_token,
    parse_idempotency_key,
    require_claims,
)
from mock_api.failure_injection import FailureInjection, failure_injection
from mock_api.idempotency_store import idempotency_store

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE_TIME = datetime.fromisoformat("2026-09-11T14:03:00-03:00")


def _load_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _load_customers() -> dict[str, Customer]:
    raw = (FIXTURES / "customers.json").read_text(encoding="utf-8")
    customers = TypeAdapter(list[Customer]).validate_json(raw)
    return {customer.customer_id: customer for customer in customers}


CUSTOMERS = _load_customers()
DEBTS_RAW: dict[str, dict[str, Any]] = _load_json("debts.json")
OPTIONS_RAW: dict[str, Any] = _load_json("options.json")

app = FastAPI(
    title="Froneus Collections Mock API",
    version="0.1.0",
    description="Backend determinista con fallas e idempotencia para el agente de cobranzas.",
)


async def _prepare(injection: FailureInjection) -> None:
    await injection.apply_latency()
    injection.raise_transport_failure()


def _malformed_or_partial(injection: FailureInjection, payload: dict[str, Any]) -> Response | None:
    if injection.mode == "malformed":
        return Response(content=b'{"payload":', media_type="application/json", status_code=200)
    if injection.mode == "partial":
        partial = dict(payload)
        for field in ("intereses", "email_registrado", "opciones", "resumen"):
            if field in partial:
                partial.pop(field)
                break
        return JSONResponse(
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            content={"data": partial, "missing_fields": [field]},
        )
    return None


def _not_found(resource: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail={"code": "NOT_FOUND", "message": f"No se encontró {resource}"},
    )


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/auth/token", response_model=TokenResponse)
async def token(request: TokenRequest) -> TokenResponse:
    if request.customer_id not in CUSTOMERS:
        raise _not_found("el cliente")
    return issue_token(request.customer_id)


@app.get("/customer/{customer_id}", response_model=Customer)
async def get_customer(
    customer_id: str,
    claims: Annotated[TokenClaims, Depends(require_claims)],
    injection: Annotated[FailureInjection, Depends(failure_injection)],
) -> Customer | Response:
    assert_subject(claims, customer_id)
    await _prepare(injection)
    customer = CUSTOMERS.get(customer_id)
    if customer is None:
        raise _not_found("el cliente")
    injected = _malformed_or_partial(injection, customer.model_dump(mode="json"))
    return injected or customer


@app.get("/debt/{customer_id}", response_model=Debt)
async def get_debt(
    customer_id: str,
    claims: Annotated[TokenClaims, Depends(require_claims)],
    injection: Annotated[FailureInjection, Depends(failure_injection)],
) -> Debt | Response:
    assert_subject(claims, customer_id)
    await _prepare(injection)
    raw = DEBTS_RAW.get(customer_id)
    if raw is None:
        raise _not_found("la deuda")
    payload = {"customer_id": customer_id, **raw}
    previous = CUSTOMERS[customer_id].acuerdos_previos
    payload.setdefault("acuerdos_previos", previous.model_dump(mode="json"))
    injected = _malformed_or_partial(injection, payload)
    return injected or Debt.model_validate_json(json.dumps(payload))


@app.get("/payment-options/{customer_id}", response_model=PaymentOptions)
async def get_payment_options(
    customer_id: str,
    claims: Annotated[TokenClaims, Depends(require_claims)],
    injection: Annotated[FailureInjection, Depends(failure_injection)],
) -> PaymentOptions | Response:
    assert_subject(claims, customer_id)
    await _prepare(injection)
    raw = OPTIONS_RAW.get(customer_id)
    if raw is None:
        raise _not_found("las opciones de pago")
    normalized = raw if isinstance(raw, dict) else {"opciones": raw}
    payload = {
        "customer_id": customer_id,
        "opciones": normalized.get("opciones", []),
        "motivo": normalized.get("motivo"),
        "policy_refs": normalized.get("policy_refs", []),
        "as_of": REFERENCE_TIME.isoformat(),
    }
    injected = _malformed_or_partial(injection, payload)
    if injected is not None:
        return injected
    return PaymentOptions.model_validate_json(json.dumps(payload))


@app.post("/payment-agreement", response_model=AgreementResponse)
async def create_payment_agreement(
    request: CreateAgreementRequest,
    claims: Annotated[TokenClaims, Depends(require_claims)],
    idempotency_key: Annotated[str, Depends(parse_idempotency_key)],
    injection: Annotated[FailureInjection, Depends(failure_injection)],
) -> AgreementResponse | Response:
    assert_subject(claims, request.customer_id)
    await _prepare(injection)
    request_payload = request.model_dump(mode="json")
    fingerprint = idempotency_store.request_fingerprint(request_payload)
    reservation = await idempotency_store.reserve(idempotency_key, request.customer_id, fingerprint)
    if reservation.kind == "reuse":
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail={
                "code": "IDEMPOTENCY_KEY_REUSE",
                "message": "La clave ya fue usada para otro request",
            },
        )
    if reservation.kind == "in_progress":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"code": "REQUEST_IN_PROGRESS", "message": "El request sigue en proceso"},
        )
    if reservation.kind == "replay":
        assert reservation.record.response is not None
        replay = {**reservation.record.response, "replayed": True}
        return AgreementResponse.model_validate_json(json.dumps(replay))

    lock = await idempotency_store.customer_lock(request.customer_id)
    try:
        async with lock:
            current = idempotency_store.active_agreement(
                request.customer_id, request.debt_fingerprint
            )
            if current is not None:
                await idempotency_store.abandon(idempotency_key)
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail={
                        "code": "AGREEMENT_EXISTS",
                        "message": "Ya existe un acuerdo activo para esta deuda",
                        "agreement_id": current["agreement_id"],
                    },
                )
            option_payload = OPTIONS_RAW.get(request.customer_id, [])
            options = (
                option_payload.get("opciones", [])
                if isinstance(option_payload, dict)
                else option_payload
            )
            valid_ids = {
                option.opcion_id
                for option in TypeAdapter(list[PaymentOption]).validate_json(json.dumps(options))
            }
            if request.opcion_id not in valid_ids:
                await idempotency_store.abandon(idempotency_key)
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
                    detail={
                        "code": "OPTION_NOT_FOUND",
                        "message": "La opción no pertenece al cliente",
                    },
                )
            response = AgreementResponse(
                agreement_id=f"AGR-{uuid4().hex[:8].upper()}",
                customer_id=request.customer_id,
                draft_id=request.draft_id,
                opcion_id=request.opcion_id,
                debt_fingerprint=request.debt_fingerprint,
                estado="active",
                created_at=datetime.now(UTC),
            )
            serialized = response.model_dump(mode="json")
            idempotency_store.save_agreement(serialized)
            await idempotency_store.complete(idempotency_key, serialized)
    except HTTPException:
        raise
    except Exception:
        await idempotency_store.abandon(idempotency_key)
        raise

    injected = _malformed_or_partial(injection, response.model_dump(mode="json"))
    return injected or response


@app.post("/transfer", response_model=TransferResponse)
async def transfer_to_human(
    request: TransferRequest,
    claims: Annotated[TokenClaims, Depends(require_claims)],
    injection: Annotated[FailureInjection, Depends(failure_injection)],
) -> TransferResponse | Response:
    assert_subject(claims, request.customer_id)
    await _prepare(injection)
    response = TransferResponse(
        ticket_id=f"TKT-{uuid4().hex[:8].upper()}",
        customer_id=request.customer_id,
        estado="queued",
        created_at=datetime.now(UTC),
    )
    injected = _malformed_or_partial(injection, response.model_dump(mode="json"))
    return injected or response
