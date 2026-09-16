from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

import httpx
from pydantic import BaseModel, ValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.obs.tracing import record_attribute, span
from app.security.scope import CustomerScope
from app.tools.schemas import (
    AgreementResponse,
    CreateAgreementRequest,
    Customer,
    Debt,
    PaymentOptions,
    ToolResult,
    ToolStatus,
    TransferRequest,
    TransferResponse,
)
from config.settings import Settings, get_settings

_AGREEMENT_ID = re.compile(r"^AGR-[A-Z0-9]{4,32}$", re.IGNORECASE)


@dataclass(slots=True)
class _CircuitState:
    failures: int = 0
    opened_at: float | None = None


class CircuitOpenError(RuntimeError):
    pass


class _RetriableReadError(RuntimeError):
    def __init__(self, result: ToolResult[Any]) -> None:
        self.result = result
        super().__init__(result.message_for_model)


class CircuitBreaker:
    def __init__(self, threshold: int, reset_seconds: float) -> None:
        self._threshold = threshold
        self._reset_seconds = reset_seconds
        self._states: dict[str, _CircuitState] = {}
        self._lock = asyncio.Lock()

    async def before_call(self, operation: str) -> None:
        async with self._lock:
            state = self._states.setdefault(operation, _CircuitState())
            if state.opened_at is None:
                return
            if time.monotonic() - state.opened_at >= self._reset_seconds:
                state.failures = 0
                state.opened_at = None
                return
            raise CircuitOpenError(f"Circuito abierto para {operation}")

    async def record_success(self, operation: str) -> None:
        async with self._lock:
            self._states[operation] = _CircuitState()

    async def record_failure(self, operation: str) -> None:
        async with self._lock:
            state = self._states.setdefault(operation, _CircuitState())
            state.failures += 1
            if state.failures >= self._threshold:
                state.opened_at = time.monotonic()


class CollectionsGateway:
    def __init__(
        self,
        client: httpx.AsyncClient | None = None,
        settings: Settings | None = None,
        breaker: CircuitBreaker | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            base_url=str(self.settings.mock_api_url).rstrip("/"),
            timeout=self.settings.tool_timeout_seconds,
        )
        # The breaker is process state, not request state: the API injects one shared per
        # dependency so consecutive turns see the same open circuit (the mock backend failing
        # twice opens it for every later turn, not per message). Tests keep the per-gateway
        # default; the API shares one through the lifespan.
        self._breaker = breaker or CircuitBreaker(
            threshold=self.settings.circuit_breaker_threshold,
            reset_seconds=self.settings.circuit_breaker_reset_seconds,
        )

    async def __aenter__(self) -> CollectionsGateway:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def get_customer(
        self, scope: CustomerScope, *, failure_mode: str | None = None
    ) -> ToolResult[Customer]:
        return await self._read(
            operation="get_customer",
            path=f"/customer/{scope.customer_id}",
            scope=scope,
            response_model=Customer,
            failure_mode=failure_mode,
        )

    async def get_debt(
        self, scope: CustomerScope, *, failure_mode: str | None = None
    ) -> ToolResult[Debt]:
        return await self._read(
            operation="get_debt",
            path=f"/debt/{scope.customer_id}",
            scope=scope,
            response_model=Debt,
            failure_mode=failure_mode,
        )

    async def get_payment_options(
        self, scope: CustomerScope, *, failure_mode: str | None = None
    ) -> ToolResult[PaymentOptions]:
        return await self._read(
            operation="get_payment_options",
            path=f"/payment-options/{scope.customer_id}",
            scope=scope,
            response_model=PaymentOptions,
            failure_mode=failure_mode,
        )

    async def create_payment_agreement(
        self,
        scope: CustomerScope,
        *,
        draft_id: str,
        opcion_id: str,
        debt_fingerprint: str,
        medio_pago: str,
        idempotency_key: str,
        failure_mode: str | None = None,
    ) -> ToolResult[AgreementResponse]:
        raw = {
            "customer_id": scope.customer_id,
            "draft_id": draft_id,
            "opcion_id": opcion_id,
            "debt_fingerprint": debt_fingerprint,
            "medio_pago": medio_pago,
        }
        try:
            request = CreateAgreementRequest.model_validate(raw)
        except ValidationError as exc:
            return ToolResult[AgreementResponse](
                status="invalid_input",
                message_for_model=f"Parámetros de acuerdo inválidos: {exc.errors()[0]['msg']}",
                correlation_id=uuid4().hex,
            )
        return await self._write(
            operation="create_payment_agreement",
            path="/payment-agreement",
            scope=scope,
            response_model=AgreementResponse,
            payload=request.model_dump(mode="json"),
            idempotency_key=idempotency_key,
            failure_mode=failure_mode,
        )

    async def transfer_to_human(
        self,
        scope: CustomerScope,
        *,
        conversation_id: str,
        motivo: str,
        resumen: str,
        failure_mode: str | None = None,
    ) -> ToolResult[TransferResponse]:
        raw = {
            "customer_id": scope.customer_id,
            "conversation_id": conversation_id,
            "motivo": motivo,
            "resumen": resumen,
        }
        try:
            request = TransferRequest.model_validate(raw)
        except ValidationError as exc:
            return ToolResult[TransferResponse](
                status="invalid_input",
                message_for_model=f"Parámetros de derivación inválidos: {exc.errors()[0]['msg']}",
                correlation_id=uuid4().hex,
            )
        return await self._write(
            operation="transfer_to_human",
            path="/transfer",
            scope=scope,
            response_model=TransferResponse,
            payload=request.model_dump(mode="json"),
            failure_mode=failure_mode,
        )

    async def _read[T: BaseModel](
        self,
        *,
        operation: str,
        path: str,
        scope: CustomerScope,
        response_model: type[T],
        failure_mode: str | None,
    ) -> ToolResult[T]:
        with span(
            "execute_tool.read",
            attributes={
                "tool.name": operation,
                "tool.method": "GET",
                "tool.path": path,
                "app.customer_id": scope.customer_id,
            },
        ) as tool_span:
            try:
                await self._breaker.before_call(operation)
                async for attempt in AsyncRetrying(
                    stop=stop_after_attempt(self.settings.tool_retry_attempts),
                    wait=wait_exponential_jitter(initial=0.01, max=0.05),
                    retry=retry_if_exception_type(_RetriableReadError),
                    reraise=True,
                ):
                    with attempt:
                        result = await self._request_once(
                            "GET",
                            path,
                            operation=operation,
                            scope=scope,
                            response_model=response_model,
                            failure_mode=failure_mode,
                        )
                        if _identity_mismatch(result.data, scope):
                            # A record of another customer inside a 200 is a data-integrity
                            # failure, not a transient one: retrying would fetch the same
                            # corruption. The record never reaches the state.
                            record_attribute(tool_span, "tool.status", "identity_mismatch")
                            return ToolResult[T](
                                status="upstream_error",
                                data=None,
                                message_for_model=(
                                    f"{operation} devolvió un registro que no corresponde al "
                                    "cliente de esta conversación."
                                ),
                                retriable=False,
                                correlation_id=result.correlation_id,
                            )
                        if result.retriable:
                            await self._breaker.record_failure(operation)
                            raise _RetriableReadError(result)
                        if result.status == "ok":
                            await self._breaker.record_success(operation)
                        record_attribute(tool_span, "tool.status", result.status)
                        return result
            except _RetriableReadError as exc:
                record_attribute(tool_span, "tool.status", exc.result.status)
                return exc.result
            except CircuitOpenError:
                record_attribute(tool_span, "tool.status", "circuit_open")
                return ToolResult[T](
                    status="upstream_error",
                    message_for_model="Circuito abierto: el servicio falló repetidamente.",
                    retriable=False,
                    correlation_id=uuid4().hex,
                )
            # AsyncRetrying with reraise=True always returns from inside the loop or
            # raises _RetriableReadError once attempts are exhausted; this is unreachable.
            raise AssertionError(  # pragma: no cover
                "unreachable: AsyncRetrying always returns or re-raises"
            )

    async def _write[T: BaseModel](
        self,
        *,
        operation: str,
        path: str,
        scope: CustomerScope,
        response_model: type[T],
        payload: dict[str, Any],
        failure_mode: str | None,
        idempotency_key: str | None = None,
    ) -> ToolResult[T]:
        try:
            await self._breaker.before_call(operation)
        except CircuitOpenError:
            return ToolResult[T](
                status="upstream_error",
                message_for_model="Circuito abierto: el servicio falló repetidamente.",
                correlation_id=uuid4().hex,
            )
        # A write without an idempotency key can never be safely retried, no matter
        # what the failure looks like: a blind retry could duplicate its side effect.
        max_attempts = 3 if idempotency_key else 1
        attempt = 0
        while True:
            result = await self._request_once(
                "POST",
                path,
                operation=operation,
                scope=scope,
                response_model=response_model,
                payload=payload,
                idempotency_key=idempotency_key,
                failure_mode=failure_mode,
            )
            if result.status == "ok":
                mismatched = _correlation_mismatch(result.data, payload)
                if mismatched:
                    # The backend confirmed a different object than the one this request
                    # created: the write's outcome is UNKNOWN, not successful (the confirmed
                    # terms are not the customer's) and not failed (something may exist).
                    # The idempotency key is never regenerated: a new one could duplicate
                    # the very side effect this uncertainty forbids us from probing.
                    await self._breaker.record_failure(operation)
                    return ToolResult[T](
                        status="upstream_error",
                        data=None,
                        message_for_model=(
                            f"{operation} confirmó un registro cuyos datos ({mismatched}) "
                            "no coinciden con los enviados."
                        ),
                        retriable=False,
                        correlation_id=result.correlation_id,
                    )
                await self._breaker.record_success(operation)
                return result
            if result.status in {"timeout", "upstream_error"}:
                await self._breaker.record_failure(operation)
            attempt += 1
            if not (result.retriable and attempt < max_attempts):
                return result
            await asyncio.sleep(0.01 * (2 ** (attempt - 1)))

    async def _request_once[T: BaseModel](
        self,
        method: str,
        path: str,
        *,
        operation: str,
        scope: CustomerScope,
        response_model: type[T],
        payload: dict[str, Any] | None = None,
        idempotency_key: str | None = None,
        failure_mode: str | None = None,
    ) -> ToolResult[T]:
        correlation_id = uuid4().hex
        headers = {**scope.as_header(), "X-Correlation-ID": correlation_id}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        if failure_mode:
            headers["X-Mock-Fail"] = failure_mode
        # A write is only safe to retry blindly when it carries an idempotency key;
        # otherwise a retry could duplicate the side effect it is meant to avoid.
        retriable_if_transient = method == "GET" or idempotency_key is not None
        try:
            response = await self._client.request(method, path, json=payload, headers=headers)
        except httpx.TimeoutException:
            return ToolResult[T](
                status="timeout",
                message_for_model=f"{operation} agotó el tiempo de espera.",
                retriable=retriable_if_transient,
                correlation_id=correlation_id,
            )
        except httpx.HTTPError:
            return ToolResult[T](
                status="upstream_error",
                message_for_model=f"{operation} no pudo conectarse al servicio.",
                retriable=retriable_if_transient,
                correlation_id=correlation_id,
            )

        if response.status_code == 206:
            try:
                body = response.json()
            except ValueError:
                body = {}
            missing_fields = body.get("missing_fields", []) if isinstance(body, dict) else []
            partial_payload = body.get("data") if isinstance(body, dict) else None
            return ToolResult[T](
                status="partial",
                data=None,
                partial_data=partial_payload if isinstance(partial_payload, dict) else None,
                message_for_model=(
                    f"{operation} devolvió datos parciales; faltan: "
                    f"{', '.join(map(str, missing_fields)) or 'campos requeridos'}."
                ),
                correlation_id=correlation_id,
            )
        if 200 <= response.status_code < 300:
            try:
                parsed = response_model.model_validate_json(response.content)
            except ValidationError:
                return ToolResult[T](
                    status="upstream_error",
                    message_for_model=f"{operation} devolvió un payload inválido.",
                    correlation_id=correlation_id,
                )
            return ToolResult[T](
                status="ok",
                data=parsed,
                message_for_model=f"{operation} completado.",
                correlation_id=correlation_id,
            )

        detail = _safe_error_detail(response)
        code = str(detail.get("code", "UPSTREAM_ERROR"))
        message = str(detail.get("message", f"{operation} falló"))
        raw_resource_id = detail.get("agreement_id")
        resource_id = (
            raw_resource_id
            if isinstance(raw_resource_id, str) and _AGREEMENT_ID.fullmatch(raw_resource_id)
            else None
        )
        if response.status_code == 404:
            status_value: ToolStatus = "not_found"
            retriable = False
        elif response.status_code == 504:
            status_value = "timeout"
            retriable = retriable_if_transient
        elif response.status_code == 409 and code == "REQUEST_IN_PROGRESS":
            # The mock is still finishing a prior write for the same idempotency key;
            # this is a transient race, not a client error, and is safe to retry.
            status_value = "upstream_error"
            retriable = retriable_if_transient
        elif response.status_code in {429, 500, 502, 503}:
            status_value = "upstream_error"
            retriable = retriable_if_transient
        elif code in {"AGREEMENT_EXISTS", "OPTION_NOT_FOUND"}:
            status_value = "rejected_by_policy"
            retriable = False
        else:
            status_value = "invalid_input"
            retriable = False
        return ToolResult[T](
            status=status_value,
            data=None,
            message_for_model=message,
            retriable=retriable,
            correlation_id=correlation_id,
            error_code=code,
            resource_id=resource_id,
        )


def _safe_error_detail(response: httpx.Response) -> dict[str, Any]:
    try:
        body = response.json()
    except ValueError:
        return {}
    detail = body.get("detail", body) if isinstance(body, dict) else {}
    return detail if isinstance(detail, dict) else {}


def agreement_idempotency_key(customer_id: str, draft_id: str) -> str:
    import hashlib

    return hashlib.sha256(f"{customer_id}|{draft_id}".encode()).hexdigest()


def _identity_mismatch(data: Any, scope: CustomerScope) -> bool:
    """Whether a successfully parsed payload belongs to another customer.

    Schema validation cannot catch this: a well-formed record of CUST-00212 answers
    the request of CUST-00125. The identity fields the domain models carry are the
    contract; a payload without any identity is checked by its schema alone.
    """
    customer_id = getattr(data, "customer_id", None)
    return customer_id is not None and customer_id != scope.customer_id


def _correlation_mismatch(data: Any, payload: dict[str, Any]) -> str | None:
    """The first request field the confirmed response contradicts, or None.

    A write is only successful when the backend confirms the object this request
    created: customer, draft, option and debt fingerprint must match exactly. Any
    difference means the outcome is unknown, whatever the HTTP status said.
    """
    for field in ("customer_id", "draft_id", "opcion_id", "debt_fingerprint"):
        expected = payload.get(field)
        confirmed = getattr(data, field, None)
        if confirmed is not None and expected is not None and confirmed != expected:
            return field
    return None
