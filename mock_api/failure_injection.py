import asyncio
from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import Header, HTTPException, status

FailureMode = Literal["timeout", "500", "partial", "malformed"]


@dataclass(frozen=True, slots=True)
class FailureInjection:
    mode: FailureMode | None
    latency_ms: int

    async def apply_latency(self) -> None:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)

    def raise_transport_failure(self) -> None:
        if self.mode == "timeout":
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"code": "MOCK_TIMEOUT", "message": "Tiempo de espera agotado"},
            )
        if self.mode == "500":
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "MOCK_INTERNAL_ERROR", "message": "Falla simulada"},
            )


def failure_injection(
    fail: Annotated[str | None, Header(alias="X-Mock-Fail")] = None,
    latency_ms: Annotated[int, Header(alias="X-Mock-Latency-Ms", ge=0, le=30_000)] = 0,
) -> FailureInjection:
    allowed: set[str] = {"timeout", "500", "partial", "malformed"}
    if fail is not None and fail not in allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "UNKNOWN_FAILURE_MODE", "message": "Modo de falla desconocido"},
        )
    return FailureInjection(mode=fail, latency_ms=latency_ms)  # type: ignore[arg-type]
