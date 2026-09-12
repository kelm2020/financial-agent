from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

import jwt
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field

from config.settings import Settings, get_settings


class TokenRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    customer_id: str = Field(pattern=r"^CUST-\d{5}$")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


class TokenClaims(BaseModel):
    model_config = ConfigDict(extra="ignore")

    sub: str = Field(pattern=r"^CUST-\d{5}$")
    aud: str
    iss: str
    iat: int
    exp: int
    scope: str


def issue_token(customer_id: str, settings: Settings | None = None) -> TokenResponse:
    resolved = settings or get_settings()
    now = datetime.now(UTC)
    expires_at = now + timedelta(seconds=resolved.mock_token_ttl_seconds)
    payload = {
        "sub": customer_id,
        "aud": resolved.mock_token_audience,
        "iss": resolved.mock_token_issuer,
        "iat": int(now.timestamp()),
        "exp": int(expires_at.timestamp()),
        "scope": "collections:customer",
    }
    encoded = jwt.encode(payload, resolved.mock_token_secret.get_secret_value(), algorithm="HS256")
    return TokenResponse(access_token=encoded, expires_in=resolved.mock_token_ttl_seconds)


def decode_token(token: str, settings: Settings | None = None) -> TokenClaims:
    resolved = settings or get_settings()
    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            resolved.mock_token_secret.get_secret_value(),
            algorithms=["HS256"],
            audience=resolved.mock_token_audience,
            issuer=resolved.mock_token_issuer,
        )
        return TokenClaims.model_validate(payload)
    except (jwt.PyJWTError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_TOKEN", "message": "Token inválido o vencido"},
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


_bearer = HTTPBearer(auto_error=False)


def require_claims(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> TokenClaims:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "MISSING_TOKEN", "message": "Falta el token acotado al cliente"},
            headers={"WWW-Authenticate": "Bearer"},
        )
    return decode_token(credentials.credentials)


def assert_subject(claims: TokenClaims, customer_id: str) -> None:
    if claims.sub != customer_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={"code": "SUBJECT_MISMATCH", "message": "El token no autoriza este recurso"},
        )


def parse_idempotency_key(
    idempotency_key: Annotated[str | None, Header(alias="Idempotency-Key")] = None,
) -> str:
    if not idempotency_key or len(idempotency_key) > 128:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_IDEMPOTENCY_KEY", "message": "Idempotency-Key requerido"},
        )
    return idempotency_key
