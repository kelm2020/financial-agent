from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field


class AuthenticatedSession(BaseModel):
    """Identity established at the trusted HTTP/CLI boundary."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    customer_id: str = Field(pattern=r"^CUST-\d{5}$")
    subject: str
    downstream_token: str
    authenticated_at: datetime


_SCOPE_ISSUER = object()


@dataclass(frozen=True, slots=True, init=False)
class CustomerScope:
    """Capability granting access to exactly one customer.

    It cannot be constructed with an arbitrary string. The authenticated boundary
    creates it from an ``AuthenticatedSession`` and repositories accept this type.
    """

    _customer_id: str
    _issued_at: datetime
    _downstream_token: str

    def __init__(
        self,
        customer_id: str,
        issued_at: datetime,
        downstream_token: str,
        *,
        _issuer: object,
    ) -> None:
        if _issuer is not _SCOPE_ISSUER:
            raise TypeError("CustomerScope can only be created from an authenticated session")
        object.__setattr__(self, "_customer_id", customer_id)
        object.__setattr__(self, "_issued_at", issued_at)
        object.__setattr__(self, "_downstream_token", downstream_token)

    @classmethod
    def from_session(cls, session: AuthenticatedSession) -> CustomerScope:
        return cls(
            session.customer_id,
            session.authenticated_at,
            session.downstream_token,
            _issuer=_SCOPE_ISSUER,
        )

    @property
    def customer_id(self) -> str:
        return self._customer_id

    @property
    def issued_at(self) -> datetime:
        return self._issued_at

    def as_header(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self._downstream_token}"}

    def cache_namespace(self) -> str:
        return f"customer:{self._customer_id}"


def session_from_token_claims(customer_id: str, token: str, subject: str) -> AuthenticatedSession:
    return AuthenticatedSession(
        customer_id=customer_id,
        subject=subject,
        downstream_token=token,
        authenticated_at=datetime.now(UTC),
    )
