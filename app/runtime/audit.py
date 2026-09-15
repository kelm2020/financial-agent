"""Append-only audit trail of the agent's effects (challenge point 10, blueprint §2).

Recorder events and logs describe a turn for debugging and evaluation; they are not a record of
effects. The audit trail is: every agreement write the customer confirmed, recorded before it is
attempted, its outcome, and every transfer to a person, written where the effect happens.

Rows carry codes and identifiers only, never customer text. The customer is pseudonymized with a
keyed hash (a plain hash of "CUST-00125" is reversed by enumerating ids), and each row is sealed
with an HMAC over its canonical content, so a row edited afterwards no longer verifies. Encryption
at rest and a WORM sink are F5.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import uuid
from dataclasses import dataclass
from typing import Literal, Protocol

from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

type AuditEventType = Literal[
    "agreement_write_requested",
    "agreement_created",
    "agreement_already_exists",
    "agreement_outcome_unknown",
    "agreement_write_rejected",
    "transfer_requested",
]


@dataclass(frozen=True, slots=True)
class AuditRecord:
    conversation_id: str
    customer_ref_hash: str
    event_type: AuditEventType
    payload: dict[str, str]
    integrity_hmac: str


class AuditLog(Protocol):
    async def append(self, record: AuditRecord) -> None: ...


class InMemoryAuditLog:
    """Local runs and tests: the same records, kept in the process."""

    def __init__(self) -> None:
        self.records: list[AuditRecord] = []

    async def append(self, record: AuditRecord) -> None:
        self.records.append(record)


class PostgresAuditLog:
    """Rows in ``audit_events`` (migration 0001). The pool runs in autocommit, so a row is durable
    when ``append`` returns."""

    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def append(self, record: AuditRecord) -> None:
        async with self._pool.connection() as connection:
            await connection.execute(
                "INSERT INTO audit_events "
                "(conversation_id, customer_ref_hash, event_type, payload, integrity_hmac) "
                "VALUES (%s, %s, %s, %s, %s)",
                (
                    _conversation_uuid(record.conversation_id),
                    record.customer_ref_hash,
                    record.event_type,
                    Jsonb(record.payload),
                    record.integrity_hmac,
                ),
            )


def _conversation_uuid(value: str) -> uuid.UUID | None:
    try:
        return uuid.UUID(value)
    except ValueError:
        return None


class AuditTrail:
    """Builds, seals and appends audit records to one sink."""

    def __init__(self, log: AuditLog, *, key: bytes) -> None:
        if not key:
            raise ValueError("An audit key is required")
        self._log = log
        self._key = key

    def customer_ref(self, customer_id: str) -> str:
        return hmac.new(self._key, f"customer|{customer_id}".encode(), hashlib.sha256).hexdigest()

    def seal(
        self,
        conversation_id: str,
        customer_ref_hash: str,
        event_type: str,
        payload: dict[str, str],
    ) -> str:
        canonical = json.dumps(
            [conversation_id, customer_ref_hash, event_type, payload],
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hmac.new(self._key, canonical.encode(), hashlib.sha256).hexdigest()

    def verifies(self, record: AuditRecord) -> bool:
        expected = self.seal(
            record.conversation_id, record.customer_ref_hash, record.event_type, record.payload
        )
        return hmac.compare_digest(expected, record.integrity_hmac)

    async def record(
        self,
        event_type: AuditEventType,
        *,
        conversation_id: str,
        customer_id: str,
        payload: dict[str, str],
    ) -> None:
        # Stored as a UUID or NULL: an id that is not a UUID is sealed as "", as it reads back.
        conversation_id = "" if _conversation_uuid(conversation_id) is None else conversation_id
        reference = self.customer_ref(customer_id)
        await self._log.append(
            AuditRecord(
                conversation_id=conversation_id,
                customer_ref_hash=reference,
                event_type=event_type,
                payload=dict(payload),
                integrity_hmac=self.seal(conversation_id, reference, event_type, payload),
            )
        )
