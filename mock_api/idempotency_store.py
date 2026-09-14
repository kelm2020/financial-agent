from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

ReservationKind = Literal["new", "replay", "reuse", "in_progress"]


@dataclass(frozen=True, slots=True)
class IdempotencyRecord:
    key: str
    customer_id: str
    fingerprint: str
    state: Literal["in_progress", "completed"]
    response: dict[str, Any] | None
    created_at: datetime


@dataclass(frozen=True, slots=True)
class Reservation:
    kind: ReservationKind
    record: IdempotencyRecord


class IdempotencyStore:
    def __init__(self, ttl: timedelta = timedelta(hours=24)) -> None:
        self._ttl = ttl
        self._records: dict[str, IdempotencyRecord] = {}
        self._agreements: dict[tuple[str, str], dict[str, Any]] = {}
        self._meta_lock = asyncio.Lock()
        self._customer_locks: dict[str, asyncio.Lock] = {}

    @staticmethod
    def request_fingerprint(payload: dict[str, Any]) -> str:
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    async def reserve(self, key: str, customer_id: str, fingerprint: str) -> Reservation:
        async with self._meta_lock:
            now = datetime.now(UTC)
            self._purge_expired(now)
            current = self._records.get(key)
            if current is None:
                current = IdempotencyRecord(
                    key=key,
                    customer_id=customer_id,
                    fingerprint=fingerprint,
                    state="in_progress",
                    response=None,
                    created_at=now,
                )
                self._records[key] = current
                return Reservation("new", current)
            if current.customer_id != customer_id or current.fingerprint != fingerprint:
                return Reservation("reuse", current)
            if current.state == "in_progress":
                return Reservation("in_progress", current)
            return Reservation("replay", current)

    async def complete(self, key: str, response: dict[str, Any]) -> None:
        async with self._meta_lock:
            current = self._records[key]
            self._records[key] = replace(current, state="completed", response=dict(response))

    async def abandon(self, key: str) -> None:
        async with self._meta_lock:
            current = self._records.get(key)
            if current is not None and current.state == "in_progress":
                del self._records[key]

    async def customer_lock(self, customer_id: str) -> asyncio.Lock:
        async with self._meta_lock:
            return self._customer_locks.setdefault(customer_id, asyncio.Lock())

    def active_agreement(self, customer_id: str, debt_fingerprint: str) -> dict[str, Any] | None:
        agreement = self._agreements.get((customer_id, debt_fingerprint))
        return dict(agreement) if agreement is not None else None

    def active_agreement_count(self, customer_id: str) -> int:
        return sum(1 for owner, _ in self._agreements if owner == customer_id)

    def save_agreement(self, agreement: dict[str, Any]) -> None:
        key = (str(agreement["customer_id"]), str(agreement["debt_fingerprint"]))
        self._agreements[key] = dict(agreement)

    async def reset(self) -> None:
        async with self._meta_lock:
            self._records.clear()
            self._agreements.clear()
            self._customer_locks.clear()

    def _purge_expired(self, now: datetime) -> None:
        expired = [
            key for key, record in self._records.items() if now - record.created_at > self._ttl
        ]
        for key in expired:
            del self._records[key]


idempotency_store = IdempotencyStore()
