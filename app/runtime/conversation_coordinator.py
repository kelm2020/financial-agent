from __future__ import annotations

import asyncio
import hashlib
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from typing import Any, Protocol

from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool, PoolTimeout


class ConversationBusyError(TimeoutError):
    """A turn could not acquire its conversation slot within the configured timeout."""


class ConversationRunCoordinator(Protocol):
    def hold(self, conversation_id: str) -> AbstractAsyncContextManager[None]: ...


@dataclass(slots=True)
class _LockEntry:
    lock: asyncio.Lock
    users: int = 0


class InMemoryConversationRunCoordinator:
    """Serialize one conversation without reducing cross-conversation concurrency.

    Entries are reference-counted under a metadata lock so cancellation cannot remove a lock
    while another waiter still references it.
    """

    def __init__(self, *, timeout_seconds: float = 10.0) -> None:
        self._timeout_seconds = timeout_seconds
        self._entries: dict[str, _LockEntry] = {}
        self._meta_lock = asyncio.Lock()

    @property
    def lock_count(self) -> int:
        return len(self._entries)

    @asynccontextmanager
    async def hold(self, conversation_id: str) -> AsyncIterator[None]:
        if not conversation_id:
            raise ValueError("conversation_id is required")
        async with self._meta_lock:
            entry = self._entries.setdefault(conversation_id, _LockEntry(asyncio.Lock()))
            entry.users += 1
        acquired = False
        try:
            try:
                await asyncio.wait_for(entry.lock.acquire(), timeout=self._timeout_seconds)
            except TimeoutError as exc:
                raise ConversationBusyError(
                    f"Conversation {conversation_id!r} is already processing another turn"
                ) from exc
            acquired = True
            yield
        finally:
            if acquired:
                entry.lock.release()
            async with self._meta_lock:
                entry.users -= 1
                if entry.users == 0 and not entry.lock.locked():
                    self._entries.pop(conversation_id, None)


def advisory_lock_key(conversation_id: str) -> int:
    digest = hashlib.blake2b(conversation_id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=True)


async def release_session_advisory_locks(connection: AsyncConnection[Any]) -> None:
    """Pool ``reset`` hook: a connection never returns to the pool holding a session lock."""
    await connection.execute("SELECT pg_advisory_unlock_all()")


class PostgresConversationRunCoordinator:
    """Session advisory lock held on one dedicated pooled connection per graph run.

    Acquisition polls ``pg_try_advisory_lock`` instead of cancelling a blocking
    ``pg_advisory_lock``: a cancelled lock request can be granted server-side just before the
    cancel arrives, leaving a reentrant session lock on a pooled connection. Polling never has
    an in-flight lock request to cancel, and the unlock always runs on the same session.
    """

    def __init__(
        self,
        pool: AsyncConnectionPool,
        *,
        timeout_seconds: float = 10.0,
        poll_interval_seconds: float = 0.05,
    ) -> None:
        self._pool = pool
        self._timeout_seconds = timeout_seconds
        self._poll_interval_seconds = poll_interval_seconds

    @asynccontextmanager
    async def hold(self, conversation_id: str) -> AsyncIterator[None]:
        if not conversation_id:
            raise ValueError("conversation_id is required")
        key = advisory_lock_key(conversation_id)
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._timeout_seconds
        busy = ConversationBusyError(
            f"Conversation {conversation_id!r} is already processing another turn"
        )
        async with AsyncExitStack() as stack:
            try:
                # Waiting for a pool slot uses the same budget: it never waits silently.
                connection = await stack.enter_async_context(
                    self._pool.connection(timeout=self._timeout_seconds)
                )
            except PoolTimeout as exc:
                raise busy from exc
            if not connection.autocommit:
                raise RuntimeError(
                    "PostgresConversationRunCoordinator requires an autocommit pool; "
                    "otherwise the session lock would hold an idle transaction during LLM calls"
                )
            while True:
                cursor = await connection.execute("SELECT pg_try_advisory_lock(%s)", (key,))
                row = await cursor.fetchone()
                if row is not None and row[0]:
                    break
                if loop.time() >= deadline:
                    raise busy
                await asyncio.sleep(self._poll_interval_seconds)
            try:
                yield
            finally:
                await connection.execute("SELECT pg_advisory_unlock(%s)", (key,))
