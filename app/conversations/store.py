from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID, uuid4

from psycopg.rows import class_row
from psycopg_pool import AsyncConnectionPool


@dataclass(frozen=True, slots=True)
class ConversationRecord:
    conversation_id: str
    customer_id: str
    thread_id: str
    channel: str
    created_at: datetime


class InMemoryConversationStore:
    def __init__(self) -> None:
        self._records: dict[str, ConversationRecord] = {}

    async def create(
        self,
        customer_id: str,
        *,
        channel: str = "chat",
        conversation_id: str | None = None,
    ) -> ConversationRecord:
        resolved_id = conversation_id or str(uuid4())
        record = ConversationRecord(
            conversation_id=resolved_id,
            customer_id=customer_id,
            thread_id=f"conversation:{resolved_id}",
            channel=channel,
            created_at=datetime.now(UTC),
        )
        self._records[resolved_id] = record
        return record

    async def get_owned(self, conversation_id: str, customer_id: str) -> ConversationRecord | None:
        record = self._records.get(conversation_id)
        return record if record is not None and record.customer_id == customer_id else None


class PostgresConversationStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self._pool = pool

    async def create(
        self,
        customer_id: str,
        *,
        channel: str = "chat",
        conversation_id: str | None = None,
    ) -> ConversationRecord:
        resolved = UUID(conversation_id) if conversation_id else uuid4()
        thread_id = f"conversation:{resolved}"
        async with self._pool.connection() as connection:
            cursor = await connection.execute(
                """
                INSERT INTO conversations (conversation_id, customer_id, thread_id, channel)
                VALUES (%s, %s, %s, %s)
                RETURNING conversation_id::text, customer_id, thread_id, channel, created_at
                """,
                (resolved, customer_id, thread_id, channel),
            )
            row = await cursor.fetchone()
            await connection.commit()
        assert row is not None
        return ConversationRecord(*row)

    async def get_owned(self, conversation_id: str, customer_id: str) -> ConversationRecord | None:
        try:
            resolved = UUID(conversation_id)
        except ValueError:
            return None
        async with self._pool.connection() as connection:
            cursor = connection.cursor(row_factory=class_row(ConversationRecord))
            await cursor.execute(
                """
                SELECT conversation_id::text, customer_id, thread_id, channel, created_at
                FROM conversations WHERE conversation_id = %s AND customer_id = %s
                """,
                (resolved, customer_id),
            )
            return await cursor.fetchone()
