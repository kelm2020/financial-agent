from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

_CHECKPOINT_TYPES = (
    ("app.guards.preflight", "PreflightResult"),
    ("app.guards.injection", "GuardRuleResult"),
    ("app.guards.injection", "GuardModelResult"),
    ("app.graph.state", "RouteResult"),
    ("app.graph.state", "ConfirmationVerdict"),
    ("app.graph.state", "ResponsePlan"),
    ("app.rag.models", "KnowledgeChunk"),
    ("app.rag.models", "SearchHit"),
    ("app.tools.schemas", "PreviousAgreements"),
    ("app.tools.schemas", "Customer"),
    ("app.tools.schemas", "DueItem"),
    ("app.tools.schemas", "LastPayment"),
    ("app.tools.schemas", "Debt"),
    ("app.tools.schemas", "PaymentOption"),
    ("app.tools.schemas", "OptionsSnapshot"),
    ("app.tools.schemas", "AgreementDraft"),
)


def checkpoint_serializer() -> JsonPlusSerializer:
    """Strict explicit allowlist for every application type persisted by F3."""
    return JsonPlusSerializer(allowed_msgpack_modules=_CHECKPOINT_TYPES)


@asynccontextmanager
async def postgres_checkpointer(
    database_url: str, *, setup: bool = False, max_size: int = 10
) -> AsyncIterator[AsyncPostgresSaver]:
    """Async saver over a pool (not one shared connection); ``setup`` is explicit bootstrap.

    Connection kwargs are the ones the package requires when it does not open the connection
    itself: autocommit, dict rows and no server-side prepared statements.
    """
    async with AsyncConnectionPool(
        database_url,
        open=False,
        max_size=max_size,
        kwargs={"autocommit": True, "row_factory": dict_row, "prepare_threshold": 0},
    ) as pool:
        await pool.wait()
        saver = AsyncPostgresSaver(conn=pool, serde=checkpoint_serializer())  # type: ignore[arg-type]
        if setup:
            await saver.setup()
        yield saver
