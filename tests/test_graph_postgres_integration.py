"""F3 checkpoint/ownership/lock behaviour against real PostgreSQL (RUN_POSTGRES_TESTS=1)."""

from __future__ import annotations

import asyncio
import os
import sys
import textwrap
from collections.abc import AsyncIterator, Iterator
from uuid import uuid4

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo
from psycopg_pool import AsyncConnectionPool
from pydantic import SecretStr

from app.conversations.store import PostgresConversationStore
from app.graph.build import build_graph
from app.graph.service import ConversationAgentService, ConversationNotFoundError
from app.main import create_app
from app.runtime.conversation_coordinator import (
    ConversationBusyError,
    PostgresConversationRunCoordinator,
    advisory_lock_key,
    release_session_advisory_locks,
)
from app.tools.schemas import AgreementDraft
from mock_api.idempotency_store import idempotency_store
from mock_api.main import app as mock_app
from tests.agent_support import agent_runtime, offline_settings

pytestmark = pytest.mark.skipif(
    os.getenv("RUN_POSTGRES_TESTS") != "1",
    reason="Set RUN_POSTGRES_TESTS=1 to exercise AsyncPostgresSaver and advisory locks",
)


def _as_url(base: str, name: str) -> str:
    prefix, _, _ = base.rpartition("/")
    return f"{prefix}/{name}"


@pytest.fixture(scope="module")
def agent_database_url() -> Iterator[str]:
    base = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/collections")
    name = f"{conninfo_to_dict(base)['dbname']}_agent_test"
    admin = make_conninfo(base, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as connection:
        connection.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(name)))
        connection.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(name)))
    url = _as_url(base, name)
    previous = os.environ.get("DATABASE_URL")
    os.environ["DATABASE_URL"] = url
    try:
        # Migrations from an empty database, plus the explicit checkpoint bootstrap.
        command.upgrade(Config("alembic.ini"), "head")
        from langgraph.checkpoint.postgres import PostgresSaver

        with PostgresSaver.from_conn_string(url) as bootstrap:
            bootstrap.setup()
        yield url
    finally:
        if previous is None:
            os.environ.pop("DATABASE_URL", None)
        else:
            os.environ["DATABASE_URL"] = previous


async def _pool(url: str, **kwargs: object) -> AsyncConnectionPool:
    pool = AsyncConnectionPool(
        url,
        open=False,
        kwargs={"autocommit": True},
        reset=release_session_advisory_locks,
        **kwargs,  # type: ignore[arg-type]
    )
    await pool.open(wait=True)
    return pool


@pytest.fixture
async def pool(agent_database_url: str) -> AsyncIterator[AsyncConnectionPool]:
    opened = await _pool(agent_database_url)
    try:
        yield opened
    finally:
        await opened.close()


async def _advisory_locks_held(url: str, key: int) -> int:
    async with await psycopg.AsyncConnection.connect(url, autocommit=True) as connection:
        unsigned = key % (1 << 64)
        # A bigint advisory key is stored split: classid = high 32 bits, objid = low 32 bits.
        cursor = await connection.execute(
            "SELECT count(*) FROM pg_locks WHERE locktype = 'advisory' AND objsubid = 1 "
            "AND classid::text::bigint = %s AND objid::text::bigint = %s",
            (unsigned >> 32, unsigned & 0xFFFFFFFF),
        )
        row = await cursor.fetchone()
        assert row is not None
        return int(row[0])


async def test_async_checkpoint_round_trip_and_two_phase_write(
    agent_database_url: str, pool: AsyncConnectionPool
) -> None:
    from app.graph.persistence import postgres_checkpointer

    await idempotency_store.reset()
    async with (
        agent_runtime() as runtime,
        # setup() is idempotent: the explicit bootstrap path is exercised on an existing schema.
        postgres_checkpointer(agent_database_url, setup=True) as saver,
    ):
        graph = build_graph(saver)
        store = PostgresConversationStore(pool)
        service = ConversationAgentService(
            graph=graph,
            conversations=store,
            coordinator=PostgresConversationRunCoordinator(pool, timeout_seconds=2),
        )
        conversation = await service.create_conversation("CUST-00125")
        assert await store.get_owned(conversation.conversation_id, "CUST-00212") is None
        assert await store.get_owned("not-a-uuid", "CUST-00125") is None
        with pytest.raises(ConversationNotFoundError):
            await service.send_message(
                conversation.conversation_id, "CUST-00212", "hola", context=runtime.context
            )

        proposed = await service.send_message(
            conversation.conversation_id,
            "CUST-00125",
            "Quiero la opción de 3 cuotas",
            context=runtime.context,
        )
        snapshot = await graph.aget_state({"configurable": {"thread_id": conversation.thread_id}})
        draft = snapshot.values["pending_draft"]
        assert isinstance(draft, AgreementDraft)
        assert draft == proposed.state["pending_draft"]
        assert (
            "response_plan" in snapshot.values
            and "text" not in type(snapshot.values["response_plan"]).model_fields
        )

        confirmed = await service.send_message(
            conversation.conversation_id, "CUST-00125", "sí, confirmo", context=runtime.context
        )
        assert confirmed.state["agreement_status"] == "active"
        (write,) = runtime.recorder.agreement_writes
        assert write["draft_id"] == draft.draft_id


async def test_same_conversation_is_serialized_across_sessions_and_released(
    agent_database_url: str,
) -> None:
    conversation_id = str(uuid4())
    first_pool = await _pool(agent_database_url)
    second_pool = await _pool(agent_database_url)
    try:
        first = PostgresConversationRunCoordinator(first_pool, timeout_seconds=2)
        second = PostgresConversationRunCoordinator(second_pool, timeout_seconds=0.3)
        entered = asyncio.Event()
        release = asyncio.Event()

        async def holder() -> None:
            async with first.hold(conversation_id):
                entered.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await entered.wait()
        with pytest.raises(ConversationBusyError):
            async with second.hold(conversation_id):
                pass
        # A different conversation is not blocked by the held lock.
        async with second.hold(str(uuid4())):
            pass
        release.set()
        await task
        async with second.hold(conversation_id):
            assert (
                await _advisory_locks_held(agent_database_url, advisory_lock_key(conversation_id))
                == 1
            )
        assert (
            await _advisory_locks_held(agent_database_url, advisory_lock_key(conversation_id)) == 0
        )
    finally:
        await first_pool.close()
        await second_pool.close()


async def test_lock_is_released_when_the_turn_fails_or_is_cancelled(
    agent_database_url: str, pool: AsyncConnectionPool
) -> None:
    coordinator = PostgresConversationRunCoordinator(pool, timeout_seconds=1)
    conversation_id = str(uuid4())
    key = advisory_lock_key(conversation_id)
    with pytest.raises(RuntimeError, match="boom"):
        async with coordinator.hold(conversation_id):
            raise RuntimeError("boom")
    assert await _advisory_locks_held(agent_database_url, key) == 0

    started = asyncio.Event()

    async def long_turn() -> None:
        async with coordinator.hold(conversation_id):
            started.set()
            await asyncio.sleep(30)

    task = asyncio.create_task(long_turn())
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert await _advisory_locks_held(agent_database_url, key) == 0


async def test_explicit_unlock_releases_even_without_pool_reset(agent_database_url: str) -> None:
    # The pool reset hook is defence in depth; the coordinator must not depend on it.
    bare = AsyncConnectionPool(
        agent_database_url, open=False, min_size=1, max_size=1, kwargs={"autocommit": True}
    )
    await bare.open(wait=True)
    try:
        conversation_id = str(uuid4())
        async with PostgresConversationRunCoordinator(bare, timeout_seconds=1).hold(
            conversation_id
        ):
            pass
        async with bare.connection() as same_session:
            cursor = await same_session.execute(
                "SELECT count(*) FROM pg_locks "
                "WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
            )
            row = await cursor.fetchone()
        assert row is not None and row[0] == 0
    finally:
        await bare.close()


async def test_pool_reset_releases_session_locks_before_connection_reuse(
    agent_database_url: str,
) -> None:
    reset_pool = await _pool(agent_database_url, min_size=1, max_size=1)
    key = advisory_lock_key(str(uuid4()))
    try:
        # Simulate an abnormal caller returning a session with a leaked lock. The pool hook is
        # defense in depth and must sanitize it before another borrower sees the connection.
        async with reset_pool.connection() as connection:
            await connection.execute("SELECT pg_advisory_lock(%s)", (key,))
        async with reset_pool.connection() as reused:
            cursor = await reused.execute(
                "SELECT count(*) FROM pg_locks "
                "WHERE locktype = 'advisory' AND pid = pg_backend_pid()"
            )
            row = await cursor.fetchone()
        assert row is not None and row[0] == 0
    finally:
        await reset_pool.close()


async def test_pool_exhaustion_is_a_bounded_busy_error(agent_database_url: str) -> None:
    tiny = await _pool(agent_database_url, min_size=1, max_size=1)
    try:
        coordinator = PostgresConversationRunCoordinator(tiny, timeout_seconds=0.3)
        async with coordinator.hold(str(uuid4())):
            with pytest.raises(ConversationBusyError):
                async with coordinator.hold(str(uuid4())):
                    pass
    finally:
        await tiny.close()


async def test_same_conversation_is_serialized_across_processes(agent_database_url: str) -> None:
    conversation_id = str(uuid4())
    script = textwrap.dedent(
        f"""
        import asyncio
        from psycopg_pool import AsyncConnectionPool
        from app.runtime.conversation_coordinator import PostgresConversationRunCoordinator

        async def main():
            options = {{"autocommit": True}}
            async with AsyncConnectionPool({agent_database_url!r}, kwargs=options) as pool:
                async with PostgresConversationRunCoordinator(pool, timeout_seconds=5).hold(
                    {conversation_id!r}
                ):
                    print("locked", flush=True)
                    await asyncio.sleep(1.5)

        asyncio.run(main())
        """
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable, "-c", script, stdout=asyncio.subprocess.PIPE
    )
    assert process.stdout is not None
    assert (await asyncio.wait_for(process.stdout.readline(), timeout=20)).strip() == b"locked"
    pool = await _pool(agent_database_url)
    try:
        with pytest.raises(ConversationBusyError):
            async with PostgresConversationRunCoordinator(pool, timeout_seconds=0.3).hold(
                conversation_id
            ):
                pass
        async with PostgresConversationRunCoordinator(pool, timeout_seconds=10).hold(
            conversation_id
        ):
            assert process.returncode is not None or await process.wait() == 0
    finally:
        await pool.close()


async def test_production_app_lifespan_installs_postgres_runtime(
    agent_database_url: str,
) -> None:
    settings = offline_settings(
        app_env="production",
        database_url=agent_database_url,
        openai_api_key=SecretStr("sk-test-not-used"),
        cohere_api_key=SecretStr("cohere-test-not-used"),
    )
    api = create_app(settings=settings, backend_app=mock_app)
    async with api.router.lifespan_context(api):
        assert isinstance(api.state.agent_service, ConversationAgentService)
        assert "render_and_validate" in api.state.graph.get_graph().nodes
