from __future__ import annotations

import asyncio

from app.runtime.conversation_coordinator import InMemoryConversationRunCoordinator


async def test_same_conversation_runs_are_serialized() -> None:
    coordinator = InMemoryConversationRunCoordinator()
    active = 0
    peak = 0

    async def run() -> None:
        nonlocal active, peak
        async with coordinator.hold("same"):
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0)
            active -= 1

    await asyncio.gather(run(), run(), run())

    assert peak == 1
    assert coordinator.lock_count == 0


async def test_different_conversations_keep_parallelism() -> None:
    coordinator = InMemoryConversationRunCoordinator()
    both_entered = asyncio.Event()
    active = 0

    async def run(conversation_id: str) -> None:
        nonlocal active
        async with coordinator.hold(conversation_id):
            active += 1
            if active == 2:
                both_entered.set()
            await asyncio.wait_for(both_entered.wait(), timeout=1)
            active -= 1

    await asyncio.gather(run("one"), run("two"))

    assert coordinator.lock_count == 0
