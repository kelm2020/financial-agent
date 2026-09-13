from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.llm.protocol import LLMClient


@dataclass(frozen=True, slots=True)
class InvariantScenario:
    """Black-box input for invariants whose implementation belongs to a later phase."""

    name: str
    customer_id: str = "CUST-00125"
    turns: tuple[str, ...] = ()
    channel: str = "chat"
    auth_tier: str = "T2"
    dtmf_confirm: str | None = None
    initial_state: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class InvariantObservation:
    responses: tuple[str, ...] = ()
    tool_calls: tuple[str, ...] = ()
    agreement_writes: tuple[dict[str, Any], ...] = ()
    final_state: dict[str, Any] = field(default_factory=dict)
    cache_keys: tuple[str, ...] = ()


class InvariantDriver(Protocol):
    async def run(self, scenario: InvariantScenario, *, llm: LLMClient) -> InvariantObservation: ...


class DeferredPhaseDriver:
    """The deliberate red edge for F5 (RLS, scoped cache) and F7 (voice).

    Nothing in the runtime implements those controls yet, so every scenario raises instead of
    pretending: the tests stay ``xfail(strict=True, raises=NotImplementedError)`` until the
    phase that owns them replaces this driver.
    """

    def __init__(self, phase: str) -> None:
        self.phase = phase

    async def run(self, scenario: InvariantScenario, *, llm: LLMClient) -> InvariantObservation:
        del llm
        raise NotImplementedError(f"Scenario {scenario.name!r} belongs to {self.phase}")
