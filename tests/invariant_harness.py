from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from app.llm.protocol import LLMClient


@dataclass(frozen=True, slots=True)
class InvariantScenario:
    """Black-box input used by the invariants before the graph exists."""

    name: str
    customer_id: str = "CUST-00125"
    turns: tuple[str, ...] = ()
    channel: str = "chat"
    auth_tier: str = "T2"
    dtmf_confirm: str | None = None
    initial_state: dict[str, Any] = field(default_factory=dict)
    injected_policy_chunks: tuple[str, ...] = ()
    concurrent_last_turns: int = 1


@dataclass(frozen=True, slots=True)
class ObservedToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class InvariantObservation:
    """Security-relevant facts emitted by a complete scenario run."""

    responses: tuple[str, ...] = ()
    tool_calls: tuple[ObservedToolCall, ...] = ()
    agreement_writes: tuple[dict[str, Any], ...] = ()
    final_state: dict[str, Any] = field(default_factory=dict)
    events: tuple[dict[str, Any], ...] = ()
    http_status: int = 200
    ownership_checks: tuple[str, ...] = ()
    checkpoint_reads: tuple[str, ...] = ()
    cache_keys: tuple[str, ...] = ()
    log_output: str = ""


class InvariantDriver(Protocol):
    async def run(self, scenario: InvariantScenario, *, llm: LLMClient) -> InvariantObservation: ...


class Phase1MissingDriver:
    """The deliberate red edge: F3 replaces this with the graph adapter."""

    async def run(self, scenario: InvariantScenario, *, llm: LLMClient) -> InvariantObservation:
        del llm
        raise NotImplementedError(
            f"Invariant scenario {scenario.name!r} needs the graph scheduled for F3"
        )
