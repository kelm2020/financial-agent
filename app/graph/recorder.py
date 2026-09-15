from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

# Turn trace events that also go to the operational log: codes, section ids and scores only.
_TRACE_EVENTS = frozenset(
    {"source_selected", "policy_retrieval", "policy_answer", "response_outcome"}
)


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True, slots=True)
class LLMCallRecord:
    task: str
    latency_ms: float


class TurnBudgetExceeded(RuntimeError):
    """A turn attempted to exceed a hard tool/LLM-call budget."""

    def __init__(self, kind: str, limit: int) -> None:
        self.kind = kind
        self.limit = limit
        super().__init__(f"Turn {kind} budget exceeded (limit={limit})")


@dataclass(slots=True)
class TurnRecorder:
    events: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    agreement_writes: list[dict[str, Any]] = field(default_factory=list)
    llm_calls: list[LLMCallRecord] = field(default_factory=list)
    trajectory: list[str] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)
    max_tool_calls: int | None = None
    max_llm_calls: int | None = None
    _turn_tool_calls: int = 0
    _turn_llm_calls: int = 0

    def start_turn(self) -> None:
        self._turn_tool_calls = 0
        self._turn_llm_calls = 0

    def record_event(self, event_type: str, **payload: Any) -> None:
        self.events.append({"type": event_type, **payload})
        if event_type in _TRACE_EVENTS:
            # Codes, section ids and scores only. The same line goes to the recorder's log sink,
            # so the PII invariant (INV-14) inspects it like every other operational log.
            line = f"policy_trace {json.dumps({'type': event_type, **payload})}"
            logging.getLogger(__name__).info(line)
            self.record_log(line)

    def record_tool(self, name: str, **arguments: Any) -> None:
        # Safety transfers remain available after a dependency or budget failure. Counting the
        # recovery action itself would make the budget prevent its own mandated escalation.
        if name != "request_human":
            if self.max_tool_calls is not None and self._turn_tool_calls >= self.max_tool_calls:
                raise TurnBudgetExceeded("tool", self.max_tool_calls)
            self._turn_tool_calls += 1
        # Identity is intentionally absent: CustomerScope is runtime authority, not an arg.
        self.tool_calls.append(ToolCallRecord(name=name, arguments=dict(arguments)))
        self.trajectory.append(name)

    def reserve_llm(self, task: str) -> None:
        if self.max_llm_calls is not None and self._turn_llm_calls >= self.max_llm_calls:
            raise TurnBudgetExceeded("llm", self.max_llm_calls)
        self._turn_llm_calls += 1
        self.trajectory.append(f"llm:{task}")

    def complete_llm(self, task: str, latency_ms: float) -> None:
        self.llm_calls.append(LLMCallRecord(task=task, latency_ms=latency_ms))

    def record_step(self, name: str) -> None:
        self.trajectory.append(name)

    def record_log(self, message: str) -> None:
        self.log_lines.append(message)

    @property
    def log_output(self) -> str:
        return "\n".join(self.log_lines)
