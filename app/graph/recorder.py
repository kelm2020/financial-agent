from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class TurnRecorder:
    events: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    agreement_writes: list[dict[str, Any]] = field(default_factory=list)
    log_lines: list[str] = field(default_factory=list)

    def record_event(self, event_type: str, **payload: Any) -> None:
        self.events.append({"type": event_type, **payload})

    def record_tool(self, name: str, **arguments: Any) -> None:
        # Identity is intentionally absent: CustomerScope is runtime authority, not an arg.
        self.tool_calls.append(ToolCallRecord(name=name, arguments=dict(arguments)))

    def record_log(self, message: str) -> None:
        self.log_lines.append(message)

    @property
    def log_output(self) -> str:
        return "\n".join(self.log_lines)
