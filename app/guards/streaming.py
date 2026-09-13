from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from typing import Any

from app.guards.output import OutputValidator, ValidationContext

# es-AR exceptions: "$ 1.000" has no space after the dot; these abbreviations do.
_ABBREVIATIONS = ("sr.", "sra.", "dr.", "dra.", "n.º", "n°.", "nro.", "art.", "inc.")
_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[¿¡\"'(A-ZÁÉÍÓÚÑ0-9$])")


def split_clauses(text: str) -> list[str]:
    """Split a validated response into sentence clauses without breaking es-AR amounts."""
    clauses: list[str] = []
    start = 0
    for match in _BOUNDARY.finditer(text):
        piece = text[start : match.start()]
        if piece.casefold().endswith(_ABBREVIATIONS):
            continue
        if piece.strip():
            clauses.append(piece.strip())
        start = match.end()
    tail = text[start:].strip()
    if tail:
        clauses.append(tail)
    return clauses


type EventWriter = Callable[[Any], None]


class ValidatedEventStream:
    """The only channel that may reach SSE: each clause is validated before it is written."""

    def __init__(self, validator: OutputValidator, context: ValidationContext) -> None:
        self._validator = validator
        self._context = context
        self._events: list[dict[str, str]] = []

    @property
    def events(self) -> tuple[dict[str, str], ...]:
        return tuple(self._events)

    def with_context(self, context: ValidationContext) -> ValidatedEventStream:
        stream = ValidatedEventStream(self._validator, context)
        stream._events = self._events
        return stream

    def _emit(self, event: str, clause: str, writer: EventWriter | None) -> bool:
        if not self._validator.validate(clause, self._context).valid:
            return False
        payload = {"event": event, "data": clause}
        self._events.append(payload)
        if writer is not None:
            writer(payload)
        return True

    async def emit_filler(self, text: str, writer: EventWriter | None = None) -> bool:
        return self._emit("filler", text, writer)

    async def emit_clauses(
        self, clauses: Sequence[str], writer: EventWriter | None = None
    ) -> list[str]:
        """Emit in order and stop at the first clause that fails: nothing after it is sent."""
        emitted: list[str] = []
        for clause in clauses:
            if not self._emit("validated_clause", clause, writer):
                break
            emitted.append(clause)
        return emitted
