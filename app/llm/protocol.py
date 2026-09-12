from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel


@dataclass(frozen=True, slots=True)
class LLMCall:
    """One structured call captured by a deterministic test double."""

    task: str
    messages: tuple[Mapping[str, str], ...]
    response_model: type[BaseModel]


class LLMClient(Protocol):
    """Small provider boundary used by graph nodes and offline invariant tests."""

    async def complete[T: BaseModel](
        self,
        *,
        task: str,
        messages: Sequence[Mapping[str, str]],
        response_model: type[T],
    ) -> T: ...


type ScriptedResponse = BaseModel | Mapping[str, Any] | Exception


class ScriptExhaustedError(RuntimeError):
    """Raised when a scenario makes more LLM calls than it declared."""


class ScriptedLLM:
    """Network-free LLM double backed by a predeclared response sequence."""

    def __init__(self, responses: Iterable[ScriptedResponse]) -> None:
        self._responses = deque(responses)
        self._calls: list[LLMCall] = []

    @property
    def calls(self) -> tuple[LLMCall, ...]:
        return tuple(self._calls)

    @property
    def remaining(self) -> int:
        return len(self._responses)

    async def complete[T: BaseModel](
        self,
        *,
        task: str,
        messages: Sequence[Mapping[str, str]],
        response_model: type[T],
    ) -> T:
        self._calls.append(
            LLMCall(
                task=task,
                messages=tuple(dict(message) for message in messages),
                response_model=response_model,
            )
        )
        if not self._responses:
            raise ScriptExhaustedError(f"No scripted response remains for task {task!r}")

        response = self._responses.popleft()
        if isinstance(response, Exception):
            raise response
        if isinstance(response, response_model):
            return response
        return response_model.model_validate(response)
