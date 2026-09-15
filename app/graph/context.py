from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from pydantic import BaseModel

from app.graph.recorder import TurnRecorder
from app.guards.output import OutputValidator
from app.llm.protocol import LLMClient
from app.rag.models import RetrievalResult, Topic
from app.runtime.clock import Clock
from app.security.scope import CustomerScope
from app.tools.client import CollectionsGateway


class Retriever(Protocol):
    async def search(
        self,
        query: str,
        *,
        topic: Topic,
        effective_on: date,
        limit: int = 4,
    ) -> RetrievalResult: ...

    async def search_for_generation(
        self,
        query: str,
        *,
        topic: Topic,
        effective_on: date,
        limit: int = 4,
    ) -> RetrievalResult: ...


class RecordedLLM:
    """Per-turn metering and budget decorator for any ``LLMClient`` implementation."""

    def __init__(self, delegate: LLMClient, recorder: TurnRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder

    async def complete[T: BaseModel](
        self,
        *,
        task: str,
        messages: Sequence[Mapping[str, str]],
        response_model: type[T],
    ) -> T:
        self._recorder.reserve_llm(task)
        started = time.perf_counter()
        try:
            return await self._delegate.complete(
                task=task, messages=messages, response_model=response_model
            )
        finally:
            self._recorder.complete_llm(task, (time.perf_counter() - started) * 1000)


@dataclass(frozen=True, slots=True)
class GraphContext:
    scope: CustomerScope
    gateway: CollectionsGateway
    clock: Clock
    recorder: TurnRecorder
    output_validator: OutputValidator
    llm: LLMClient | None = None
    guard_classifier: LLMClient | None = None
    retriever: Retriever | None = None
    system_prompt: str = ""
    # Without a model, policy answers come from the calibrated retrieval gate. Production turns it
    # off: every policy answer there is judged answerable by the model.
    offline_policy_allowed: bool = True
