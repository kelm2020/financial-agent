from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Protocol

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
