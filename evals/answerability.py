"""End-to-end answerability of policy questions through the real agent.

The retrieval split measures ranking and the evidence gate in isolation. This split measures what
the customer reads: whether an answerable question gets an answer citing a correct section and an
unanswerable one gets no cited answer, after routing, max-recall retrieval and verified quotes.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path
from typing import Literal, Self

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.graph.context import Retriever
from app.llm.protocol import LLMClient
from evals.environment import agent_session

DATASET_PATH = Path(__file__).parent / "answerability.yaml"
_CITATION = re.compile(r"\[((?:POL-NEG|PAY-MET|ESC|FAQ)-\d{3})\]", re.IGNORECASE)

type Outcome = Literal["correct", "wrong_section", "answered", "abstained"]


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AnswerableQuestion(_Frozen):
    id: str
    query: str = Field(min_length=1)
    expected_section_ids: tuple[str, ...] = Field(min_length=1)


class UnanswerableQuestion(_Frozen):
    id: str
    query: str = Field(min_length=1)


class AnswerabilityDataset(_Frozen):
    split: Literal["test"]
    effective_on: date
    customer_id: str
    positive: tuple[AnswerableQuestion, ...] = Field(min_length=1)
    negative: tuple[UnanswerableQuestion, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        ids = [case.id for case in self.positive] + [case.id for case in self.negative]
        if len(ids) != len(set(ids)):
            raise ValueError("Los ids de los casos deben ser únicos")
        return self


class QuestionResult(_Frozen):
    id: str
    query: str
    kind: Literal["positive", "negative"]
    intent: str
    cited: tuple[str, ...]
    outcome: Outcome
    text: str
    # Diagnostics: what the agent retrieved and which output checks fired.
    retrieved: tuple[str, ...] = ()
    flags: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return self.outcome == ("correct" if self.kind == "positive" else "abstained")


class AnswerabilityReport(_Frozen):
    results: tuple[QuestionResult, ...]

    def _count(self, kind: str, *, passed: bool | None = None) -> int:
        return sum(
            result.kind == kind and (passed is None or result.passed == passed)
            for result in self.results
        )

    @property
    def answered_correctly(self) -> int:
        return self._count("positive", passed=True)

    @property
    def positive_cases(self) -> int:
        return self._count("positive")

    @property
    def abstained_negatives(self) -> int:
        return self._count("negative", passed=True)

    @property
    def negative_cases(self) -> int:
        return self._count("negative")

    @property
    def wrong_citations(self) -> int:
        return sum(result.outcome in {"wrong_section", "answered"} for result in self.results)

    @property
    def failures(self) -> tuple[QuestionResult, ...]:
        return tuple(result for result in self.results if not result.passed)


def load_answerability_dataset(path: Path = DATASET_PATH) -> AnswerabilityDataset:
    return AnswerabilityDataset.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def classify(text: str, expected: tuple[str, ...]) -> tuple[tuple[str, ...], Outcome]:
    cited = tuple(dict.fromkeys(match.upper() for match in _CITATION.findall(text)))
    if not cited:
        return cited, "abstained"
    if not expected:
        return cited, "answered"
    return cited, "correct" if set(cited) & set(expected) else "wrong_section"


async def evaluate_answerability(
    dataset: AnswerabilityDataset, *, llm: LLMClient | None, retriever: Retriever
) -> AnswerabilityReport:
    """One fresh conversation per question, so no answer depends on an earlier one."""
    questions: list[tuple[AnswerableQuestion | UnanswerableQuestion, tuple[str, ...]]] = [
        *((question, question.expected_section_ids) for question in dataset.positive),
        *((question, ()) for question in dataset.negative),
    ]
    results: list[QuestionResult] = []
    for question, expected in questions:
        async with agent_session(dataset.customer_id, llm=llm, retriever=retriever) as session:
            observation = await session.send(question.query)
        cited, outcome = classify(observation.text, expected)
        route = observation.state.get("route_result")
        results.append(
            QuestionResult(
                id=question.id,
                query=question.query,
                kind="positive" if expected else "negative",
                intent=str(getattr(route, "intent", "")),
                cited=cited,
                outcome=outcome,
                text=observation.text,
                retrieved=tuple(
                    dict.fromkeys(
                        hit.chunk.section_id for hit in observation.state.get("retrieved", [])
                    )
                ),
                flags=tuple(observation.state.get("guard_flags", [])),
            )
        )
    return AnswerabilityReport(results=tuple(results))
