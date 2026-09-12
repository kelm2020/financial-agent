from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.rag.models import Topic
from app.rag.retriever import PolicyRetriever

EVALS_PATH = Path(__file__).parents[2] / "evals"
# Held-out: the nine labeled queries of Anexo E.1. Never read while tuning anything.
TEST_DATASET_PATH = EVALS_PATH / "retrieval.yaml"
# Tuning split: thresholds and any lexical decision are calibrated here only.
DEV_DATASET_PATH = EVALS_PATH / "retrieval_dev.yaml"

type Split = Literal["dev", "test"]
DATASET_PATHS: dict[Split, Path] = {"dev": DEV_DATASET_PATH, "test": TEST_DATASET_PATH}


class _Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PositiveCase(_Frozen):
    id: str
    query: str = Field(min_length=1)
    topic: Topic
    # Any of these sections is a correct answer (FAQ entries restate policy sections).
    expected_section_ids: tuple[str, ...] = Field(min_length=1)


class NegativeCase(_Frozen):
    id: str
    query: str = Field(min_length=1)
    topic: Topic


class RetrievalDataset(_Frozen):
    split: Split
    effective_on: date
    positive: tuple[PositiveCase, ...] = Field(min_length=1)
    negative: tuple[NegativeCase, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        ids = [case.id for case in self.positive] + [case.id for case in self.negative]
        if len(ids) != len(set(ids)):
            raise ValueError("Los ids de los casos deben ser únicos")
        return self


class CaseResult(_Frozen):
    id: str
    query: str
    kind: Literal["positive", "negative"]
    status: Literal["ok", "no_evidence"]
    ranked_sections: tuple[str, ...]
    dense_evidence: float | None
    passed: bool
    rank: int | None = None


class RetrievalMetrics(_Frozen):
    split: Split
    recall_at_3: float
    mrr: float
    abstentions: int
    negative_cases: int
    positive_cases: int
    cases: tuple[CaseResult, ...]

    @property
    def failures(self) -> tuple[CaseResult, ...]:
        return tuple(case for case in self.cases if not case.passed)


def load_retrieval_dataset(split: Split, path: Path | None = None) -> RetrievalDataset:
    source = path or DATASET_PATHS[split]
    dataset = RetrievalDataset.model_validate(yaml.safe_load(source.read_text(encoding="utf-8")))
    if dataset.split != split:
        raise ValueError(f"{source} declara split={dataset.split}, se pidió {split}")
    return dataset


async def evaluate_retriever(
    retriever: PolicyRetriever, dataset: RetrievalDataset
) -> RetrievalMetrics:
    results: list[CaseResult] = []
    reciprocal_rank = 0.0
    for positive in dataset.positive:
        result = await retriever.search(
            positive.query, topic=positive.topic, effective_on=dataset.effective_on
        )
        ranked = tuple(hit.chunk.section_id for hit in result.hits)
        rank = next(
            (
                index
                for index, section in enumerate(ranked, 1)
                if section in positive.expected_section_ids
            ),
            None,
        )
        reciprocal_rank += 1 / rank if rank is not None else 0.0
        results.append(
            CaseResult(
                id=positive.id,
                query=positive.query,
                kind="positive",
                status=result.status,
                ranked_sections=ranked,
                dense_evidence=result.evidence_score,
                passed=rank is not None and rank <= 3,
                rank=rank,
            )
        )
    for negative in dataset.negative:
        result = await retriever.search(
            negative.query, topic=negative.topic, effective_on=dataset.effective_on
        )
        results.append(
            CaseResult(
                id=negative.id,
                query=negative.query,
                kind="negative",
                status=result.status,
                ranked_sections=tuple(hit.chunk.section_id for hit in result.hits),
                dense_evidence=result.evidence_score,
                passed=result.status == "no_evidence",
            )
        )
    positives = [case for case in results if case.kind == "positive"]
    return RetrievalMetrics(
        split=dataset.split,
        recall_at_3=sum(case.passed for case in positives) / len(positives),
        mrr=reciprocal_rank / len(positives),
        abstentions=sum(case.passed for case in results if case.kind == "negative"),
        negative_cases=len(dataset.negative),
        positive_cases=len(positives),
        cases=tuple(results),
    )


class Calibration(_Frozen):
    gate: Literal["dense", "rerank"]
    threshold: float
    positives_answered: int
    positive_cases: int
    negatives_abstained: int
    negative_cases: int


def _separating_threshold(positive: Sequence[float], negative: Sequence[float]) -> float:
    """Balanced accuracy (answered positives + abstained negatives); ties go to the widest
    gap, so the threshold sits between data points instead of on one."""
    values = sorted(set(positive) | set(negative))
    if len(values) < 2:
        raise ValueError("Los scores no varían: no hay umbral que calibrar")
    edges = [values[0] - 0.01, *values, values[-1] + 0.01]
    best = (-1.0, -1.0, edges[0])  # (objective, margin, threshold)
    for lower, upper in pairwise(edges):
        threshold = (lower + upper) / 2
        answered = sum(score >= threshold for score in positive)
        abstained = sum(score < threshold for score in negative)
        candidate = (answered / len(positive) + abstained / len(negative), upper - lower, threshold)
        if candidate[:2] > best[:2]:
            best = candidate
    return round(best[2], 3)


async def _calibrate(
    retriever: PolicyRetriever,
    dataset: RetrievalDataset,
    gate: Literal["dense", "rerank"],
) -> Calibration:
    if dataset.split != "dev":
        raise ValueError("La calibración sólo puede usar el split dev")

    async def score(query: str, topic: Topic) -> float:
        retrieval = await retriever.retrieve(query, topic=topic, effective_on=dataset.effective_on)
        if gate == "dense":
            return retrieval.evidence
        if not retrieval.hits:
            return -1.0
        if retrieval.hits[0].rerank_score is None:
            raise ValueError("Calibrar el reranker requiere un retriever con reranker")
        return retrieval.hits[0].rerank_score

    positive = [await score(case.query, case.topic) for case in dataset.positive]
    negative = [await score(case.query, case.topic) for case in dataset.negative]
    threshold = _separating_threshold(positive, negative)
    return Calibration(
        gate=gate,
        threshold=threshold,
        positives_answered=sum(value >= threshold for value in positive),
        positive_cases=len(positive),
        negatives_abstained=sum(value < threshold for value in negative),
        negative_cases=len(negative),
    )


async def calibrate_min_dense_score(
    retriever: PolicyRetriever, dataset: RetrievalDataset
) -> Calibration:
    """Dense gate on the dev split; build the retriever without reranker and gates."""
    return await _calibrate(retriever, dataset, "dense")


async def calibrate_min_rerank_score(
    retriever: PolicyRetriever, dataset: RetrievalDataset
) -> Calibration:
    """Cross-encoder gate on the dev split; the retriever must carry the reranker."""
    return await _calibrate(retriever, dataset, "rerank")
