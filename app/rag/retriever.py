from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Literal

from app.rag.embeddings import EmbeddingClient
from app.rag.models import NoEvidenceAction, RetrievalResult, SearchHit, Topic
from app.rag.rerank import Reranker
from app.rag.store import HybridStore

# §7.4 3.b: an incomplete answer about payment conditions costs more than one about opening
# hours, so negotiation and escalation topics derive directly when there is no evidence.
_HIGH_RISK_TOPICS = frozenset({"negociacion", "escalamiento"})


def no_evidence_action(
    topic: Topic, hits: tuple[SearchHit, ...] | list[SearchHit]
) -> NoEvidenceAction:
    resolved = topic if topic != "any" or not hits else hits[0].chunk.topic
    if resolved == "any" or resolved in _HIGH_RISK_TOPICS:
        return "derivar"
    return "ofrecer_derivacion"


def dense_evidence(hits: tuple[SearchHit, ...] | list[SearchHit]) -> float:
    """Strongest semantic match among ``hits`` (-1 when there are none)."""
    return max((hit.dense_score for hit in hits), default=-1.0)


@dataclass(frozen=True, slots=True)
class Retrieval:
    hits: list[SearchHit]
    evidence: float  # nearest-neighbour dense similarity over all fused candidates
    best_rrf: float  # top fused score, before any reranking


class PolicyRetriever:
    """Hybrid retrieval with evidence gates.

    RRF is rank-based: the top fused hit scores high even for an unrelated query, so it is
    not evidence by itself. The absolute evidence gate is ``evidence_gate``: the dense
    similarity (``min_dense_score``) or the cross-encoder score (``min_rerank_score``). Which
    gate and which threshold are decided on the dev split only (``make calibrate-rag`` /
    ``make calibrate-rerank``); ``min_rrf_score`` is the §7.4 fusion gate.
    """

    def __init__(
        self,
        store: HybridStore,
        embeddings: EmbeddingClient,
        *,
        min_rrf_score: float,
        min_dense_score: float,
        reranker: Reranker | None = None,
        min_rerank_score: float = 0.50,
        evidence_gate: Literal["dense", "rerank"] = "dense",
        rrf_k: int = 60,
        candidate_limit: int = 20,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._reranker = reranker
        self._min_rrf_score = min_rrf_score
        self._min_dense_score = min_dense_score
        self._min_rerank_score = min_rerank_score
        if evidence_gate == "rerank" and reranker is None:
            raise ValueError("El gate 'rerank' requiere un reranker")
        self._evidence_gate = evidence_gate
        self._rrf_k = rrf_k
        self._candidate_limit = candidate_limit

    async def retrieve(
        self,
        query: str,
        *,
        topic: Topic = "any",
        effective_on: date,
        limit: int = 4,
    ) -> Retrieval:
        """Ranked hits plus dense evidence, before any gate (calibration uses this too).

        Evidence is measured over every fused candidate, not only the returned ``limit``:
        the dense leg's nearest neighbour is always a candidate, so the evidence equals the
        best similarity in the index regardless of how each store ranks the lexical leg.
        """
        vectors = await self._embeddings.embed([query])
        fused = await self._store.search(
            query,
            vectors[0],
            topic=topic,
            effective_on=effective_on,
            limit=2 * self._candidate_limit,
            candidate_limit=self._candidate_limit,
            rrf_k=self._rrf_k,
        )
        evidence = dense_evidence(fused)
        best_rrf = fused[0].rrf_score if fused else 0.0
        if self._reranker is None:
            return Retrieval(hits=fused[:limit], evidence=evidence, best_rrf=best_rrf)
        reranked = await self._reranker.rerank(query, fused[: self._candidate_limit])
        return Retrieval(hits=reranked[:limit], evidence=evidence, best_rrf=best_rrf)

    async def search(
        self,
        query: str,
        *,
        topic: Topic = "any",
        effective_on: date,
        limit: int = 4,
    ) -> RetrievalResult:
        if not query.strip():
            return RetrievalResult(
                status="no_evidence",
                reason="La consulta está vacía.",
                on_no_evidence=no_evidence_action(topic, []),
            )
        retrieval = await self.retrieve(query, topic=topic, effective_on=effective_on, limit=limit)
        hits = retrieval.hits
        if not hits:
            return RetrievalResult(
                status="no_evidence",
                reason="No hay documentos aplicables.",
                on_no_evidence=no_evidence_action(topic, hits),
            )
        evidence_score, passed = self._evidence(retrieval)
        if not passed:
            return RetrievalResult(
                status="no_evidence",
                reason="La evidencia recuperada no supera el umbral configurado.",
                on_no_evidence=no_evidence_action(topic, hits),
                evidence_score=evidence_score,
            )
        return RetrievalResult(
            status="ok",
            hits=tuple(hits),
            source_chunk_ids=tuple(dict.fromkeys(hit.chunk.section_id for hit in hits)),
            evidence_score=evidence_score,
        )

    async def search_for_generation(
        self,
        query: str,
        *,
        topic: Topic = "any",
        effective_on: date,
        limit: int = 4,
    ) -> RetrievalResult:
        """Max-recall retrieval for grounded generation.

        Unlike :meth:`search`, this path does not turn a calibrated relevance score into an
        abstention decision: a policy answer is accepted only when every emitted sentence has an
        extractive citation verified by the output boundary. ``evidence_gate_passed`` still says
        whether a verbatim extract of these hits would be relevant.
        """
        if not query.strip():
            return RetrievalResult(
                status="no_evidence",
                reason="La consulta está vacía.",
                on_no_evidence=no_evidence_action(topic, []),
            )
        retrieval = await self.retrieve(query, topic=topic, effective_on=effective_on, limit=limit)
        if not retrieval.hits:
            return RetrievalResult(
                status="no_evidence",
                reason="No hay documentos aplicables.",
                on_no_evidence=no_evidence_action(topic, []),
            )
        evidence_score, passed = self._evidence(retrieval)
        return RetrievalResult(
            status="ok",
            hits=tuple(retrieval.hits),
            source_chunk_ids=tuple(dict.fromkeys(hit.chunk.section_id for hit in retrieval.hits)),
            evidence_score=evidence_score,
            evidence_gate_passed=passed,
        )

    def _evidence(self, retrieval: Retrieval) -> tuple[float | None, bool]:
        """Evidence score of non-empty hits and whether it clears the gate chosen on the dev
        split: the reranker may order candidates while the dense similarity decides abstention.
        RRF is checked on the fused ranking."""
        if self._evidence_gate == "rerank":
            score, threshold = retrieval.hits[0].rerank_score, self._min_rerank_score
        else:
            score, threshold = retrieval.evidence, self._min_dense_score
        passed = (
            retrieval.best_rrf >= self._min_rrf_score and score is not None and score >= threshold
        )
        return score, passed
