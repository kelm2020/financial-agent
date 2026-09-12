from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

type Topic = Literal["negociacion", "medios_pago", "escalamiento", "faq", "any"]
type SegmentName = Literal["mora_temprana", "mora_media", "mora_tardia", "prejudicial"]
# §7.4 3.b: without evidence, low-risk topics offer a human; negotiation derives directly.
type NoEvidenceAction = Literal["ofrecer_derivacion", "derivar"]


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class KnowledgeDocument(_StrictModel):
    doc_id: str
    titulo: str
    version: str
    status: Literal["approved", "draft", "retired"]
    effective_from: date
    effective_to: date | None
    audiencia: tuple[str, ...]


class KnowledgeChunk(_StrictModel):
    chunk_id: str
    document_id: str
    section_id: str
    topic: ExcludeAnyTopic
    heading: str
    content: str
    contextualized_content: str
    policy_version: str
    status: Literal["approved", "draft", "retired"]
    valid_from: date
    valid_until: date | None
    audience: tuple[str, ...]
    applicable_segments: tuple[SegmentName, ...] = ()

    @property
    def embedding_text(self) -> str:
        return f"{self.contextualized_content}\n{self.heading}\n{self.content}"


type ExcludeAnyTopic = Literal["negociacion", "medios_pago", "escalamiento", "faq"]


class IndexMetadata(_StrictModel):
    kb_version: str
    embedding_model: str
    embedding_dimensions: int = Field(gt=0)
    corpus_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    chunk_count: int = Field(ge=0)


class SearchHit(_StrictModel):
    chunk: KnowledgeChunk
    lexical_score: float = Field(ge=0)
    dense_score: float = Field(ge=-1, le=1)
    lexical_rank: int | None = Field(default=None, ge=1)
    dense_rank: int | None = Field(default=None, ge=1)
    rrf_score: float = Field(ge=0, le=1)
    rerank_score: float | None = Field(default=None, ge=0, le=1)


class RetrievalResult(_StrictModel):
    status: Literal["ok", "no_evidence"]
    hits: tuple[SearchHit, ...] = ()
    source_chunk_ids: tuple[str, ...] = ()
    reason: str | None = None
    on_no_evidence: NoEvidenceAction | None = None
    # Strongest dense similarity seen, kept for traces even when abstaining (no chunks leak).
    evidence_score: float | None = None


class StoredChunk(_StrictModel):
    chunk: KnowledgeChunk
    embedding: tuple[float, ...]


class IndexSnapshot(_StrictModel):
    metadata: IndexMetadata
    records: tuple[StoredChunk, ...]
    indexed_at: datetime | None = None
