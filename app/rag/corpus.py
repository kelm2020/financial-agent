from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from datetime import date
from pathlib import Path

import yaml  # type: ignore[import-untyped]

from app.rag.models import KnowledgeChunk, KnowledgeDocument, SegmentName
from app.rag.text import normalize_text

KB_PATH = Path(__file__).parents[2] / "kb"
_HEADING = re.compile(r"^##\s+(.*)$", re.MULTILINE)
_SECTION = re.compile(r"^([A-Z]+(?:-[A-Z]+)*-\d{3})\s+·\s+(.+)$")

_TOPICS = {
    "POL-NEG": "negociacion",
    "PAY-MET": "medios_pago",
    "ESC": "escalamiento",
    "FAQ": "faq",
}
_SEGMENT_PATTERNS: tuple[tuple[SegmentName, re.Pattern[str]], ...] = (
    ("mora_temprana", re.compile(r"\bmora temprana\b")),
    ("mora_media", re.compile(r"\bmora media\b")),
    ("mora_tardia", re.compile(r"\bmora tardia\b")),
    ("prejudicial", re.compile(r"\bprejudicial(es)?\b")),
)
_SEGMENT_LABELS: dict[SegmentName, str] = {
    "mora_temprana": "mora temprana",
    "mora_media": "mora media",
    "mora_tardia": "mora tardía",
    "prejudicial": "prejudicial",
}


def _slug(text: str) -> str:
    normalized = normalize_text(text)
    return re.sub(r"[^a-z0-9]+", "-", normalized).strip("-")


def parse_document(path: Path) -> tuple[KnowledgeDocument, str]:
    source = path.read_text(encoding="utf-8")
    if not source.startswith("---\n"):
        raise ValueError(f"{path} no tiene front matter")
    parts = source.split("---\n", maxsplit=2)
    if len(parts) != 3:
        raise ValueError(f"{path} tiene front matter inválido")
    raw_metadata = yaml.safe_load(parts[1])
    if not isinstance(raw_metadata, dict):
        raise ValueError(f"{path} tiene metadata inválida")
    document = KnowledgeDocument.model_validate_json(json.dumps(raw_metadata, default=str))
    return document, parts[2].strip()


def applicable_segments(text: str) -> tuple[SegmentName, ...]:
    normalized = normalize_text(text)
    return tuple(name for name, pattern in _SEGMENT_PATTERNS if pattern.search(normalized))


def _split_large_section(content: str, max_words: int, overlap: int) -> list[str]:
    """Split on whitespace words; size, overlap and threshold all use the same unit."""
    tokens = content.split()
    if len(tokens) <= max_words:
        return [content.strip()]
    pieces: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_words, len(tokens))
        pieces.append(" ".join(tokens[start:end]))
        if end == len(tokens):
            break
        start = max(end - overlap, start + 1)
    return pieces


def _context(
    document: KnowledgeDocument, section_id: str, title: str, segments: Iterable[SegmentName]
) -> str:
    labels = [_SEGMENT_LABELS[segment] for segment in segments]
    scope = f"aplica a clientes en {', '.join(labels)}" if labels else "aplica a todos los clientes"
    return f"Sección {section_id} «{title}» del documento {document.titulo}; {scope}."


def chunk_document(
    document: KnowledgeDocument,
    body: str,
    *,
    max_words: int = 520,  # ≈ 700 tokens in Spanish (§7.2)
    overlap: int = 60,  # ≈ 80 tokens
) -> list[KnowledgeChunk]:
    if document.doc_id not in _TOPICS:
        raise ValueError(f"No hay topic configurado para {document.doc_id}")
    headings = list(_HEADING.finditer(body))
    if not headings:
        raise ValueError(f"{document.doc_id} no contiene secciones ## válidas")
    chunks: list[KnowledgeChunk] = []
    for index, heading in enumerate(headings):
        section = _SECTION.match(heading.group(1).strip())
        if section is None:
            # A malformed heading would otherwise be silently merged into the previous chunk.
            raise ValueError(f"{document.doc_id}: encabezado ## inválido: {heading.group(1)!r}")
        section_id, title = section.groups()
        end = headings[index + 1].start() if index + 1 < len(headings) else len(body)
        section_content = body[heading.end() : end].strip()
        segments = applicable_segments(f"{title}\n{section_content}")
        parts = _split_large_section(section_content, max_words, overlap)
        for part_index, content in enumerate(parts, start=1):
            suffix = "" if len(parts) == 1 else f"-parte-{part_index}"
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document.doc_id}#{_slug(f'{section_id}-{title}')}{suffix}",
                    document_id=document.doc_id,
                    section_id=section_id,
                    topic=_TOPICS[document.doc_id],  # type: ignore[arg-type]
                    heading=title,
                    content=content,
                    contextualized_content=_context(document, section_id, title, segments),
                    policy_version=document.version,
                    status=document.status,
                    valid_from=document.effective_from,
                    valid_until=document.effective_to,
                    audience=document.audiencia,
                    applicable_segments=segments,
                )
            )
    return chunks


def load_corpus(
    root: Path = KB_PATH,
    *,
    effective_on: date,
) -> list[KnowledgeChunk]:
    chunks: list[KnowledgeChunk] = []
    for path in sorted(root.glob("*.md")):
        document, body = parse_document(path)
        if document.status != "approved":
            continue
        if document.effective_from > effective_on:
            continue
        if document.effective_to is not None and document.effective_to < effective_on:
            continue
        chunks.extend(chunk_document(document, body))
    if not chunks:
        raise ValueError("El corpus vigente está vacío")
    return chunks


def corpus_sha256(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
