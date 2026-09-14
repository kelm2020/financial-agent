from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict

from app.guards.codes import GuardFlag
from app.guards.config import guardrail_config
from app.guards.normalize import detection_skeleton

CITATION_LABEL = re.compile(r"\[(?:POL-NEG|PAY-MET|ESC|FAQ)-\d{3}\]", re.IGNORECASE)
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+")


class GroundedClaim(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sentence: str
    section_id: str
    quote: str


class GroundedReply(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    claims: tuple[GroundedClaim, ...]


def plain_text(text: str) -> str:
    """Markdown-free view of a KB chunk, shared by extracts, quotes and their sources."""
    lines: list[str] = []
    previous_row = False
    for line in _merge_wrapped_items(text.splitlines()):
        stripped = line.strip()
        if not stripped:
            continue
        if re.fullmatch(r"\|?\s*:?-{3,}.*", stripped):
            # The row above a separator is the table header: column labels, not a statement.
            if previous_row:
                lines.pop()
            previous_row = False
            continue
        previous_row = stripped.startswith("|")
        # Bullets and table rows are standalone statements; wrapped paragraph lines are not.
        standalone = bool(re.match(r"^(?:[-*]\s+|\d+\.\s+|\|)", stripped))
        stripped = re.sub(r"^(?:[-*]\s+|\d+\.\s+)", "", stripped)
        stripped = stripped.replace("**", "").replace("__", "").strip("| ").replace(" | ", ": ")
        if standalone and stripped[-1] not in ".:!?":
            stripped = f"{stripped}."
        lines.append(stripped)
    return " ".join(lines)


_ITEM = re.compile(r"^\s*(?:[-*]\s+|\d+\.\s+)")


def _merge_wrapped_items(lines: list[str]) -> list[str]:
    """A list item wrapped onto an indented line is still one statement."""
    merged: list[str] = []
    for line in lines:
        wrapped = (
            line[:1].isspace() and line.strip() and not re.match(r"^\s*(?:[-*]|\d+\.|\|)", line)
        )
        if wrapped and merged and _ITEM.match(merged[-1]):
            merged[-1] = f"{merged[-1].rstrip()} {line.strip()}"
        else:
            merged.append(line)
    return merged


def _canonical_sentence(sentence: str) -> str:
    return " ".join(detection_skeleton(CITATION_LABEL.sub(" ", sentence)).split()).strip(" .")


def split_sentences(text: str) -> list[str]:
    """Sentences with trailing citation-only fragments merged into the previous sentence."""
    sentences: list[str] = []
    for fragment in _SENTENCE_BREAK.split(text):
        stripped = fragment.strip()
        if not stripped:
            continue
        if sentences and not CITATION_LABEL.sub("", stripped).strip(" ."):
            sentences[-1] = f"{sentences[-1]} {stripped}"
        else:
            sentences.append(stripped)
    return sentences


def verify_grounded_reply(
    reply: GroundedReply,
    sources: dict[str, str],
    *,
    min_quote_words: int | None = None,
) -> tuple[GuardFlag, ...]:
    """Extractive grounding: every quote literally in its cited chunk, every sentence claimed.

    Literal containment is compared on the detection skeleton (accents and case folded,
    whitespace collapsed) and a quote must carry at least ``min_quote_words`` words, so an
    empty or one-word quote can never "support" a sentence. Semantic support of a real quote
    remains a declared residual risk (§10.1.9).
    """
    minimum = min_quote_words or guardrail_config().grounding_min_quote_words
    flags: list[GuardFlag] = []
    normalized_sources = {
        section_id: " ".join(detection_skeleton(plain_text(text)).split())
        for section_id, text in sources.items()
    }
    covered: set[str] = set()
    for claim in reply.claims:
        source = normalized_sources.get(claim.section_id)
        quote = " ".join(detection_skeleton(plain_text(claim.quote)).split()).strip(" .")
        if source is None or len(quote.split()) < minimum or quote not in source:
            flags.append("quote_not_in_source")
            continue
        covered.add(_canonical_sentence(claim.sentence))
    sentences = {
        canonical
        for sentence in split_sentences(reply.text)
        if (canonical := _canonical_sentence(sentence))
    }
    if not sentences or sentences - covered:
        flags.append("unsupported_sentence")
    return tuple(dict.fromkeys(flags))
