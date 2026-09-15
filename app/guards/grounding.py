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


def _quote_key(text: str) -> str:
    return " ".join(detection_skeleton(plain_text(text)).split()).strip(" .")


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


# Words a paraphrase may take from the question without stating anything about the policy.
_ECHO_IGNORED = frozenset(
    {"puedo", "puede", "pueda", "tengo", "tiene", "hacer", "hacen", "cuand", "cuant", "donde",
     "quier", "esta", "estan", "como", "para", "pero", "sobre", "desde", "hasta", "todo", "toda",
     "hola", "buena", "graci"}
)  # fmt: skip


def content_stems(text: str) -> set[str]:
    return {word[:5] for word in re.findall(r"[a-z]{4,}", detection_skeleton(text))} - _ECHO_IGNORED


_content_stems = content_stems


def echoed_terms(sentence: str, question: str, source: str) -> set[str]:
    """Terms of an answer sentence taken from the question but absent from its cited section.

    A real quote does not make a sentence true: "La comisión del asesor es del 10 % del saldo
    total" cites "desde el 10 % del saldo total" and restates the question as if the policy said
    it, and "Sí, hay descuentos por transferencia o cupón" answers a question about discounts with
    a section that never mentions them. One such term is enough to reject the sentence: a lost
    paraphrase only costs a regeneration or the verbatim extract.
    """
    return (_content_stems(sentence) & _content_stems(question)) - _content_stems(
        plain_text(source)
    )


def sentence_supported(sentence: str, source: str) -> bool:
    """At least two thirds of a claim sentence's content terms appear in its cited section.

    A literal quote proves the section says something, not that the sentence says the same. A
    sentence that carries another section's content under this section's quote ("Podés cambiar
    el medio de pago…" cited as FAQ-003, with a real FAQ-003 quote) keeps about half its terms;
    a faithful paraphrase keeps nearly all. Very short sentences are left to the quote.

    It is a lexical proxy, not entailment: the agent uses it to drop a claim from an answer, never
    to block an output (as a block it rejected a correct held-out output, ADR-010).
    """
    terms = content_stems(sentence)
    if len(terms) < 3:
        return True
    missing = terms - content_stems(plain_text(source))
    return 3 * len(missing) <= len(terms)


def quote_verified(quote: str, source: str, *, min_quote_words: int | None = None) -> bool:
    """The quote is literal in the source (accents, case and whitespace folded) and carries at
    least ``min_quote_words`` words, or is a whole statement of the source: a table row or list
    item ("Prejudicial: requiere operador.") is complete support even when it is shorter."""
    minimum = min_quote_words or guardrail_config().grounding_min_quote_words
    key = _quote_key(quote)
    if key not in " ".join(detection_skeleton(plain_text(source)).split()):
        return False
    return len(key.split()) >= minimum or key in {
        _quote_key(sentence) for sentence in split_sentences(plain_text(source))
    }


def verify_grounded_reply(
    reply: GroundedReply,
    sources: dict[str, str],
    *,
    min_quote_words: int | None = None,
    question: str = "",
) -> tuple[GuardFlag, ...]:
    """Extractive grounding: every quote literally in its cited chunk, every sentence claimed.

    Literal containment is compared on the detection skeleton (accents and case folded,
    whitespace collapsed) and a quote must carry at least ``min_quote_words`` words, so an
    empty or one-word quote can never "support" a sentence. Semantic support of a real quote
    remains a declared residual risk (§10.1.9).
    """
    flags: list[GuardFlag] = []
    covered: set[str] = set()
    for claim in reply.claims:
        source = sources.get(claim.section_id)
        if source is None or not quote_verified(
            claim.quote, source, min_quote_words=min_quote_words
        ):
            flags.append("quote_not_in_source")
            continue
        if question and echoed_terms(claim.sentence, question, sources[claim.section_id]):
            continue  # the sentence stays uncovered: unsupported_sentence
        covered.add(_canonical_sentence(claim.sentence))
        # A claim may group several sentences under one quote; each of them is covered.
        covered.update(_canonical_sentence(part) for part in split_sentences(claim.sentence))
    sentences = {
        canonical
        for sentence in split_sentences(reply.text)
        if (canonical := _canonical_sentence(sentence))
    }
    if not sentences or sentences - covered:
        flags.append("unsupported_sentence")
    return tuple(dict.fromkeys(flags))
