from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from collections.abc import Sequence
from functools import lru_cache
from typing import Protocol, cast

import snowballstemmer  # type: ignore[import-untyped]

_WORD = re.compile(r"[a-z0-9]+")
# Closed-class Spanish function words only (articles, prepositions, conjunctions and clitic
# pronouns). Content words never belong here, and neither does any synonym table: lexical
# normalization is accent folding plus the Snowball Spanish stemmer, so nothing in this module
# is tuned against evaluation queries. Postgres stores these same lexemes (migration 0003).
_STOPWORDS = frozenset(
    {
        "a", "al", "ante", "con", "contra", "de", "del", "desde", "e", "el", "en", "entre",
        "es", "esa", "ese", "eso", "esta", "este", "esto", "ha", "hay", "la", "las", "le",
        "les", "lo", "los", "me", "mi", "mis", "ni", "nos", "o", "para", "pero", "por",
        "que", "se", "si", "sin", "sobre", "su", "sus", "te", "tu", "tus", "u", "un", "una",
        "unas", "uno", "unos", "y", "ya",
    }
)  # fmt: skip


class _Stemmer(Protocol):
    def stemWords(self, words: list[str]) -> list[str]: ...  # third-party method name


@lru_cache(maxsize=1)
def _stemmer() -> _Stemmer:
    return cast(_Stemmer, snowballstemmer.stemmer("spanish"))


def normalize_text(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(character for character in decomposed if not unicodedata.combining(character))


def words(text: str) -> tuple[str, ...]:
    """Accent-folded lowercase ``[a-z0-9]+`` words (hence safe inside ``to_tsquery``)."""
    return tuple(_WORD.findall(normalize_text(text)))


def tokenize(text: str) -> tuple[str, ...]:
    content = [word for word in words(text) if len(word) > 1 and word not in _STOPWORDS]
    return tuple(_stemmer().stemWords(content))


def hashing_embedding(text: str, dimensions: int = 256) -> tuple[float, ...]:
    """Deterministic bag-of-stems vector for mechanics tests. It carries no semantics."""
    values = [0.0] * dimensions
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode()).digest()
        index = int.from_bytes(digest[:4]) % dimensions
        sign = 1.0 if digest[4] & 1 else -1.0
        values[index] += sign
    norm = math.sqrt(sum(value * value for value in values))
    if norm == 0:
        return tuple(values)
    return tuple(value / norm for value in values)


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    """True cosine similarity. Provider vectors are only approximately unit-norm (and are
    rounded in the cache), so a bare dot product would drift from pgvector's ``<=>``."""
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    norms = math.sqrt(sum(a * a for a in left)) * math.sqrt(sum(b * b for b in right))
    return dot / norms if norms else 0.0
