from __future__ import annotations

import re

from app.graph.state import ConfirmationVerdict
from app.guards.normalize import detection_skeleton
from app.llm.protocol import LLMClient

# §8.3 lexicons, normalized (lowercase, no accents, no punctuation). The negative list wins
# ALWAYS; the affirmative verdict can only come from the positive list below.
NEGATIVE = (
    "no",
    "nop",
    "todavia no",
    "mejor no",
    "cancela",
    "cancelar",
    "cancelalo",
    "dejalo",
    "espera",
    "no quiero",
    "no confirmo",
    "asi no",
)
POSITIVE = frozenset(
    {
        "si",
        "si confirmo",
        "confirmo",
        "confirma",
        "dale",
        "acepto",
        "de acuerdo",
        "correcto",
        "ok",
        "okey",
        "listo",
        "perfecto",
        "obvio",
        "mandale",
    }
)
POSITIVE_WORDS = frozenset(phrase for phrase in POSITIVE if " " not in phrase)
_POSITIVE_PHRASES = tuple(sorted((p for p in POSITIVE if " " in p), key=len, reverse=True))
# "para"/"pará" is a stop request only as the whole message: inside a question ("¿y para
# débito?") it is a preposition, and treating it as "no" would cancel a draft on a question.
_STANDALONE_NEGATIVE = frozenset({"para", "para ya", "frena"})


def normalize_confirmation(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", detection_skeleton(text)).split())


def deterministic_confirmation(text: str) -> ConfirmationVerdict | None:
    """Return ``no``/``yes`` from the lexicons, or ``None`` when neither list decides."""
    normalized = normalize_confirmation(text)
    if normalized in _STANDALONE_NEGATIVE or any(
        re.search(rf"\b{re.escape(value)}\b", normalized) for value in NEGATIVE
    ):
        return ConfirmationVerdict(verdict="no")
    remainder = f" {normalized} "
    phrases = 0
    for phrase in _POSITIVE_PHRASES:
        while f" {phrase} " in remainder:
            remainder = remainder.replace(f" {phrase} ", " ", 1)
            phrases += 1
    words = remainder.split()
    if (phrases or words) and phrases + len(words) <= 3 and all(w in POSITIVE_WORDS for w in words):
        return ConfirmationVerdict(verdict="yes")
    return None


async def parse_confirmation(text: str, llm: LLMClient | None) -> ConfirmationVerdict:
    verdict = deterministic_confirmation(text)
    if verdict is not None:
        return verdict
    if llm is None:
        return ConfirmationVerdict(verdict="other")
    try:
        model = await llm.complete(
            task="confirmation",
            messages=(
                {
                    "role": "system",
                    "content": "Clasificá sólo como no u other; yes no es válido.",
                },
                {"role": "user", "content": text},
            ),
            response_model=ConfirmationVerdict,
        )
    except Exception:
        return ConfirmationVerdict(verdict="other")
    # The model may only separate "no" from "other": it never produces an affirmative.
    return ConfirmationVerdict(verdict="no" if model.verdict == "no" else "other")
