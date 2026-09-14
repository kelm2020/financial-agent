from __future__ import annotations

import re
from typing import Literal

from app.graph.recorder import TurnBudgetExceeded
from app.graph.state import ConfirmationVerdict
from app.guards.normalize import detection_skeleton
from app.llm.protocol import LLMClient

# §8.3 lexicons, normalized (lowercase, no accents, no punctuation). A negative always beats an
# affirmative (INV-7) and the affirmative verdict can only come from the positive list below.
#
# ADR-010 refines §8.3 in one direction only, towards "other": doubt ("no sé", "lo pienso") and
# affirmative idioms that contain "no" ("no hay problema, dale") neither execute nor destroy the
# draft. They keep it and ask again. Both lead to zero writes; the difference is that a doubtful
# customer is not told "cancelé la propuesta" for something they did not reject.
STRONG_NEGATIVE = (
    "nop",
    "mejor no",
    "cancela",
    "cancelar",
    "cancelalo",
    "dejalo",
    "no quiero",
    "no confirmo",
    "asi no",
)
WEAK_NEGATIVE = ("no", "todavia no", "espera")
NEGATIVE = STRONG_NEGATIVE + WEAK_NEGATIVE
DOUBT = (
    "no se",
    "no estoy seguro",
    "no estoy segura",
    "no estoy tan seguro",
    "no estoy tan segura",
    "no estoy convencido",
    "no estoy convencida",
    "lo pienso",
    "lo tengo que pensar",
    "dejame pensar",
    "dejame pensarlo",
    "pensarlo",
    "no puedo decidir",
    "te respondo despues",
    "te confirmo manana",
    "te aviso",
    "capaz",
    "quizas",
    "quiza",
    "tal vez",
    "supongo",
    "creo que",
    "puede ser",
    "ni idea",
    "no llego",
    "no se si llego",
)
AFFIRMATIVE_IDIOMS = ("no hay problema", "no pasa nada", "no te preocupes", "por que no", "como no")
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
_WEAK_NEGATIVE_MAX_WORDS = 4

type RecheckReason = Literal["doubt", "idiom"]


def normalize_confirmation(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", detection_skeleton(text)).split())


def _contains(normalized: str, values: tuple[str, ...]) -> bool:
    return any(re.search(rf"\b{re.escape(value)}\b", normalized) for value in values)


def _without(normalized: str, values: tuple[str, ...]) -> str:
    for value in values:
        normalized = re.sub(rf"\b{re.escape(value)}\b", " ", normalized)
    return " ".join(normalized.split())


def recheck_reason(text: str) -> RecheckReason | None:
    """Why a pending confirmation must be asked again instead of cancelled or executed."""
    normalized = normalize_confirmation(text)
    if normalized in _STANDALONE_NEGATIVE or _contains(normalized, STRONG_NEGATIVE):
        return None
    if _contains(normalized, DOUBT):
        return "doubt"
    if _contains(normalized, AFFIRMATIVE_IDIOMS) and not _contains(
        _without(normalized, AFFIRMATIVE_IDIOMS), WEAK_NEGATIVE
    ):
        return "idiom"
    return None


def _weak_rejection(normalized: str) -> bool:
    """A bare "no"/"espera" rejects in a short reply or when it opens the reply ("no, dale").

    Inside a longer message it is usually incidental ("te respondo esta tarde, ahora no puedo
    decidir"): the lexicon then does not decide and the turn goes to the model tie-breaker, which
    can only return no or other. Strong negatives ("mejor no", "cancelalo") still win anywhere."""
    if not _contains(normalized, WEAK_NEGATIVE):
        return False
    words = normalized.split()
    opens = any(
        normalized == value or normalized.startswith(f"{value} ") for value in WEAK_NEGATIVE
    )
    return len(words) <= _WEAK_NEGATIVE_MAX_WORDS or opens


def deterministic_confirmation(text: str) -> ConfirmationVerdict | None:
    """Return ``no``/``yes``/``other`` from the lexicons, or ``None`` when nothing decides."""
    normalized = normalize_confirmation(text)
    if normalized in _STANDALONE_NEGATIVE or _contains(normalized, STRONG_NEGATIVE):
        return ConfirmationVerdict(verdict="no")
    if recheck_reason(text) is not None:
        return ConfirmationVerdict(verdict="other")
    if _weak_rejection(normalized):
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
    except TurnBudgetExceeded:
        raise
    except Exception:
        return ConfirmationVerdict(verdict="other")
    # The model may only separate "no" from "other": it never produces an affirmative.
    return ConfirmationVerdict(verdict="no" if model.verdict == "no" else "other")
