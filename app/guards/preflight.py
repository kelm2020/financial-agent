from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from app.guards.codes import GuardFlag
from app.guards.config import guardrail_config
from app.guards.normalize import detection_skeleton, normalize_visible

_JWT = re.compile(r"\beyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+\b")
_CARD = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
# A DNI is only redacted when the customer says it is one: "1.240.000" is an amount in a
# collections chat, and redacting it would both corrupt the turn and restrict it.
_DNI = re.compile(
    r"(?i)\b(?:dni|d\.n\.i\.?|documento(?: nacional de identidad)?|nro\.? de documento)"
    r"\s*(?:es|:|n[°º]?|numero|número)?\s*(\d{1,2}[.]?\d{3}[.]?\d{3})(?!\d)"
)
_CVV = re.compile(r"(?i)\b(?:cvv|cvc|codigo de seguridad|código de seguridad)\s*[:#-]?\s*\d{3,4}\b")
_SECRET = re.compile(r"(?i)\b(?:token|clave|password|contraseña)\s*(?:es|:|=)\s*\S+")
_ENCODED = re.compile(r"(?:[A-Fa-f0-9]{80,}|[A-Za-z0-9+/]{80,}={0,2})")


class PreflightPolicy(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    max_characters: int = Field(
        default_factory=lambda: guardrail_config().message_max_characters, ge=1, le=100_000
    )


class PreflightResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    flags: tuple[GuardFlag, ...] = ()
    rejected: bool = False
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PreflightOutcome:
    sanitized_text: str
    detection_text: str
    result: PreflightResult


def _luhn(candidate: str) -> bool:
    digits = [int(value) for value in re.sub(r"\D", "", candidate)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        value = digit * 2 if index % 2 == parity else digit
        checksum += value - 9 if value > 9 else value
    return checksum % 10 == 0


def _redact_cards(text: str) -> tuple[str, bool]:
    changed = False

    def replace(match: re.Match[str]) -> str:
        nonlocal changed
        if not _luhn(match.group()):
            return match.group()
        changed = True
        return "[TARJETA REDACTADA]"

    return _CARD.sub(replace, text), changed


def preflight_message(text: str, *, policy: PreflightPolicy) -> PreflightOutcome:
    visible = normalize_visible(text)
    if len(visible) > policy.max_characters:
        return PreflightOutcome(
            sanitized_text="",
            detection_text="",
            result=PreflightResult(rejected=True, reason="message_too_long"),
        )

    flags: list[GuardFlag] = []
    sanitized, card_changed = _redact_cards(visible)
    sensitive = card_changed
    for pattern, replacement in (
        (_CVV, "[CÓDIGO REDACTADO]"),
        (_JWT, "[TOKEN REDACTADO]"),
        (_SECRET, "[SECRETO REDACTADO]"),
    ):
        sanitized, count = pattern.subn(replacement, sanitized)
        sensitive = sensitive or count > 0
    sanitized, dni_count = _DNI.subn(
        lambda match: match.group(0).replace(match.group(1), "[DNI REDACTADO]"), sanitized
    )
    sensitive = sensitive or dni_count > 0
    if sensitive:
        flags.append("sensitive_input")
    if _ENCODED.search(sanitized):
        flags.append("encoded_payload")
    return PreflightOutcome(
        sanitized_text=sanitized,
        detection_text=detection_skeleton(sanitized),
        result=PreflightResult(flags=tuple(flags)),
    )
