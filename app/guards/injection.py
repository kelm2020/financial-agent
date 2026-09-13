from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator

from app.guards.codes import GuardFlag
from app.guards.config import GuardrailConfig, guardrail_config

GuardVerdict = Literal["allow", "restrict", "deflect"]

# Patterns run on the detection skeleton (accents removed, casefolded, confusables folded).
INJECTION_PATTERNS = tuple(
    re.compile(pattern)
    for pattern in (
        r"\bignora (?:(?:tus|las|todas las) (?:instrucciones|reglas)|todo lo anterior|todo)",
        r"\bolvida (?:tus|las) (?:instrucciones|reglas)\b",
        r"\ba partir de ahora (?:eres|sos|actua|actuas|vas a ser)",
        (
            r"\b(?:sos|eres|actua como) un (?:sistema|agente|asistente|modelo) "
            r"sin (?:restricciones|reglas|limites)"
        ),
        r"\b(?:modo|rol) (?:desarrollador|developer|dios|sin restricciones|jailbreak)\b",
        (
            r"\b(?:revela|revelame|mostra|mostrame|imprime|decime|pasame|copia) "
            r"(?:el |tu |tus |las |los )?"
            r"(?:system prompt|prompt|instrucciones internas|instrucciones del sistema"
            r"|reglas internas)"
        ),
        r"\b(?:usa|selecciona|consulta|cambia a) (?:la cuenta |el cliente )?cust-\d{5}\b",
        r"\bignore (?:all|your|the|any) (?:previous |prior )?(?:instructions|rules)\b",
        r"\b(?:you are now|act as) (?:an? )?(?:unrestricted|jailbroken|dan)\b",
        r"\b(?:reveal|print|show) (?:your |the )?(?:system prompt|hidden instructions)\b",
        r"<</?(?:datos_kb|datos_backend|system)",
    )
)
_CUSTOMER_ID = re.compile(r"\bcust-\d{5}\b")
_SPACED_LETTERS = re.compile(r"\b(?:\w ){4,}\w\b")
# Letter-by-letter spacing loses word boundaries after whitespace normalization, so spaced runs
# are compared compactly against these markers.
_COMPACT_MARKERS = (
    "ignoratusinstrucciones",
    "ignoralasinstrucciones",
    "ignoratodo",
    "olvidatusinstrucciones",
    "systemprompt",
    "instruccionesinternas",
    "ignoreallpreviousinstructions",
    "ignoreyourinstructions",
)


def spaced_injection(detection_text: str) -> bool:
    runs = (match.group(0).replace(" ", "") for match in _SPACED_LETTERS.finditer(detection_text))
    return any(marker in run for run in runs for marker in _COMPACT_MARKERS)


class GuardRuleResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    restrict: bool = False
    injection_matched: bool = False
    flags: tuple[GuardFlag, ...] = ()


class GuardModelResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    label: Literal["benign", "injection", "jailbreak", "exfiltracion"] = "benign"
    confidence: float = 0.0

    @field_validator("confidence")
    @classmethod
    def valid_confidence(cls, value: float) -> float:
        if not 0 <= value <= 1:
            raise ValueError("confidence must be between zero and one")
        return value


class GuardDecision(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    verdict: GuardVerdict
    flags: tuple[GuardFlag, ...] = ()


def mentions_foreign_customer(detection_text: str, session_customer_id: str | None) -> bool:
    session = (session_customer_id or "").casefold()
    return any(value != session for value in _CUSTOMER_ID.findall(detection_text))


def evaluate_rules(
    detection_text: str,
    preflight_flags: tuple[GuardFlag, ...] = (),
    *,
    session_customer_id: str | None = None,
) -> GuardRuleResult:
    matched = (
        any(pattern.search(detection_text) for pattern in INJECTION_PATTERNS)
        or spaced_injection(detection_text)
        or mentions_foreign_customer(detection_text, session_customer_id)
    )
    boundary_restricts = bool({"sensitive_input", "encoded_payload"} & set(preflight_flags))
    flags: list[GuardFlag] = list(preflight_flags)
    if matched:
        flags.append("suspected_injection")
    return GuardRuleResult(
        restrict=matched or boundary_restricts,
        injection_matched=matched,
        flags=tuple(dict.fromkeys(flags)),
    )


def resolve_guard(
    rules: GuardRuleResult,
    model: GuardModelResult,
    *,
    config: GuardrailConfig | None = None,
) -> GuardDecision:
    """Deterministic precedence (§10.1.3). A model verdict can add restriction, never lift it."""
    thresholds = config or guardrail_config()
    flags: list[GuardFlag] = list(rules.flags)
    model_restricts = (
        model.label != "benign" and model.confidence >= thresholds.classifier_medium_confidence
    )
    model_deflects = (
        model.label in {"jailbreak", "exfiltracion"}
        and model.confidence >= thresholds.classifier_high_confidence
    )
    if model_deflects and (
        rules.injection_matched or model.confidence >= thresholds.classifier_solo_deflect_confidence
    ):
        flags.append("injection_deflected")
        return GuardDecision(verdict="deflect", flags=tuple(dict.fromkeys(flags)))
    if rules.restrict or model_restricts:
        if rules.injection_matched or model_restricts:
            flags.append("suspected_injection")
        return GuardDecision(verdict="restrict", flags=tuple(dict.fromkeys(flags)))
    return GuardDecision(verdict="allow", flags=tuple(dict.fromkeys(flags)))
