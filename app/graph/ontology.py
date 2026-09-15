"""Business concepts used for routing and risk, never as evidence of an answer.

These families deliberately do not map phrases to section IDs. Unknown modifiers remain
in the original question for the semantic support verifier to reject or resolve.
"""

from __future__ import annotations

import re
from typing import Literal

from app.guards.normalize import detection_skeleton

CONCEPTS: dict[str, str] = {
    "quita": r"\b(?:quita|descuento|reduccion|rebaja|condonacion)\w*\b|\bperdon\w*.*interes",
    "anticipo": r"\b(?:anticipo|entrada|adelanto)\b|\b(?:pago|entrega) inicial\b",
    "parcial": r"\bpago parcial\b|\b(?:pagar|pagando|abonar|abonando) (?:una )?parte\b|"
    r"\bporcentaje\b|"
    r"\bentrega a cuenta\b|\b(?:ir )?pagando de a poco\b",
    "refinanciacion": r"\b(?:refinancia\w*|financia\w*|plan|convenio|acuerdo)\b",
    "acreditacion": r"\b(?:acredit\w*|impact\w*|reflej\w*|figur\w*|actualiz\w*)\b",
    "incumplimiento": r"\b(?:incumpl\w*|atras\w*|impag\w*)\b|\bdejo de pagar\b|"
    r"\bdejar de pagar\b|\bno (?:llego|llegar) a pagar\b|\bcae\w*.*plan\b",
    "fecha": r"\bvenc\w*\b|\b(?:fecha|dia) de pago\b",
    "medios": r"\b(?:tarjeta|transferencia|debito|cupon)\b|\bmedios? de pago\b",
    "vigencia": r"\b(?:vigencia|validez|valida|vale|valen)\b.*\b(?:oferta|propuesta)\b|"
    r"\b(?:oferta|propuesta)\b.*\b(?:vigencia|vale|vence|dura)\b",
    "financiero_legal": r"\b(?:impuest\w*|ganancias|deduc\w*|prescri\w*|embarg\w*|"
    r"tasa|cft|costo financiero|cesion|entidad|sindicato)\b",
}


def concepts(text: str) -> frozenset[str]:
    normalized = detection_skeleton(text)
    return frozenset(name for name, pattern in CONCEPTS.items() if re.search(pattern, normalized))


def policy_risk(text: str, topic: str = "any") -> Literal["low", "high"]:
    high = {
        "quita",
        "anticipo",
        "parcial",
        "refinanciacion",
        "incumplimiento",
        "fecha",
        "vigencia",
        "financiero_legal",
    }
    return "high" if topic in {"negociacion", "escalamiento"} or concepts(text) & high else "low"


def policy_request(text: str) -> bool:
    normalized = detection_skeleton(text)
    found = concepts(text)
    strong = found & {"quita", "anticipo", "parcial", "vigencia", "financiero_legal"}
    change_date = "fecha" in found and bool(
        re.search(r"\b(?:cambiar|cambio|correr|mover|modificar|posponer)\b", normalized)
    )
    consequence = "incumplimiento" in found and bool(
        re.search(r"\b(?:que pasa|que ocurre|consecuencias|si |dejo|dejar)\b", normalized)
    )
    return bool(
        strong
        or change_date
        or consequence
        or (
            "acreditacion" in found and re.search(r"\b(?:tarda|demora|cuando|cuanto)\b", normalized)
        )
    )


def mixed_request(text: str) -> bool:
    """Conservative multi-source contract: clarify before answering either part."""
    normalized = detection_skeleton(text)
    account = bool(
        re.search(
            r"\bcuanto debo\b|\b(?:mi|mis) (?:saldo|deuda|cuotas impagas|opciones)\b|"
            r"\bopciones concretas\b|\bopciones tengo\b",
            normalized,
        )
    )
    policy = policy_request(text) or "medios" in concepts(text)
    return account and policy and bool(re.search(r"\by\b|\bademas\b|[?].*[?]", normalized))
