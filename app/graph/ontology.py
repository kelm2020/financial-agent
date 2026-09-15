"""Business concepts used for routing and risk, never as evidence of an answer.

These families deliberately do not map phrases to section IDs. Unknown modifiers remain
in the original question for the semantic support verifier to reject or resolve.

A family names word stems, not phrasings: every inflection of its verbs and nouns names the
concept ("descuenten", "reducen", "adelantar"), because customers conjugate what a lexicon of
nouns only lists once (policy regression H01, ADR-011).
"""

from __future__ import annotations

import re
from typing import Literal

from app.guards.normalize import detection_skeleton
from app.rag.text import STOPWORDS

_OFFER = r"(?:ofert|propuest|propusi|ofrecie)\w*"
_VALIDITY = (
    r"(?:vigencia|validez|valida|vale|valen|vence|caduca|expira|dura|fija|fecha limite|"
    r"hasta cuando)"
)

CONCEPTS: dict[str, str] = {
    "quita": r"\b(?:quita|descuent|reduc|rebaj|condon)\w*|\bperdon\w*.*interes|"
    r"\b(?:elimin|borr|perdon|olvid)\w* (?:el|lo) (?:resto|que falta)\b",
    "anticipo": r"\b(?:anticip|adelant)\w*|\bentrada\b|\b(?:pago|entrega) inicial\b",
    "parcial": r"\bpago parcial\b|\bparcialmente\b|\bporcentaje\b|"
    r"\b(?:pag|abon|sald|cancel|entreg|tom|acept|recib)\w* (?:solo |nada mas que )?"
    r"(?:una |un )?(?:parte|pedazo|pedacito|porcion)\b|"
    r"\bentrega a cuenta\b|\b(?:ir )?pagando de a poco\b",
    "refinanciacion": r"\b(?:refinancia\w*|financia\w*|plan|convenio|acuerdo)\b",
    "acreditacion": r"\b(?:acredit\w*|impact\w*|reflej\w*|figur\w*|actualiz\w*)\b",
    "incumplimiento": r"\b(?:incumpl\w*|atras\w*|impag\w*)\b|\b(?:dejo|deje|dejar) de pagar\b|"
    r"\bno (?:llego|llegar) a pagar\b|\bno pag(?:o|as|a|ar|ue|uen)\b|\bcae\w*.*plan\b",
    "fecha": r"\bvenc\w*\b|\b(?:fecha|dia) de pago\b",
    "medios": r"\b(?:tarjeta|transferencia|debito|cupon|cheque|efectivo|deposit\w*|cripto\w*|"
    r"bitcoin|dolares|moneda|cajero|pago digital|billeter\w*)\b|\bmedios? de pago\b",
    "vigencia": rf"\b{_VALIDITY}\b.*\b{_OFFER}|\b{_OFFER}.*\b{_VALIDITY}\b",
    "financiero_legal": r"\b(?:impuest\w*|impositiv\w*|fiscal\w*|tribut\w*|ganancias|deduc\w*|"
    r"prescri\w*|embarg\w*|tasa|tna|cft|costo financiero|comision\w*|cesion|entidad|sindicato)\b|"
    r"\binteres(?:es)? (?:anual|mensual|punitorio|compensatorio)\w*",
}


def concepts(text: str) -> frozenset[str]:
    normalized = detection_skeleton(text)
    return frozenset(name for name, pattern in CONCEPTS.items() if re.search(pattern, normalized))


def concept_trigger_words(text: str, *families: str) -> dict[str, list[tuple[str, ...]]]:
    """The content-word groups of ``text`` that triggered the requested ontological concepts.

    The abstention of a model-free policy answer checks them against the retrieved corpus
    (app/graph/nodes/respond.py): "¿me cobran comisión?" retrieves the payment-method section,
    but if no retrieved section documents comisiones, the material does not answer the
    question. Each regex match is one group: a multi-word match ("interés anual", "costo
    financiero") is one subject whose words must co-occur in one section, while one-word
    matches of the same family are co-referent names of its topic ("cripto", "bitcoin"), so
    one of them documented is the topic documented. Function words never belong to a group,
    and each word is stemmed exactly as the index lexes it.
    """
    normalized = detection_skeleton(text)
    triggers: dict[str, list[tuple[str, ...]]] = {}
    for name, pattern in CONCEPTS.items():
        if families and name not in families:
            continue
        groups = triggers.setdefault(name, [])
        for match in re.finditer(pattern, normalized):
            # The vocabulary bridge already mapped these synonyms to the corpus's own terms.
            words = tuple(
                dict.fromkeys(
                    word
                    for word in re.findall(r"[a-z0-9]+", match.group())
                    if word not in STOPWORDS and word not in KB_TERMS.values()
                )
            )
            if words:
                groups.append(words)
    return {name: groups for name, groups in triggers.items() if groups}


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
    # The customer's own data: the balance, the debt or the options offered to them.
    account = bool(
        re.search(
            r"\bcuanto debo\b|\b(?:mi|mis) (?:saldo|deuda|cuotas impagas|opciones|alternativas)\b|"
            r"\b(?:que|cual es (?:el|mi)) saldo\b|\bsaldo (?:que )?(?:tengo|pendiente)\b|"
            r"\b(?:opciones|alternativas) (?:concretas|habilitadas|tengo)\b",
            normalized,
        )
    )
    policy = policy_request(text) or "medios" in concepts(text)
    return account and policy and bool(re.search(r"\by\b|\bademas\b|[?].*[?]", normalized))


# The knowledge base's word for a concept customers name otherwise. Measured with the Cohere
# reranker: POL-NEG-003 ("Quitas de interés") ranked 13th for "¿me harían alguna rebaja?" and 23rd
# for "¿me descuenten algo?", and 1st and 6th with its term appended (ADR-011). Only measured
# families are listed.
KB_TERMS: dict[str, str] = {"quita": "quita"}


def retrieval_query(text: str) -> str:
    """The question plus the knowledge base's terms for the concepts it names in other words. Only
    retrieval reads it: the answer model and the check read the customer's question."""
    normalized = detection_skeleton(text)
    terms = [
        KB_TERMS[name]
        for name in sorted(concepts(text))
        if name in KB_TERMS and KB_TERMS[name] not in normalized
    ]
    return f"{text}\nTérminos de la base: {', '.join(terms)}" if terms else text
