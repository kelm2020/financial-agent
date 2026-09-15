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
    r"\b(?:elimin|borr|perdon|olvid)\w* (?:el|lo) (?:resto|que falta)\b|"
    # A discount named as a lower price for paying ("precio más bajo" blind phrasing).
    r"\b(?:precio|valor|tarifa)\w* (?:mas )?(?:bajo|menor)\b",
    "anticipo": r"\b(?:anticip|adelant)\w*|\bentrada\b|\bsena\b|"
    r"\b(?:pago|entrega) inicial\b|"
    # The up-front money before the plan starts, in the customer's own register.
    r"\b(?:algo )?(?:de )?(?:guita|plata|dinero) al principi[oa]\b|"
    r"\bprimer pago\b.{0,25}\barranc\w*|"
    r"\b(?:algún |algun )?movimient\w* (?:de dinero )?previ[oa]\b",
    "parcial": r"\bpago parcial\b|\bparcialmente\b|\b(?:un |mi )?monto parcial\b|"
    r"\b(?:un |una )?suma parcial\b|\b(?:dejar|dejando) (?:solo |un )?(?:un )?poco\b|"
    r"\b(?:pag|abon|sald|cancel|entreg|tom|acept|recib|imput|anot|amortiz)\w* "
    r"(?:solo |nada mas que )?"
    r"(?:una |un )?(?:parte|pedazo|pedacito|porcion)\b|"
    r"\bentrega a cuenta\b|\b(?:ir )?pagando de a poco\b|"
    # A smaller amount than the balance, not a smaller monthly installment (that negotiates).
    r"\b(?:pag|abon|deposit)\w* (?:un |una )?(?:monto|importe|suma) (?:mas )?"
    r"(?:chic[oa]|menor|reducid[oa]|baj[oa])\b(?! (?:por mes|mensual|de cuota|cada mes))",
    "refinanciacion": r"\b(?:refinancia\w*|financia\w*|plan|convenio|acuerdo)\b",
    "acreditacion": r"\b(?:acredit\w*|impact\w*|reflej\w*|figur\w*|actualiz\w*|tard\w*|demor\w*)\b|"
    # "Queda registrado el pago": registering only counts as accreditation next to the payment.
    r"\bregistr\w*\b.{0,25}\b(?:el|un|mi) pago\b|\b(?:el|un|mi) pago\b.{0,25}\bregistr\w*",
    "incumplimiento": r"\b(?:incumpl\w*|atras\w*|impag\w*)\b|\b(?:dejo|deje|dejar) de pagar\b|"
    r"\bno (?:llego|llegar) a pagar\b|\bno pag(?:o|as|a|ar|ue|uen)\b|\bcae\w*.*plan\b|"
    # Losing the plan's benefit or the plan itself for a missed installment.
    r"\b(?:cort|quita|pierdo|pierde|perdi)s?\w*\b.{0,30}\b(?:benefici|plan|acuerd|conveni)\w*",
    # "fecha de débito" is a date about a method, not a composite subject: the group stays
    # single-word so its documentation is the word the section carries.
    "fecha": r"\bvenc\w*\b|\bfecha\b|\bdia de pago\b|"
    r"\bpas\w* el pago\b.{0,30}(?:despu|otro dia)",
    "medios": r"\b(?:tarjeta|transferencia|debito|cupon|cheque|efectivo|cripto\w*|"
    r"bitcoin|dolares|euros|moneda|divisa|cajero|pago digital|billeter\w*|wallet|"
    r"usdt|stablecoin\w*)\b|\bmedios? de pago\b",
    # Currency the payment would use. Not co-referent with the vehicle that carries it: a
    # transfer in dollars is a documented vehicle with an undocumented currency, and only the
    # currency is what the customer asks about ("queda en USD?").
    "moneda": r"\b(?:dolares?|euros?|divisa|usdt|usd|stablecoin\w*)\b|"
    r"\bpesos? (?:uruguay|chilen|mexican|colombian|dominican|filipin|cuban)\w*|"
    r"\b(?:en )?(?:la |otra |una )?moneda (?:original|extranjera|de otro pais)\b",
    # Getting money back. The base documents no reimbursement: the anticipo section answers
    # the neighbour question ("se exige anticipo") and never this one.
    "devolucion": r"\b(?:devoluc\w*|reembols\w*|devuelv\w*)\b",
    "vigencia": rf"\b{_VALIDITY}\b.*\b{_OFFER}|\b{_OFFER}.*\b{_VALIDITY}\b|"
    r"\btom\w* cuando (?:quiera|quie)\b",
    "financiero_legal": r"\b(?:impuest\w*|impositiv\w*|fiscal\w*|tribut\w*|ganancias|deduc\w*|"
    r"prescri\w*|embarg\w*|tasa|tna|cft|costo financiero|comision\w*|cesion|entidad|sindicato|"
    r"bienes personales|gananci\w*|afip|arca)\b|"
    r"\b(?:declar|liquid)\w*\b.{0,40}\b(?:afip|arca|ganancias|impuest|bienes personales)\b|"
    r"\b(?:porcentaje|tasa|interes(?:es)?) (?:anual|mensual)\b|"
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
        re.search(
            r"\b(?:cambiar|cambio|correr|mover|modificar|posponer|reagendar|postergar)\b",
            normalized,
        )
    )
    consequence = "incumplimiento" in found and bool(
        re.search(
            r"\b(?:que pasa|que ocurre|consecuencias|si |dejo|dejar|"
            r"en caso que no|no pueda|cortan|pierdo)\b",
            normalized,
        )
    )
    return bool(
        strong
        or change_date
        or consequence
        or (
            "acreditacion" in found
            and re.search(r"\b(?:tarda|demora|cuando|cuanto|plazo)\b", normalized)
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
# reranker and the dense gate: POL-NEG-003 ("Quitas de interés") ranked 13th for "¿me harían
# alguna rebaja?" and 23rd for "¿me descuenten algo?", and 1st and 6th with its term appended;
# POL-NEG-005 stayed below the calibrated gate for "¿tengo que dar algo de entrada?" (0.413)
# and cleared it with "anticipo" (0.550); PAY-MET-002 ranked below FAQ-010 for "¿puedo pagar
# con transferencia y cuánto tarda?" until "acreditación" promoted it to first; FAQ-001 and
# POL-NEG-006 rose for "¿me toman una parte y eliminan el resto?" with "pago parcial"
# (ADR-011). Only measured families are listed.
KB_TERMS: dict[str, str] = {
    "quita": "quita",
    "anticipo": "anticipo",
    "acreditacion": "acreditación",
    "parcial": "pago parcial",
    # Measured with the dense gate: "correr la fecha de débito" left FAQ-003 at 0.496 (below
    # the 0.505 gate) and POL-NEG-007 above it with the generic "fecha de pago"; the section's
    # own wording clears the gate (0.603) with FAQ-003 first (ADR-011).
    "fecha": "cambiar la fecha de vencimiento",
}

# A customer word documented by a corpus word with the same meaning, not a different subject.
# "USDT" is what PAY-MET-003 rejects as criptomonedas and "wallet" as billetera virtual: without
# the equivalence the undocumented-term check abstains on questions the base does answer, and the
# retrieval bridge misses the section (measured: PAY-MET-003 below the calibrated gate at 0.478
# for "¿pago desde mi wallet en USDT?", first with the equivalent terms at 0.605).
TERM_EQUIVALENTS: dict[str, tuple[str, ...]] = {
    "usdt": ("criptomonedas",),
    "stablecoins": ("criptomonedas",),
    "cripto": ("criptomonedas",),
    "bitcoin": ("criptomonedas",),
    "ethereum": ("criptomonedas",),
    "wallet": ("billeteras virtuales",),
    "euros": ("moneda extranjera",),
    "euro": ("moneda extranjera",),
}


def retrieval_query(text: str) -> str:
    """The question plus the knowledge base's terms for the concepts it names in other words. Only
    retrieval reads it: the answer model and the check read the customer's question. A declared
    equivalent (``TERM_EQUIVALENTS``) is the corpus word for a customer word, so it joins the
    bridge terms the same way a family's does ("wallet" is what the base calls a billetera
    virtual)."""
    normalized = detection_skeleton(text)
    terms = [
        KB_TERMS[name]
        for name in sorted(concepts(text))
        if name in KB_TERMS and KB_TERMS[name] not in normalized
    ]
    for word, equivalents in TERM_EQUIVALENTS.items():
        if word in normalized:
            terms.extend(equivalent for equivalent in equivalents if equivalent not in normalized)
    return f"{text}\nTérminos de la base: {', '.join(dict.fromkeys(terms))}" if terms else text
