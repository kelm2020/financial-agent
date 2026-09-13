from __future__ import annotations

import re

from app.graph.state import RouteResult
from app.guards.normalize import detection_skeleton

_OPTION = re.compile(r"\bOPT-[A-Z0-9]{2,10}\b", re.IGNORECASE)
_INSTALLMENTS = re.compile(r"\b(\d{1,2})\s+cuotas?\b", re.IGNORECASE)


def _has(text: str, *values: str) -> bool:
    return any(value in text for value in values)


def route_turn(text: str) -> RouteResult:
    """Deterministic routing table (§8.2). The model only classifies what this leaves ambiguous."""
    normalized = detection_skeleton(text)
    option_match = _OPTION.search(text)
    installments_match = _INSTALLMENTS.search(text)
    option_id = option_match.group().upper() if option_match else None
    installments = int(installments_match.group(1)) if installments_match else None

    if _has(normalized, "cuando derivan", "motivos de derivacion"):
        return RouteResult(intent="consulta_general", topic="escalamiento")
    if _has(normalized, "abogado", "demanda", "judicial", "carta documento"):
        return RouteResult(intent="pedido_humano", escalation_motivo="amenaza_legal")
    if _has(normalized, "vulnerab", "me quede sin trabajo", "estoy enfermo", "estoy enferma"):
        return RouteResult(intent="pedido_humano", escalation_motivo="vulnerabilidad")
    if _has(normalized, "una persona", "un operador", "un asesor", "un humano", "con alguien"):
        return RouteResult(intent="pedido_humano", escalation_motivo="pedido_explicito")
    choosing = _has(normalized, "quiero", "elijo", "opcion", "la de ", "me quedo", "tomo", "dale")
    if option_id or (installments is not None and choosing):
        return RouteResult(intent="aceptar_opcion", option_id=option_id, installments=installments)
    if _has(normalized, "lo que pueda", "no se cuanto", "algo puedo"):
        return RouteResult(intent="ambiguo")
    if _has(normalized, "quita", "anticipo", "requisitos para refinanciar", "politica de cuotas"):
        return RouteResult(intent="consulta_general", topic="negociacion")
    if _has(normalized, "opciones", "alternativas", "cuotas", "negoci"):
        return RouteResult(intent="negociacion", installments=installments)
    if _has(normalized, "cuanto debo", "saldo", "deuda", "vencimiento"):
        return RouteResult(intent="consulta_deuda")
    if _has(
        normalized,
        "tarjeta",
        "acredit",
        "transferencia",
        "debito",
        "cupon",
        "medio de pago",
        "medios de pago",
    ):
        return RouteResult(intent="consulta_general", topic="medios_pago")
    if _has(normalized, "donde llamo", "pagar", "politica"):
        return RouteResult(intent="consulta_general", topic="medios_pago")
    if _has(normalized, "mundial", "capital de", "receta", "futbol", "clima", "partido"):
        return RouteResult(intent="fuera_de_dominio")
    if re.search(r"\b(?:hola|buen dia|buenas|chau|gracias)\b", normalized):
        return RouteResult(intent="saludo_despedida")
    return RouteResult(intent="ambiguo")
