from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from decimal import ROUND_HALF_UP, Decimal
from functools import lru_cache
from itertools import pairwise
from pathlib import Path
from typing import Annotated, Literal, get_args

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.tools.schemas import Customer, Debt, EscalationMotivo, MedioPago, PaymentOption

type Segment = Literal["mora_temprana", "mora_media", "mora_tardia", "prejudicial"]
type Decision = Literal["aceptable", "contraoferta", "rechazada", "derivar"]

RULES_PATH = Path(__file__).with_name("rules.yaml")
SEGMENTS: tuple[Segment, ...] = ("mora_temprana", "mora_media", "mora_tardia", "prejudicial")
PAYMENT_METHODS: tuple[MedioPago, ...] = get_args(MedioPago)
_PESO = Decimal("1")
_HUNDRED = Decimal("100")


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True)


class SegmentRule(_StrictModel):
    dias_min: int = Field(ge=1)
    dias_max: int | None = Field(ge=1)

    @model_validator(mode="after")
    def ordered(self) -> SegmentRule:
        if self.dias_max is not None and self.dias_max < self.dias_min:
            raise ValueError("dias_max debe ser mayor o igual que dias_min")
        return self


class RefinancingRule(_StrictModel):
    cuotas_max: int | None = Field(default=None, ge=1)
    quita_interes_max_pct: Decimal = Field(default=Decimal(0), ge=0, le=100)
    anticipo_min_pct: Decimal = Field(default=Decimal(0), ge=0, le=100)
    requiere_operador: bool = False

    @model_validator(mode="after")
    def operator_or_limits(self) -> RefinancingRule:
        if not self.requiere_operador and self.cuotas_max is None:
            raise ValueError("Un segmento sin operador necesita cuotas_max")
        return self


class RefinancingRules(_StrictModel):
    mora_temprana: RefinancingRule
    mora_media: RefinancingRule
    mora_tardia: RefinancingRule
    prejudicial: RefinancingRule
    anticipo_requerido_desde_cuotas: int = Field(ge=1)

    def for_segment(self, segment: Segment) -> RefinancingRule:
        rule: RefinancingRule = getattr(self, segment)
        return rule


class GlobalLimits(_StrictModel):
    cuota_minima_ars: Decimal = Field(gt=0)
    monto_minimo_pago_parcial_pct: Decimal = Field(gt=0, le=100)
    vigencia_oferta_horas: int = Field(gt=0)
    max_acuerdos_activos: int = Field(ge=1)
    primera_cuota_dias_min: int = Field(ge=0)
    primera_cuota_dias_max: int = Field(ge=0)
    redondeo_cuota: Literal["al_peso"]
    baja_plan_dias_impago: int = Field(gt=0)


class PaymentMethodRule(_StrictModel):
    cuotas_max: int | None = Field(ge=1)


class _EscalationRuleBase(_StrictModel):
    motivo: EscalationMotivo
    descripcion: str = Field(min_length=1)
    policy_refs: tuple[str, ...] = Field(min_length=1)


type SignalName = Literal[
    "pedido_explicito_de_humano",
    "presenta_reclamo",
    "cliente_declara_vulnerabilidad",
    "menciona_abogado_o_demanda",
    "indicios_fraude",
    "outcome_escritura_desconocido",
    "falla_tecnica",
    "tres_intentos_sin_avance",
]


class SignalEscalationRule(_EscalationRuleBase):
    regla: SignalName


class CustomerFlagEscalationRule(_EscalationRuleBase):
    regla: Literal["identidad_no_verificada", "marca_vulnerabilidad"]


class BrokenAgreementsEscalationRule(_EscalationRuleBase):
    regla: Literal["acuerdos_previos_incumplidos"]
    minimo: int = Field(ge=1)


class SegmentEscalationRule(_EscalationRuleBase):
    regla: Literal["segmento"]
    segmentos: tuple[Segment, ...] = Field(min_length=1)


type EscalationRule = Annotated[
    SignalEscalationRule
    | CustomerFlagEscalationRule
    | BrokenAgreementsEscalationRule
    | SegmentEscalationRule,
    Field(discriminator="regla"),
]


class PolicySignals(_StrictModel):
    cliente_declara_vulnerabilidad: bool = False
    menciona_abogado_o_demanda: bool = False
    pedido_explicito_de_humano: bool = False
    tres_intentos_sin_avance: bool = False
    presenta_reclamo: bool = False
    indicios_fraude: bool = False
    falla_tecnica: bool = False
    outcome_escritura_desconocido: bool = False


class PolicyRules(_StrictModel):
    version: str
    segmentos: dict[Segment, SegmentRule]
    refinanciacion: RefinancingRules
    recargo_financiacion_pct: dict[str, Decimal]
    limites_globales: GlobalLimits
    medios_pago: dict[MedioPago, PaymentMethodRule]
    escalamiento_obligatorio: tuple[EscalationRule, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def consistent(self) -> PolicyRules:
        _validate_segments(self.segmentos)
        _validate_surcharges(self.recargo_financiacion_pct, self._max_installments())
        if set(self.medios_pago) != set(PAYMENT_METHODS):
            raise ValueError("medios_pago debe definir todos los medios habilitados")
        names = [rule.regla for rule in self.escalamiento_obligatorio]
        if len(names) != len(set(names)):
            raise ValueError("Cada regla de escalamiento debe aparecer una sola vez")
        missing = set(PolicySignals.model_fields) - set(names)
        if missing:
            raise ValueError(f"Señales sin regla de escalamiento: {sorted(missing)}")
        return self

    def _max_installments(self) -> int:
        return max(
            rule.cuotas_max
            for rule in (self.refinanciacion.for_segment(segment) for segment in SEGMENTS)
            if rule.cuotas_max is not None
        )


def _validate_segments(segments: Mapping[Segment, SegmentRule]) -> None:
    if set(segments) != set(SEGMENTS):
        raise ValueError("segmentos debe definir los cuatro segmentos")
    ordered = sorted(segments.values(), key=lambda rule: rule.dias_min)
    if ordered[0].dias_min != 1:
        raise ValueError("Los segmentos deben empezar en 1 día de mora")
    for current, following in pairwise(ordered):
        if current.dias_max is None or following.dias_min != current.dias_max + 1:
            raise ValueError("Los segmentos deben ser contiguos y sin solapamiento")
    if ordered[-1].dias_max is not None:
        raise ValueError("El último segmento no puede tener techo de días")


def _parse_interval(interval: str) -> tuple[int, int]:
    lower, _, upper = interval.partition("-")
    return int(lower), int(upper or lower)


def _validate_surcharges(surcharges: Mapping[str, Decimal], max_installments: int) -> None:
    covered: list[int] = []
    for interval in surcharges:
        lower, upper = _parse_interval(interval)
        if lower > upper:
            raise ValueError(f"Intervalo de recargo inválido: {interval}")
        covered.extend(range(lower, upper + 1))
    if sorted(covered) != list(range(1, max_installments + 1)):
        raise ValueError("Los recargos deben cubrir cada cantidad de cuotas una sola vez")


class NegotiationProposal(_StrictModel):
    monto_ofrecido: Decimal | None = Field(default=None, gt=0)
    cuotas_pedidas: int | None = Field(default=None, ge=1)
    fecha_pago_propuesta: date | None = None
    motivo_dificultad: str | None = None
    opcion_elegida_id: str | None = None
    medio_pago: MedioPago | None = None


class EscalationReason(_StrictModel):
    code: EscalationMotivo
    reason: str
    policy_refs: tuple[str, ...]


class PolicyDecision(_StrictModel):
    decision: Decision
    motivo: str
    contraoferta: PaymentOption | None = None
    policy_refs: tuple[str, ...]


@lru_cache(maxsize=1)
def load_rules(path: Path = RULES_PATH) -> PolicyRules:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("rules.yaml debe contener un mapping")
    # Strict domain models intentionally accept monetary values through JSON mode,
    # mirroring how the backend fixtures are validated.
    return PolicyRules.model_validate_json(json.dumps(raw))


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("as_of debe tener zona horaria")


def segmentar(debt: Debt, *, rules: PolicyRules | None = None) -> Segment | None:
    if debt.estado == "paid" or debt.saldo_total == 0:
        return None
    resolved = rules or load_rules()
    for name, bounds in resolved.segmentos.items():
        if bounds.dias_min <= debt.dias_mora and (
            bounds.dias_max is None or debt.dias_mora <= bounds.dias_max
        ):
            return name
    return None


def _escalation_applies(
    rule: EscalationRule,
    customer: Customer,
    debt: Debt,
    signals: PolicySignals,
    rules: PolicyRules,
) -> bool:
    match rule:
        case SignalEscalationRule():
            applies: bool = getattr(signals, rule.regla)
            return applies
        case CustomerFlagEscalationRule(regla="identidad_no_verificada"):
            return not customer.identidad_verificada
        case CustomerFlagEscalationRule():
            return customer.marca_vulnerabilidad
        case BrokenAgreementsEscalationRule():
            return customer.acuerdos_previos.incumplidos >= rule.minimo
        case SegmentEscalationRule():
            return segmentar(debt, rules=rules) in rule.segmentos


def requiere_escalamiento(
    customer: Customer,
    debt: Debt,
    signals: PolicySignals | Mapping[str, object] | None = None,
    *,
    rules: PolicyRules | None = None,
) -> EscalationReason | None:
    resolved_rules = rules or load_rules()
    resolved_signals = (
        signals
        if isinstance(signals, PolicySignals)
        else PolicySignals.model_validate(dict(signals or {}), strict=True)
    )
    for rule in resolved_rules.escalamiento_obligatorio:
        if _escalation_applies(rule, customer, debt, resolved_signals, resolved_rules):
            return EscalationReason(
                code=rule.motivo,
                reason=rule.descripcion,
                policy_refs=rule.policy_refs,
            )
    return None


def _surcharge_for(installments: int, rules: PolicyRules) -> Decimal:
    for interval, percentage in rules.recargo_financiacion_pct.items():
        lower, upper = _parse_interval(interval)
        if lower <= installments <= upper:
            return percentage
    raise ValueError(f"No hay recargo configurado para {installments} cuotas")


def _money(value: Decimal) -> Decimal:
    return value.quantize(_PESO, rounding=ROUND_HALF_UP)


def _is_option_allowed(
    option: PaymentOption,
    debt: Debt,
    segment: Segment,
    rules: PolicyRules,
    as_of: datetime,
) -> bool:
    segment_rule = rules.refinanciacion.for_segment(segment)
    limits = rules.limites_globales
    if segment_rule.requiere_operador or segment_rule.cuotas_max is None:
        return False
    if option.valid_until <= as_of or option.cuotas > segment_rule.cuotas_max:
        return False
    days_to_first = (option.primer_vencimiento - as_of.date()).days
    if not limits.primera_cuota_dias_min <= days_to_first <= limits.primera_cuota_dias_max:
        return False

    if option.tipo == "pago_unico":
        max_discount = debt.intereses * segment_rule.quita_interes_max_pct / _HUNDRED
        expected_total = debt.saldo_total - option.quita_interes
        return (
            option.cuotas == 1
            and option.anticipo == 0
            and option.recargo_pct == 0
            and option.quita_interes <= max_discount
            and option.monto_total == expected_total
            and option.monto_cuota == expected_total
        )

    if option.quita_interes != 0:
        return False
    minimum_advance = (
        debt.saldo_total * segment_rule.anticipo_min_pct / _HUNDRED
        if option.cuotas >= rules.refinanciacion.anticipo_requerido_desde_cuotas
        else Decimal(0)
    )
    if option.anticipo < minimum_advance:
        return False
    financed = debt.saldo_total - option.anticipo
    surcharge = _surcharge_for(option.cuotas, rules)
    financed_total = _money(financed * (Decimal(1) + surcharge / _HUNDRED))
    expected_installment = _money(financed_total / option.cuotas)
    # Rounding is "al peso": the last installment absorbs the difference and must also
    # respect the minimum installment.
    last_installment = financed_total - expected_installment * (option.cuotas - 1)
    return (
        option.recargo_pct == surcharge
        and (option.monto_financiado is None or option.monto_financiado == financed)
        and option.monto_cuota == expected_installment
        and option.monto_total == option.anticipo + financed_total
        and option.monto_cuota >= limits.cuota_minima_ars
        and last_installment >= limits.cuota_minima_ars
    )


def opciones_permitidas(
    customer: Customer,
    debt: Debt,
    backend_options: Sequence[PaymentOption],
    *,
    as_of: datetime,
    rules: PolicyRules | None = None,
) -> list[PaymentOption]:
    """Return only backend options that satisfy every policy limit at ``as_of``.

    ``as_of`` is the decision instant (the caller's clock), never the debt snapshot time:
    an old ``debt.as_of`` must not keep an expired offer alive.
    """
    _require_aware(as_of)
    resolved = rules or load_rules()
    if debt.customer_id != customer.customer_id:
        return []
    if debt.estado == "paid" or debt.saldo_total == 0:
        return []
    if customer.acuerdos_activos >= resolved.limites_globales.max_acuerdos_activos:
        return []
    if requiere_escalamiento(customer, debt, rules=resolved) is not None:
        return []
    segment = segmentar(debt, rules=resolved)
    if segment is None:
        return []
    return [
        option
        for option in backend_options
        if _is_option_allowed(option, debt, segment, resolved, as_of)
    ]


def medio_pago_permitido(
    option: PaymentOption, medio_pago: MedioPago, *, rules: PolicyRules | None = None
) -> bool:
    limit = (rules or load_rules()).medios_pago[medio_pago].cuotas_max
    return limit is None or option.cuotas <= limit


def vencimiento_oferta(
    option: PaymentOption, ofrecida_en: datetime, *, rules: PolicyRules | None = None
) -> datetime:
    """An offer expires at the earlier of the backend validity and the policy window."""
    _require_aware(ofrecida_en)
    hours = (rules or load_rules()).limites_globales.vigencia_oferta_horas
    return min(option.valid_until, ofrecida_en + timedelta(hours=hours))


def _decision(
    decision: Decision,
    motivo: str,
    policy_refs: Sequence[str],
    contraoferta: PaymentOption | None = None,
) -> PolicyDecision:
    return PolicyDecision(
        decision=decision,
        motivo=motivo,
        policy_refs=tuple(policy_refs),
        contraoferta=contraoferta,
    )


def _upfront_requirement(option: PaymentOption) -> Decimal:
    return max(option.anticipo, option.monto_cuota)


def evaluar_propuesta(
    customer: Customer,
    debt: Debt,
    propuesta: NegotiationProposal,
    backend_options: Sequence[PaymentOption],
    *,
    as_of: datetime,
    ofrecida_en: datetime | None = None,
    signals: PolicySignals | Mapping[str, object] | None = None,
    rules: PolicyRules | None = None,
) -> PolicyDecision:
    _require_aware(as_of)
    resolved = rules or load_rules()
    escalation = requiere_escalamiento(customer, debt, signals, rules=resolved)
    if escalation is not None:
        return _decision("derivar", escalation.reason, escalation.policy_refs)
    segment = segmentar(debt, rules=resolved)
    if segment is None:
        return _decision("rechazada", "No hay deuda vencida para negociar.", ("POL-NEG-001",))
    allowed = opciones_permitidas(customer, debt, backend_options, as_of=as_of, rules=resolved)

    if propuesta.opcion_elegida_id is not None:
        return _evaluate_selected_option(propuesta, allowed, as_of, ofrecida_en, resolved)

    limits = resolved.limites_globales
    max_installments = resolved.refinanciacion.for_segment(segment).cuotas_max
    if propuesta.cuotas_pedidas is not None and (
        max_installments is None or propuesta.cuotas_pedidas > max_installments
    ):
        return _decision(
            "derivar",
            "La cantidad solicitada es una excepción de política.",
            ("POL-NEG-004", "POL-NEG-009"),
        )
    if propuesta.fecha_pago_propuesta is not None:
        days = (propuesta.fecha_pago_propuesta - as_of.date()).days
        if not limits.primera_cuota_dias_min <= days <= limits.primera_cuota_dias_max:
            return _decision(
                "derivar",
                "La fecha solicitada está fuera de la ventana permitida.",
                ("POL-NEG-007", "POL-NEG-009"),
            )
    if propuesta.monto_ofrecido is not None:
        minimum_partial = debt.saldo_total * limits.monto_minimo_pago_parcial_pct / _HUNDRED
        if propuesta.monto_ofrecido < minimum_partial:
            return _decision(
                "rechazada",
                "El monto es inferior al pago parcial mínimo permitido.",
                ("POL-NEG-006",),
            )
    if propuesta.medio_pago is not None and propuesta.cuotas_pedidas is not None:
        method_limit = resolved.medios_pago[propuesta.medio_pago].cuotas_max
        if method_limit is not None and propuesta.cuotas_pedidas > method_limit:
            return _decision(
                "rechazada",
                "El medio de pago elegido no admite esa cantidad de cuotas.",
                ("PAY-MET-001", "PAY-MET-003"),
            )
    if propuesta.cuotas_pedidas is None and propuesta.monto_ofrecido is None:
        return _decision(
            "rechazada",
            "Faltan el monto o la cantidad de cuotas para armar una opción.",
            ("POL-NEG-001",),
        )

    candidates = [
        option
        for option in allowed
        if propuesta.medio_pago is None
        or medio_pago_permitido(option, propuesta.medio_pago, rules=resolved)
    ]
    if not candidates:
        return _decision(
            "rechazada",
            "No hay opciones habilitadas para este cliente.",
            ("POL-NEG-001",),
        )
    return _counteroffer(propuesta, candidates)


def _evaluate_selected_option(
    propuesta: NegotiationProposal,
    allowed: Sequence[PaymentOption],
    as_of: datetime,
    ofrecida_en: datetime | None,
    rules: PolicyRules,
) -> PolicyDecision:
    chosen = next(
        (option for option in allowed if option.opcion_id == propuesta.opcion_elegida_id),
        None,
    )
    if chosen is None:
        return _decision(
            "rechazada",
            "La opción no está vigente o no fue habilitada para este cliente.",
            ("POL-NEG-001", "POL-NEG-007"),
        )
    if ofrecida_en is not None and as_of >= vencimiento_oferta(chosen, ofrecida_en, rules=rules):
        return _decision(
            "rechazada",
            "La oferta venció; hay que recalcularla.",
            ("POL-NEG-007",),
        )
    if propuesta.medio_pago is not None and not medio_pago_permitido(
        chosen, propuesta.medio_pago, rules=rules
    ):
        return _decision(
            "rechazada",
            "El medio de pago elegido no admite esa cantidad de cuotas.",
            ("PAY-MET-001", "PAY-MET-003"),
        )
    return _decision(
        "aceptable",
        "La opción cumple todos los límites de la política.",
        chosen.policy_refs,
    )


def _counteroffer(
    propuesta: NegotiationProposal, candidates: Sequence[PaymentOption]
) -> PolicyDecision:
    affordable = (
        [
            option
            for option in candidates
            if _upfront_requirement(option) <= propuesta.monto_ofrecido
        ]
        if propuesta.monto_ofrecido is not None
        else list(candidates)
    )
    if not affordable:
        # POL-NEG-006: offer the most accessible plan instead of one the amount cannot cover.
        closest = min(candidates, key=lambda option: (_upfront_requirement(option), option.cuotas))
        return _decision(
            "contraoferta",
            "El monto ofrecido no cubre ninguna opción; se ofrece la más accesible.",
            closest.policy_refs,
            closest,
        )
    requested = propuesta.cuotas_pedidas
    if requested is not None:
        # Nearest installment count; ties go to fewer installments (lower total cost).
        chosen = min(affordable, key=lambda option: (abs(option.cuotas - requested), option.cuotas))
        motivo = (
            "Hay una opción habilitada con la cantidad de cuotas pedida."
            if chosen.cuotas == requested
            else "No hay una opción con esa cantidad de cuotas; se ofrece la más cercana."
        )
    else:
        chosen = min(affordable, key=lambda option: option.cuotas)
        motivo = "Se ofrece la opción de menos cuotas que el monto ofrecido puede cubrir."
    return _decision("contraoferta", motivo, chosen.policy_refs, chosen)
