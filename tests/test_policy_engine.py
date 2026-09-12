from __future__ import annotations

import json
import re
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from app.policy.engine import (
    GlobalLimits,
    NegotiationProposal,
    PolicyRules,
    PolicySignals,
    RefinancingRule,
    SegmentRule,
    _is_option_allowed,
    _surcharge_for,
    evaluar_propuesta,
    load_rules,
    medio_pago_permitido,
    opciones_permitidas,
    requiere_escalamiento,
    segmentar,
    vencimiento_oferta,
)
from app.tools.schemas import (
    Customer,
    Debt,
    EscalationMotivo,
    PaymentOption,
    PaymentOptions,
    RequestHumanArgs,
)

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "mock_api" / "fixtures"
# Reference instant of Anexo F: the decision clock is always explicit in these tests.
AS_OF = datetime(2026, 9, 11, 14, 3, tzinfo=timezone(timedelta(hours=-3)))


def _fixture_json(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


CUSTOMERS = {
    customer.customer_id: customer
    for customer in TypeAdapter(list[Customer]).validate_json(
        (FIXTURES / "customers.json").read_text(encoding="utf-8")
    )
}


def debt_for(customer_id: str) -> Debt:
    payload = {"customer_id": customer_id, **_fixture_json("debts.json")[customer_id]}
    payload.setdefault(
        "acuerdos_previos",
        CUSTOMERS[customer_id].acuerdos_previos.model_dump(mode="json"),
    )
    return Debt.model_validate_json(json.dumps(payload))


def options_for(customer_id: str) -> list[PaymentOption]:
    payload = _fixture_json("options.json")[customer_id]
    options = payload.get("opciones", []) if isinstance(payload, dict) else payload
    return TypeAdapter(list[PaymentOption]).validate_json(json.dumps(options))


def changed[T](model: T, **updates: object) -> T:
    payload = model.model_dump(mode="json")  # type: ignore[attr-defined]
    payload.update(updates)
    return type(model).model_validate_json(json.dumps(payload))  # type: ignore[attr-defined, no-any-return]


def option_by_id(customer_id: str, option_id: str) -> PaymentOption:
    return next(option for option in options_for(customer_id) if option.opcion_id == option_id)


def raw_rules() -> dict[str, Any]:
    return json.loads(load_rules().model_dump_json())  # type: ignore[no-any-return]


def rules_with(mutate: Callable[[dict[str, Any]], None]) -> PolicyRules:
    raw = raw_rules()
    mutate(raw)
    return PolicyRules.model_validate_json(json.dumps(raw))


VERIFIED_377 = changed(CUSTOMERS["CUST-00377"], identidad_verificada=True)
M_125 = (CUSTOMERS["CUST-00125"], debt_for("CUST-00125"))


def allowed_ids(customer: Customer, debt: Debt, options: list[PaymentOption]) -> list[str]:
    return [o.opcion_id for o in opciones_permitidas(customer, debt, options, as_of=AS_OF)]


# --------------------------------------------------------------------------- segmentation


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (0, None),
        (1, "mora_temprana"),
        (60, "mora_temprana"),
        (61, "mora_media"),
        (120, "mora_media"),
        (121, "mora_tardia"),
        (180, "mora_tardia"),
        (181, "prejudicial"),
        (10_000, "prejudicial"),
    ],
)
def test_segment_boundaries(days: int, expected: str | None) -> None:
    debt = changed(debt_for("CUST-00125"), dias_mora=days, estado="mora_temprana")
    assert segmentar(debt) == expected


def test_very_old_debt_still_escalates_as_prejudicial() -> None:
    reason = requiere_escalamiento(M_125[0], changed(M_125[1], dias_mora=10_000))
    assert reason is not None
    assert reason.code == "fuera_de_politica"


def test_paid_debt_has_no_segment_even_with_days() -> None:
    assert segmentar(changed(debt_for("CUST-00450"), dias_mora=63)) is None


# ------------------------------------------------------------------- rules file contract


def test_rules_load_exact_financial_limits() -> None:
    rules = load_rules()
    assert rules.version == "2026.09"
    assert rules.refinanciacion.for_segment("mora_media").cuotas_max == 9
    assert rules.refinanciacion.for_segment("mora_media").quita_interes_max_pct == Decimal("20")
    assert rules.refinanciacion.anticipo_requerido_desde_cuotas == 4
    assert rules.limites_globales.cuota_minima_ars == Decimal("15000")
    assert rules.medios_pago["tarjeta"].cuotas_max == 1


def test_rules_file_must_be_a_mapping(tmp_path: Path) -> None:
    invalid = tmp_path / "rules.yaml"
    invalid.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_rules(invalid)


def test_rule_value_objects_reject_invalid_ranges() -> None:
    with pytest.raises(ValidationError, match="dias_max"):
        SegmentRule(dias_min=10, dias_max=1)
    with pytest.raises(ValidationError, match="cuotas_max"):
        RefinancingRule()
    with pytest.raises(ValidationError):
        GlobalLimits(
            cuota_minima_ars=Decimal("15000"),
            monto_minimo_pago_parcial_pct=Decimal("10"),
            vigencia_oferta_horas=48,
            max_acuerdos_activos=1,
            primera_cuota_dias_min=5,
            primera_cuota_dias_max=15,
            redondeo_cuota="otro",  # type: ignore[arg-type]
            baja_plan_dias_impago=10,
        )


def _drop_segment(raw: dict[str, Any]) -> None:
    del raw["segmentos"]["mora_media"]


def _gap(raw: dict[str, Any]) -> None:
    raw["segmentos"]["mora_media"]["dias_min"] = 62


def _overlap(raw: dict[str, Any]) -> None:
    raw["segmentos"]["mora_media"]["dias_min"] = 60


def _start(raw: dict[str, Any]) -> None:
    raw["segmentos"]["mora_temprana"]["dias_min"] = 2


def _ceiling(raw: dict[str, Any]) -> None:
    raw["segmentos"]["prejudicial"]["dias_max"] = 9999


def _surcharge_gap(raw: dict[str, Any]) -> None:
    del raw["recargo_financiacion_pct"]["7-9"]


def _surcharge_overlap(raw: dict[str, Any]) -> None:
    raw["recargo_financiacion_pct"]["3-4"] = "0"


def _surcharge_inverted(raw: dict[str, Any]) -> None:
    raw["recargo_financiacion_pct"]["9-7"] = raw["recargo_financiacion_pct"].pop("7-9")


def _payment_method(raw: dict[str, Any]) -> None:
    del raw["medios_pago"]["cupon"]


def _duplicate_escalation(raw: dict[str, Any]) -> None:
    raw["escalamiento_obligatorio"].append(raw["escalamiento_obligatorio"][0])


def _signal_without_rule(raw: dict[str, Any]) -> None:
    raw["escalamiento_obligatorio"] = [
        rule for rule in raw["escalamiento_obligatorio"] if rule["regla"] != "falla_tecnica"
    ]


def _unknown_motivo(raw: dict[str, Any]) -> None:
    raw["escalamiento_obligatorio"][0]["motivo"] = "fraude"


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (_drop_segment, "cuatro segmentos"),
        (_gap, "contiguos"),
        (_overlap, "contiguos"),
        (_start, "empezar en 1"),
        (_ceiling, "techo"),
        (_surcharge_gap, "recargos"),
        (_surcharge_overlap, "recargos"),
        (_surcharge_inverted, "Intervalo"),
        (_payment_method, "medios_pago"),
        (_duplicate_escalation, "una sola vez"),
        (_signal_without_rule, "falla_tecnica"),
        (_unknown_motivo, "motivo"),
    ],
)
def test_inconsistent_rules_fail_loudly(
    mutate: Callable[[dict[str, Any]], None], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        rules_with(mutate)


def test_surcharge_lookup_rejects_unconfigured_installments() -> None:
    rules = load_rules()
    assert _surcharge_for(1, rules) == 0
    assert _surcharge_for(12, rules) == Decimal("24")
    with pytest.raises(ValueError, match="No hay recargo"):
        _surcharge_for(13, rules)


# ------------------------------------------ Anexo F.1: arithmetic reproduced with literals


def test_options_match_rules() -> None:
    """Anexo F.1, with the expected figures written as literals (independent oracle)."""
    customer, debt = M_125
    allowed = {
        o.opcion_id: o
        for o in opciones_permitidas(customer, debt, options_for("CUST-00125"), as_of=AS_OF)
    }
    assert list(allowed) == ["OPT-1P", "OPT-3C", "OPT-6C", "OPT-9C"]
    assert (allowed["OPT-1P"].quita_interes, allowed["OPT-1P"].monto_total) == (
        Decimal("6500"),
        Decimal("178000"),
    )
    assert allowed["OPT-3C"].monto_cuota == Decimal("61500")
    assert (allowed["OPT-6C"].anticipo, allowed["OPT-6C"].monto_cuota) == (
        Decimal("18450"),
        Decimal("29889"),
    )
    assert (allowed["OPT-9C"].monto_cuota, allowed["OPT-9C"].monto_total) == (
        Decimal("21402"),
        Decimal("211068"),
    )
    assert allowed_ids(VERIFIED_377, debt_for("CUST-00377"), options_for("CUST-00377")) == [
        "OPT-1P",
        "OPT-2C",
    ]


def _plan_377(cuotas: int, recargo: str, cuota: str, total: str) -> PaymentOption:
    return changed(
        option_by_id("CUST-00377", "OPT-2C"),
        opcion_id=f"OPT-{cuotas}C",
        cuotas=cuotas,
        recargo_pct=recargo,
        monto_cuota=cuota,
        monto_total=total,
    )


# Each case is arithmetically coherent and violates exactly ONE rule, so removing that rule
# from the engine turns the test red (verified by mutation).


@pytest.mark.parametrize(
    ("cuotas", "recargo", "cuota", "total"),
    [
        (3, "0", "12967", "38900"),  # F.1: 38.900 / 3 = 12.966,67
        (4, "8", "10503", "42012"),  # F.1: 38.900 x 1,08 / 4 = 10.503
    ],
)
def test_minimum_installment_rejects_otherwise_valid_plan(
    cuotas: int, recargo: str, cuota: str, total: str
) -> None:
    base = option_by_id("CUST-00377", "OPT-2C")
    plan = _plan_377(cuotas, recargo, cuota, total)
    assert allowed_ids(VERIFIED_377, debt_for("CUST-00377"), [base, plan]) == ["OPT-2C"]


def test_rounded_down_installment_below_minimum_is_rejected_even_if_last_reaches_it() -> None:
    debt = changed(debt_for("CUST-00377"), saldo_total="44998")
    # 44.998 / 3 = 14.999,33 → cuota 14.999 (< mínimo); the last one is 15.000.
    plan = _plan_377(3, "0", "14999", "44998")
    assert allowed_ids(VERIFIED_377, debt, [plan]) == []


def test_last_installment_absorbing_rounding_must_respect_minimum() -> None:
    debt = changed(debt_for("CUST-00377"), saldo_total="44999", capital="41099")
    # 44.999 / 3 = 14.999,67 → cuota 15.000; the last one absorbs the difference: 14.999.
    plan = _plan_377(3, "0", "15000", "44999")
    assert allowed_ids(VERIFIED_377, debt, [plan]) == []
    rounded_up = changed(debt, saldo_total="45001")
    assert allowed_ids(VERIFIED_377, rounded_up, [changed(plan, monto_total="45001")]) == ["OPT-3C"]


@pytest.mark.parametrize(
    ("anticipo", "financiado", "cuota", "total", "expected"),
    [
        ("18450", "166050", "29889", "197784", ["OPT-6C"]),  # exactly 10 %: allowed
        ("18449", "166051", "29889", "197784", []),  # one peso short
        ("0", "184500", "33210", "199260", []),  # no advance at all
    ],
)
def test_minimum_advance_is_enforced_from_four_installments(
    anticipo: str, financiado: str, cuota: str, total: str, expected: list[str]
) -> None:
    plan = changed(
        option_by_id("CUST-00125", "OPT-6C"),
        anticipo=anticipo,
        monto_financiado=financiado,
        monto_cuota=cuota,
        monto_total=total,
    )
    assert allowed_ids(*M_125, [plan]) == expected


def test_three_installments_do_not_require_advance() -> None:
    assert allowed_ids(*M_125, [option_by_id("CUST-00125", "OPT-3C")]) == ["OPT-3C"]


@pytest.mark.parametrize(
    ("anticipo", "financiado", "cuota", "total", "expected"),
    [
        ("0", "184500", "49815", "199260", []),  # 184.500 x 1,08 / 4: advance required at 4
        ("18450", "166050", "44834", "197784", ["OPT-4C"]),  # 179.334 / 4 = 44.833,5 → 44.834
    ],
)
def test_advance_threshold_starts_exactly_at_four_installments(
    anticipo: str, financiado: str, cuota: str, total: str, expected: list[str]
) -> None:
    plan = changed(
        option_by_id("CUST-00125", "OPT-6C"),
        opcion_id="OPT-4C",
        cuotas=4,
        anticipo=anticipo,
        monto_financiado=financiado,
        monto_cuota=cuota,
        monto_total=total,
    )
    assert allowed_ids(*M_125, [plan]) == expected


def test_financed_amount_must_match_balance_minus_advance() -> None:
    plan = changed(option_by_id("CUST-00125", "OPT-6C"), monto_financiado="170000")
    assert allowed_ids(*M_125, [plan]) == []


@pytest.mark.parametrize(
    ("option_id", "updates"),
    [
        ("OPT-6C", {"valid_until": "2026-09-11T14:03:00-03:00"}),  # expired at as_of
        ("OPT-6C", {"cuotas": 10}),  # above segment maximum
        ("OPT-6C", {"primer_vencimiento": "2026-10-10"}),  # outside 5-15 days
        ("OPT-6C", {"primer_vencimiento": "2026-09-15"}),  # 4 days: before window
        ("OPT-6C", {"recargo_pct": "12"}),  # wrong surcharge
        ("OPT-6C", {"quita_interes": "1"}),  # discount on an installment plan
        ("OPT-1P", {"quita_interes": "7000", "monto_total": "177500", "monto_cuota": "177500"}),
        ("OPT-1P", {"monto_total": "184500", "monto_cuota": "178000"}),  # inconsistent
        ("OPT-1P", {"cuotas": 2}),  # single payment must be one installment
    ],
)
def test_invalid_backend_options_are_filtered(option_id: str, updates: dict[str, object]) -> None:
    assert allowed_ids(*M_125, [changed(option_by_id("CUST-00125", option_id), **updates)]) == []


def test_operator_segment_never_allows_options() -> None:
    debt = debt_for("CUST-00212")
    option = option_by_id("CUST-00125", "OPT-3C")
    assert not _is_option_allowed(option, debt, "prejudicial", load_rules(), AS_OF)


# ------------------------------------------------------------------------ decision clock


def test_as_of_is_required_and_timezone_aware() -> None:
    customer, debt = M_125
    with pytest.raises(TypeError):
        opciones_permitidas(customer, debt, options_for("CUST-00125"))  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="zona horaria"):
        opciones_permitidas(customer, debt, [], as_of=datetime(2026, 9, 11))
    with pytest.raises(ValueError, match="zona horaria"):
        evaluar_propuesta(customer, debt, NegotiationProposal(), [], as_of=datetime(2026, 9, 11))


def test_old_debt_snapshot_does_not_keep_expired_offers_alive() -> None:
    customer, debt = M_125
    later = datetime(2026, 9, 20, tzinfo=AS_OF.tzinfo)
    assert debt.as_of < later
    assert opciones_permitidas(customer, debt, options_for("CUST-00125"), as_of=later) == []


def test_offer_expiry_is_the_earlier_of_backend_and_policy_window() -> None:
    option = option_by_id("CUST-00125", "OPT-3C")
    assert vencimiento_oferta(option, AS_OF) == AS_OF + timedelta(hours=48)
    late = option.valid_until - timedelta(hours=1)
    assert vencimiento_oferta(option, late) == option.valid_until
    with pytest.raises(ValueError, match="zona horaria"):
        vencimiento_oferta(option, datetime(2026, 9, 11))


# ------------------------------------------------------------------------- escalation


def test_customer_212_escalates_out_of_policy_as_eval_case_n04_expects() -> None:
    reason = requiere_escalamiento(CUSTOMERS["CUST-00212"], debt_for("CUST-00212"))
    assert reason is not None
    assert reason.code == "fuera_de_politica"
    assert reason.policy_refs == ("POL-NEG-008", "ESC-001")


def test_prejudicial_segment_forces_escalation_independently() -> None:
    customer = changed(CUSTOMERS["CUST-00212"], acuerdos_previos={"total": 0, "incumplidos": 0})
    reason = requiere_escalamiento(customer, debt_for("CUST-00212"), PolicySignals())
    assert reason is not None
    assert (reason.code, reason.policy_refs) == ("fuera_de_politica", ("POL-NEG-002", "ESC-001"))


def test_judicial_flag_alone_is_not_an_escalation_rule() -> None:
    customer = changed(CUSTOMERS["CUST-00125"], en_gestion_judicial=True)
    assert requiere_escalamiento(customer, debt_for("CUST-00125")) is None


@pytest.mark.parametrize(
    ("customer_updates", "code"),
    [
        ({"identidad_verificada": False}, "identidad_no_verificada"),
        ({"marca_vulnerabilidad": True}, "vulnerabilidad"),
        ({"acuerdos_previos": {"total": 2, "incumplidos": 2}}, "fuera_de_politica"),
    ],
)
def test_customer_data_forces_escalation(customer_updates: dict[str, object], code: str) -> None:
    customer = changed(CUSTOMERS["CUST-00125"], **customer_updates)
    reason = requiere_escalamiento(customer, debt_for("CUST-00125"))
    assert reason is not None
    assert reason.code == code


def test_one_broken_agreement_does_not_escalate() -> None:
    customer = changed(CUSTOMERS["CUST-00125"], acuerdos_previos={"total": 2, "incumplidos": 1})
    assert requiere_escalamiento(customer, debt_for("CUST-00125")) is None


@pytest.mark.parametrize(
    ("signal", "code"),
    [
        ("pedido_explicito_de_humano", "pedido_explicito"),
        ("presenta_reclamo", "reclamo"),
        ("cliente_declara_vulnerabilidad", "vulnerabilidad"),
        ("menciona_abogado_o_demanda", "amenaza_legal"),
        ("indicios_fraude", "identidad_no_verificada"),
        ("outcome_escritura_desconocido", "outcome_de_escritura_desconocido"),
        ("falla_tecnica", "falla_tecnica"),
        ("tres_intentos_sin_avance", "loop_sin_avance"),
    ],
)
def test_each_runtime_signal_forces_the_expected_escalation(signal: str, code: str) -> None:
    reason = requiere_escalamiento(*M_125, {signal: True})
    assert reason is not None
    assert reason.code == code


def test_valid_customer_does_not_require_escalation() -> None:
    assert requiere_escalamiento(*M_125) is None


def test_escalation_comes_from_rules_file_not_from_code() -> None:
    unverified = changed(CUSTOMERS["CUST-00125"], identidad_verificada=False)
    without_identity = rules_with(
        lambda raw: raw.update(
            escalamiento_obligatorio=[
                rule
                for rule in raw["escalamiento_obligatorio"]
                if rule["regla"] != "identidad_no_verificada"
            ]
        )
    )
    assert requiere_escalamiento(unverified, debt_for("CUST-00125")) is not None
    assert requiere_escalamiento(unverified, debt_for("CUST-00125"), rules=without_identity) is None


def test_escalation_codes_are_valid_request_human_motives() -> None:
    schema_motives = set(get_args(RequestHumanArgs.model_fields["motivo"].annotation))
    assert schema_motives == set(get_args(EscalationMotivo))
    assert {rule.motivo for rule in load_rules().escalamiento_obligatorio} <= schema_motives


# ----------------------------------------------------------------------- availability


def test_unverified_customer_backend_options_are_not_offerable() -> None:
    options = options_for("CUST-00377")
    assert allowed_ids(CUSTOMERS["CUST-00377"], debt_for("CUST-00377"), options) == []


def test_mismatched_customer_or_existing_agreement_has_no_options() -> None:
    customer, debt = M_125
    options = options_for("CUST-00125")
    assert allowed_ids(customer, changed(debt, customer_id="CUST-00450"), options) == []
    assert allowed_ids(changed(customer, acuerdos_activos=1), debt, options) == []


def test_paid_or_unsegmentable_debt_has_no_options() -> None:
    assert allowed_ids(CUSTOMERS["CUST-00450"], debt_for("CUST-00450"), []) == []
    assert allowed_ids(M_125[0], changed(M_125[1], dias_mora=0), options_for("CUST-00125")) == []


def test_payment_method_limits() -> None:
    assert medio_pago_permitido(option_by_id("CUST-00125", "OPT-1P"), "tarjeta")
    assert not medio_pago_permitido(option_by_id("CUST-00125", "OPT-3C"), "tarjeta")
    assert medio_pago_permitido(option_by_id("CUST-00125", "OPT-9C"), "debito_automatico")


# ---------------------------------------------------------------------- evaluar_propuesta


def evaluate(proposal: NegotiationProposal, **kwargs: Any) -> Any:
    return evaluar_propuesta(*M_125, proposal, options_for("CUST-00125"), as_of=AS_OF, **kwargs)


def test_selected_valid_option_is_acceptable() -> None:
    decision = evaluate(NegotiationProposal(opcion_elegida_id="OPT-3C"))
    assert decision.decision == "aceptable"
    assert decision.contraoferta is None


def test_unknown_selected_option_is_rejected() -> None:
    assert evaluate(NegotiationProposal(opcion_elegida_id="OPT-ZZ")).decision == "rechazada"


def test_selected_option_expires_after_policy_window() -> None:
    offered = AS_OF - timedelta(hours=48)
    expired = evaluate(NegotiationProposal(opcion_elegida_id="OPT-3C"), ofrecida_en=offered)
    fresh = evaluate(
        NegotiationProposal(opcion_elegida_id="OPT-3C"),
        ofrecida_en=AS_OF - timedelta(hours=47),
    )
    assert (expired.decision, expired.policy_refs) == ("rechazada", ("POL-NEG-007",))
    assert fresh.decision == "aceptable"


def test_card_only_accepts_single_payment() -> None:
    card_plan = evaluate(NegotiationProposal(opcion_elegida_id="OPT-3C", medio_pago="tarjeta"))
    card_single = evaluate(NegotiationProposal(opcion_elegida_id="OPT-1P", medio_pago="tarjeta"))
    requested = evaluate(NegotiationProposal(cuotas_pedidas=6, medio_pago="tarjeta"))
    assert (card_plan.decision, card_plan.policy_refs) == (
        "rechazada",
        ("PAY-MET-001", "PAY-MET-003"),
    )
    assert card_single.decision == "aceptable"
    assert requested.decision == "rechazada"


@pytest.mark.parametrize(
    ("proposal", "expected_option"),
    [
        (NegotiationProposal(cuotas_pedidas=6), "OPT-6C"),  # exact
        (NegotiationProposal(cuotas_pedidas=4), "OPT-3C"),  # nearest; tie → fewer
        (NegotiationProposal(cuotas_pedidas=5), "OPT-6C"),  # nearest
        (NegotiationProposal(cuotas_pedidas=2), "OPT-1P"),  # tie 1P/3C → fewer
        (NegotiationProposal(monto_ofrecido=Decimal("200000")), "OPT-1P"),
        (NegotiationProposal(monto_ofrecido=Decimal("30000")), "OPT-6C"),  # affordable, fewest
        (NegotiationProposal(monto_ofrecido=Decimal("30000"), cuotas_pedidas=3), "OPT-6C"),
        (NegotiationProposal(monto_ofrecido=Decimal("200000"), medio_pago="tarjeta"), "OPT-1P"),
    ],
)
def test_counteroffer_uses_amount_installments_and_method(
    proposal: NegotiationProposal, expected_option: str
) -> None:
    decision = evaluate(proposal)
    assert decision.decision == "contraoferta"
    assert decision.contraoferta is not None
    assert decision.contraoferta.opcion_id == expected_option


def test_nearest_counteroffer_explains_it_is_not_the_requested_count() -> None:
    assert "más cercana" in evaluate(NegotiationProposal(cuotas_pedidas=4)).motivo
    assert "cuotas pedida" in evaluate(NegotiationProposal(cuotas_pedidas=6)).motivo


def test_amount_that_covers_nothing_gets_the_most_accessible_plan() -> None:
    decision = evaluate(NegotiationProposal(monto_ofrecido=Decimal("20000")))
    assert decision.decision == "contraoferta"
    assert decision.contraoferta is not None
    assert decision.contraoferta.opcion_id == "OPT-9C"  # 21.402 is the lowest requirement
    assert "no cubre" in decision.motivo


def test_minimum_installment_blocked_request_gets_longest_valid_plan() -> None:
    options = [*options_for("CUST-00377"), _plan_377(3, "0", "12967", "38900")]
    decision = evaluar_propuesta(
        VERIFIED_377,
        debt_for("CUST-00377"),
        NegotiationProposal(cuotas_pedidas=3),
        options,
        as_of=AS_OF,
    )
    assert decision.contraoferta is not None
    assert decision.contraoferta.opcion_id == "OPT-2C"


def test_proposal_without_amount_or_installments_asks_for_data() -> None:
    decision = evaluate(NegotiationProposal(motivo_dificultad="me quedé sin trabajo"))
    assert decision.decision == "rechazada"
    assert "Faltan" in decision.motivo


def test_excess_installments_or_out_of_window_date_are_escalated() -> None:
    late = AS_OF.date() + timedelta(days=16)
    assert evaluate(NegotiationProposal(cuotas_pedidas=10)).decision == "derivar"
    assert evaluate(NegotiationProposal(fecha_pago_propuesta=late)).decision == "derivar"
    in_window = NegotiationProposal(
        fecha_pago_propuesta=AS_OF.date() + timedelta(days=9), cuotas_pedidas=3
    )
    assert evaluate(in_window).decision == "contraoferta"


def test_partial_payment_below_minimum_is_rejected() -> None:
    decision = evaluate(NegotiationProposal(monto_ofrecido=Decimal("18449")))
    assert (decision.decision, decision.policy_refs) == ("rechazada", ("POL-NEG-006",))


def test_customer_with_active_agreement_gets_no_counteroffer() -> None:
    decision = evaluar_propuesta(
        changed(M_125[0], acuerdos_activos=1),
        M_125[1],
        NegotiationProposal(cuotas_pedidas=3),
        options_for("CUST-00125"),
        as_of=AS_OF,
    )
    assert decision.decision == "rechazada"
    assert decision.contraoferta is None


def test_escalation_wins_before_negotiation() -> None:
    decision = evaluate(
        NegotiationProposal(opcion_elegida_id="OPT-3C"),
        signals=PolicySignals(cliente_declara_vulnerabilidad=True),
    )
    assert (decision.decision, decision.policy_refs) == ("derivar", ("ESC-002",))


def test_paid_debt_proposal_is_rejected_without_escalation() -> None:
    decision = evaluar_propuesta(
        CUSTOMERS["CUST-00450"],
        debt_for("CUST-00450"),
        NegotiationProposal(cuotas_pedidas=1),
        [],
        as_of=AS_OF,
    )
    assert (decision.decision, decision.policy_refs) == ("rechazada", ("POL-NEG-001",))


def test_options_response_fixture_remains_parseable() -> None:
    response = PaymentOptions(
        customer_id="CUST-00125",
        opciones=options_for("CUST-00125"),
        as_of=debt_for("CUST-00125").as_of,
    )
    assert len(response.opciones) == 4


# ------------------------------------------------------------- knowledge base ↔ rules.yaml

_NUMBER_WORDS = {1: "un", 2: "dos", 3: "tres"}
_SEGMENT_LABELS = {
    "mora_temprana": "Mora temprana",
    "mora_media": "Mora media",
    "mora_tardia": "Mora tardía",
    "prejudicial": "Prejudicial",
}


def _kb_text() -> str:
    raw = "\n".join(path.read_text(encoding="utf-8") for path in sorted((ROOT / "kb").glob("*.md")))
    return re.sub(r"\s+", " ", raw)


def _money_ars(value: Decimal) -> str:
    return f"${value:,.0f}".replace(",", ".")


def kb_inconsistencies(rules: PolicyRules, kb: str) -> list[str]:
    """Every number the KB states must be derivable from rules.yaml."""
    limits = rules.limites_globales
    refinancing = rules.refinanciacion
    expected: list[str] = []
    for segment, bounds in rules.segmentos.items():
        upper = "o más" if bounds.dias_max is None else f"a {bounds.dias_max}"
        expected.append(f"| {_SEGMENT_LABELS[segment]} | {bounds.dias_min} {upper} |")
        rule = refinancing.for_segment(segment)
        label = _SEGMENT_LABELS[segment]
        if rule.requiere_operador:
            expected.append(f"| {label} | requiere operador |")
            continue
        expected += [
            f"| {label} | {rule.quita_interes_max_pct} % |",
            f"{label.lower()} **{rule.cuotas_max}**",
            f"{label.lower()} **{rule.anticipo_min_pct} %**",
        ]
    for interval, percentage in rules.recargo_financiacion_pct.items():
        lower, _, upper = interval.partition("-")
        label = f"{lower} (pago único)" if not upper else f"{lower} a {upper}"
        expected.append(f"| {label} | {percentage} % |")
    from_installments = refinancing.anticipo_requerido_desde_cuotas
    broken = next(
        rule.minimo
        for rule in rules.escalamiento_obligatorio
        if rule.regla == "acuerdos_previos_incumplidos"
    )
    card_limit = rules.medios_pago["tarjeta"].cuotas_max
    assert card_limit is not None
    expected += [
        f"a partir de **{from_installments} cuotas**",
        f"hasta {from_installments - 1} cuotas no requieren anticipo",
        f"**{_money_ars(limits.cuota_minima_ars)}**",
        f"**{limits.monto_minimo_pago_parcial_pct} % del saldo total**",
        f"desde el {limits.monto_minimo_pago_parcial_pct} % del saldo total",  # FAQ-001
        f"vale **{limits.vigencia_oferta_horas} horas**",
        f"{limits.vigencia_oferta_horas} horas desde que se comunicó",  # FAQ-012
        f"entre **{limits.primera_cuota_dias_min} y "
        f"{limits.primera_cuota_dias_max} días corridos**",
        f"impaga **{limits.baja_plan_dias_impago} días corridos**",
        f"impaga {limits.baja_plan_dias_impago} días corridos",  # FAQ-004
        f"más de **{_NUMBER_WORDS[limits.max_acuerdos_activos]} acuerdo activo**",
        f"admite {_NUMBER_WORDS[limits.max_acuerdos_activos]} acuerdo activo",  # FAQ-006
        f"**{_NUMBER_WORDS[broken]} o más planes incumplidos**",
        f"registra {_NUMBER_WORDS[broken]} o más planes incumplidos",  # ESC-001
        f"en **{_NUMBER_WORDS[card_limit]} pago**",  # PAY-MET-001
        f"la tarjeta de crédito se usa en {_NUMBER_WORDS[card_limit]} pago",  # FAQ-008
    ]
    return [phrase for phrase in expected if phrase not in kb]


def test_kb_matches_rules() -> None:
    assert kb_inconsistencies(load_rules(), _kb_text()) == []


def _set_limit(key: str, value: object) -> Callable[[dict[str, Any]], None]:
    return lambda raw: raw["limites_globales"].__setitem__(key, value)


def _set_segment(segment: str, key: str, value: object) -> Callable[[dict[str, Any]], None]:
    return lambda raw: raw["refinanciacion"][segment].__setitem__(key, value)


def _set_media_max(raw: dict[str, Any]) -> None:
    raw["segmentos"]["mora_media"]["dias_max"] = 121
    raw["segmentos"]["mora_tardia"]["dias_min"] = 122


def _set_broken(raw: dict[str, Any]) -> None:
    for rule in raw["escalamiento_obligatorio"]:
        if rule["regla"] == "acuerdos_previos_incumplidos":
            rule["minimo"] = 3


@pytest.mark.parametrize(
    "mutate",
    [
        _set_media_max,
        _set_limit("cuota_minima_ars", "16000"),
        _set_limit("vigencia_oferta_horas", 72),
        _set_limit("baja_plan_dias_impago", 15),
        _set_limit("monto_minimo_pago_parcial_pct", "15"),
        _set_segment("mora_temprana", "quita_interes_max_pct", "5"),
        _set_segment("mora_tardia", "anticipo_min_pct", "20"),
        _set_segment("mora_media", "cuotas_max", 8),
        lambda raw: raw["recargo_financiacion_pct"].__setitem__("4-6", "10"),
        _set_broken,
    ],
)
def test_kb_consistency_check_detects_rule_drift(mutate: Callable[[dict[str, Any]], None]) -> None:
    assert kb_inconsistencies(rules_with(mutate), _kb_text()) != []
