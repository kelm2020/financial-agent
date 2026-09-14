"""Code assertions over the customer-facing behavior that ADR-010 fixed.

Templates are verified here with deterministic assertions, not with the LLM judge: a fixed text
either has a next step, acknowledges a vulnerability and avoids internal jargon, or it does not.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from app.graph.confirmation import (
    deterministic_confirmation,
    parse_confirmation,
    recheck_reason,
)
from app.graph.nodes.respond import (
    _ESCALATION_TEXTS,
    _STATIC_TEMPLATES,
    _template_text,
    policy_extract,
)
from app.graph.routing import (
    asks_debt_composition,
    asks_due_dates,
    bare_installments,
    declares_crisis,
    escalation_evidence_holds,
    listed_choice,
    monthly_amount,
    proposal_reply,
    reopens_offer,
    route_turn,
)
from app.graph.state import ResponsePlan, RouteResult
from app.guards.injection import GuardModelResult
from app.llm.protocol import ScriptedLLM
from app.tools.client import CollectionsGateway
from mock_api.idempotency_store import idempotency_store
from tests.agent_support import (
    REFERENCE_NOW,
    AgentRuntime,
    BackendFault,
    StaticRetriever,
    agent_runtime,
    corpus_chunk,
    fixture_draft,
)


async def _say(runtime: AgentRuntime, text: str) -> Any:
    conversation = await runtime.service.create_conversation("CUST-00125")
    return await runtime.service.send_message(
        conversation.conversation_id, conversation.customer_id, text, context=runtime.context
    )


# ---------------------------------------------------------------------------------- routing


@pytest.mark.parametrize(
    ("text", "intent", "motivo"),
    [
        ("Perdí el laburo, ¿puedo pagar en 3 cuotas?", "pedido_humano", "vulnerabilidad"),
        ("Se murió mi papá y estoy sin ingresos", "pedido_humano", "vulnerabilidad"),
        ("Tengo que hacer quimio este mes", "pedido_humano", "vulnerabilidad"),
        (
            "Me rescindieron el contrato y las changas no alcanzan",
            "pedido_humano",
            "vulnerabilidad",
        ),
        ("No quiero seguir viviendo así", "pedido_humano", "vulnerabilidad"),
        ("Esa deuda no es mía", "pedido_humano", "reclamo"),
        ("Voy a iniciar acciones legales", "pedido_humano", "amenaza_legal"),
        ("No quiero hablar con un bot", "pedido_humano", "pedido_explicito"),
        ("¿Puede pagar otra persona por mí?", "consulta_general", None),
        ("¿Cuándo tenía que pagar?", "consulta_deuda", None),
        ("¿De qué se compone mi deuda?", "consulta_deuda", None),
        ("¿Qué interés tiene el plan de 6 cuotas?", "consulta_general", None),
        ("¿Me conviene comprar dólares?", "fuera_de_dominio", None),
        ("¿Puedo pagar en dólares?", "consulta_general", None),
        ("¿Cuánto pagaría en total por la opción de 6 cuotas?", "negociacion", None),
        ("La opción de 3 cuotas me sirve, ¿podemos dejarlo así?", "aceptar_opcion", None),
        ("La opción de 3 cuotas me interesa. ¿Podemos hacer ese acuerdo?", "aceptar_opcion", None),
        ("Creo que voy a aceptar las 3 cuotas, ¿podés registrarlo?", "aceptar_opcion", None),
        ("No voy a aceptar las 3 cuotas", "negociacion", None),
        ("Ni idea de cuánto puedo poner", "ambiguo", None),
    ],
)
def test_routing_lexicons_generalize_by_category(
    text: str, intent: str, motivo: str | None
) -> None:
    route = route_turn(text)
    assert (route.intent, route.escalation_motivo) == (intent, motivo)


def test_signal_helpers() -> None:
    assert declares_crisis("Pienso en quitarme la vida")
    assert not declares_crisis("Estoy cansado de esperar")
    assert asks_due_dates("¿Qué fechas de pago tengo atrasadas?")
    assert asks_debt_composition("¿Por qué me cobran intereses?")
    assert not asks_debt_composition("¿La quita es sobre los intereses?")


# ----------------------------------------------------------------------------- confirmation


@pytest.mark.parametrize(
    ("text", "verdict", "reason"),
    [
        ("no sé", "other", "doubt"),
        ("sí, aunque no sé si llego", "other", "doubt"),
        ("esperá que lo pienso", "other", "doubt"),
        ("Ahora no puedo decidir; te confirmo mañana", "other", "doubt"),
        ("creo que sí", "other", "doubt"),
        ("no hay problema, dale", "other", "idiom"),
        ("no hay problema, pero no", "no", None),
        ("no sé, mejor no", "no", None),
        ("no, dale", "no", None),
        ("dale, pero no", "no", None),
        ("esperá", "no", None),
        ("pará", "no", None),
        ("sí sí", "yes", None),
        ("mmm bueno, eh", None, None),
    ],
)
def test_doubt_and_idioms_keep_the_draft_while_negation_still_wins(
    text: str, verdict: str | None, reason: str | None
) -> None:
    result = deterministic_confirmation(text)
    assert (result.verdict if result is not None else None) == verdict
    assert recheck_reason(text) == reason


async def test_confirmation_window_is_minutes_bounded_by_the_offer() -> None:
    async with agent_runtime() as runtime:
        result = await _say(runtime, "Quiero la opción de 3 cuotas")
    assert result.state["pending_draft"].expires_at == REFERENCE_NOW + timedelta(minutes=10)


# ----------------------------------------------------------------------------- escalation


def test_escalation_messages_follow_esc_policies() -> None:
    for motivo, text in _ESCALATION_TEXTS.items():
        assert "derivé" in text, motivo
        assert "Prefiero" not in text and "le pasé el detalle" not in text, motivo
    vulnerable = _ESCALATION_TEXTS["vulnerabilidad"]
    assert vulnerable.startswith("Gracias por contármelo")  # ESC-002: acknowledge first
    assert "No hace falta que me des más detalles" in vulnerable
    assert "con prioridad" in vulnerable
    assert "?" not in vulnerable  # asks nothing of a person who just disclosed


async def test_vulnerability_summary_is_a_priority_flag_not_a_transcript(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[dict[str, Any]] = []
    original = CollectionsGateway.transfer_to_human

    async def spy(self: CollectionsGateway, scope: Any, **kwargs: Any) -> Any:
        captured.append(kwargs)
        return await original(self, scope, **kwargs)

    monkeypatch.setattr(CollectionsGateway, "transfer_to_human", spy)
    async with agent_runtime() as runtime:
        result = await _say(runtime, "Falleció mi mamá y no llego con las cuotas")
    assert "Gracias por contármelo" in result.text
    assert [item["motivo"] for item in captured] == ["vulnerabilidad"]
    assert "mamá" not in captured[0]["resumen"] and "Prioridad alta" in captured[0]["resumen"]


async def test_crisis_message_survives_a_failed_transfer() -> None:
    faults = (BackendFault("POST", "/transfer", "500"),)
    async with agent_runtime(faults=faults) as runtime:
        result = await _say(runtime, "Pienso en quitarme la vida")
    assert "pedí ayuda ahora" in result.text
    assert "No pude completar la derivación" in result.text


def test_escalation_template_edges() -> None:
    generic = ResponsePlan(
        kind="escalate", template_id="human", facts={"motivo": "loop_sin_avance"}
    )
    assert _template_text(generic, {}) == _STATIC_TEMPLATES["human"]
    exception = ResponsePlan(
        kind="escalate",
        template_id="human",
        facts={"motivo": "fuera_de_politica", "source": "exception"},
    )
    without_options = _template_text(exception, {})
    assert "ya te derivé" in without_options and "cuota más baja" not in without_options


# -------------------------------------------------------------------------------- debt text


async def test_single_due_date_is_singular_and_offers_a_next_step() -> None:
    async with agent_runtime() as runtime:
        current = await runtime.context.gateway.get_debt(runtime.context.scope)
    assert current.data is not None
    debt = current.data.model_copy(update={"vencimientos": current.data.vencimientos[:1]})
    text = _template_text(ResponsePlan(kind="direct", template_id="debt_due_dates"), {"debt": debt})
    assert "tenés 1 vencimiento impago: 10/07/2026 ($61.500)." in text
    assert text.endswith("¿Querés que veamos alternativas para regularizarlo?")


async def test_debt_composition_comes_from_the_system_even_without_evidence() -> None:
    async with agent_runtime() as runtime:
        result = await _say(runtime, "¿Por qué me cobran intereses?")
        assert [call.name for call in runtime.recorder.tool_calls] == ["get_customer", "get_debt"]
    assert "$152.000 de capital vencido y $32.500 de intereses" in result.text
    assert "[FAQ" not in result.text

    class BrokenRetriever(StaticRetriever):
        async def search(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("index down")

    async with agent_runtime(retriever=BrokenRetriever([])) as runtime:
        degraded = await _say(runtime, "Quiero entender los intereses de la deuda")
        assert any(event["type"] == "retriever_unavailable" for event in runtime.recorder.events)
    assert "$32.500 de intereses" in degraded.text

    async with agent_runtime(retriever=StaticRetriever([corpus_chunk("FAQ-013")])) as runtime:
        cited = await _say(runtime, "¿De qué se compone mi deuda?")
    assert "[FAQ-013]" in cited.text


def test_debt_and_option_template_edges() -> None:
    composition = ResponsePlan(kind="direct", template_id="debt_composition")
    assert "problema para acceder" in _template_text(composition, {})
    missing_option = ResponsePlan(
        kind="negotiation", template_id="options_requested", facts={"option_id": "OPT-NONE"}
    )
    assert _template_text(missing_option, {}).startswith("No hay opciones automáticas")


# ------------------------------------------------------------------------ template assertions

_NEXT_STEP_MARKERS = (
    "?",
    "deriv",
    "asesor",
    "podemos revisar",
    "probá de nuevo",
    "canales oficiales",
    "sí o no",
)
# §10.1: an injection deflection does not re-engage, and an explicit "no, gracias" is a
# terminal closing rather than another invitation to continue.
_EXEMPT_FROM_NEXT_STEP = frozenset({"deflect", "no_thanks"})


def test_every_static_template_leaves_a_next_step() -> None:
    for key, text in _STATIC_TEMPLATES.items():
        if key in _EXEMPT_FROM_NEXT_STEP:
            continue
        assert any(marker in text.casefold() for marker in _NEXT_STEP_MARKERS), key


def test_policy_extract_drops_agent_instructions_and_stays_short() -> None:
    plan = ResponsePlan(
        kind="policy", template_id="policy_extract", cited_section_ids=("POL-NEG-003",)
    )
    extract = policy_extract(
        plan,
        {
            "retrieved": [corpus_chunk("POL-NEG-003")],
            "last_user_text": "¿Hay quita de intereses si pago todo?",
        },
    )
    assert extract is not None
    assert "agente" not in extract.text.casefold()
    assert 1 <= len(extract.claims) <= 3
    assert "únicamente sobre los intereses devengados" in extract.text


async def test_confirmation_classifier_failure_keeps_the_draft() -> None:
    # Outside every lexicon, so the model is asked; its failure is "other", never "no" or "yes".
    verdict = await parse_confirmation("eso lo vemos después", ScriptedLLM([RuntimeError("down")]))
    assert verdict.verdict == "other"


@pytest.mark.parametrize(
    ("text", "verdict"),
    [
        ("te lo confirmo mañana, ahora no puedo decidir nada", "other"),
        ("no gracias, paso", "no"),  # opens with the negation
        ("dale, pero no", "no"),  # short reply
        ("esperá", "no"),
    ],
)
def test_weak_negation_rejects_only_when_short_or_leading(text: str, verdict: str | None) -> None:
    result = deterministic_confirmation(text)
    assert (result.verdict if result is not None else None) == verdict


async def test_classifier_signal_adds_a_derivation_the_router_missed() -> None:
    signal = GuardModelResult(
        escalation_signal="vulnerabilidad", escalation_evidence="redujeron a cero las horas"
    )
    async with agent_runtime(guard_classifier=ScriptedLLM([signal])) as runtime:
        result = await _say(
            runtime, "En el trabajo me redujeron a cero las horas y ahora no puedo pagar"
        )
        motivos = [
            c.arguments["motivo"] for c in runtime.recorder.tool_calls if c.name == "request_human"
        ]
        assert motivos == ["vulnerabilidad"]
        assert any(
            e["type"] == "escalation_signal_from_classifier" for e in runtime.recorder.events
        )
    assert "Gracias por contármelo" in result.text


@pytest.mark.parametrize(
    ("text", "evidence", "signal", "holds"),
    [
        ("Me quedé sin trabajo y no llego a pagar", "Me quedé sin trabajo", "vulnerabilidad", True),
        (
            "No llego con el total, ¿qué opciones hay?",
            "No llego con el total",
            "vulnerabilidad",
            False,
        ),
        ("Este mes estoy complicado", "este mes estoy complicado", "vulnerabilidad", False),
        ("No sé cuánto puedo pagar", "No sé cuánto puedo pagar", "vulnerabilidad", False),
        ("No puedo pagar todo este mes", "me quedé sin trabajo", "vulnerabilidad", False),
        ("No puedo pagar todo", "", "vulnerabilidad", False),
        ("Ese importe no lo reconozco, cobran de más", "cobran de más", "reclamo", True),
        (
            "¿Me lo puede pagar un familiar?",
            "un familiar",
            "pedido_explicito",
            False,
        ),
        # Third-party payer (FAQ-010): names a person without asking to talk to one.
        ("¿Puede pagar otra persona por mí?", "otra persona", "pedido_explicito", False),
        # A role or a rejected automated channel is evidence even outside the router table.
        (
            "Pasame con un supervisor, por favor",
            "Pasame con un supervisor",
            "pedido_explicito",
            True,
        ),
        (
            "Basta. No quiero seguir hablando con la grabación",
            "No quiero seguir hablando con la grabación",
            "pedido_explicito",
            True,
        ),
        # Clauses joined by the model with a period are still verbatim, amounts included.
        (
            "Me operaron, estoy de licencia sin sueldo y no llego",
            "Me operaron. estoy de licencia sin sueldo",
            "vulnerabilidad",
            True,
        ),
        (
            "Me cargaron $5.600 por algo que no pedí",
            "Me cargaron $5.600 por algo que no pedí",
            "reclamo",
            True,
        ),
        # Scattered single words are not a quote.
        ("Me quedé sin trabajo", "sin. trabajo", "vulnerabilidad", False),
        # Rejecting a plan is not a hardship (promoted from evals/blind A-62:b1, addendum 3).
        (
            "No llego con esas tres cuotas, lo descarto.",
            "No llego con esas tres cuotas, lo descarto.",
            "vulnerabilidad",
            False,
        ),
        (
            "Me bajaron las horas en la obra y cobro la mitad",
            "Me bajaron las horas en la obra",
            "vulnerabilidad",
            True,
        ),
        ("Quiero hablar con una persona ya", "hablar con una persona", "pedido_explicito", True),
        # Rejecting a plan is not a dispute (promoted from evals/blind A-62:b1, addendum 3).
        ("Mirá, yo no puedo en 3 cuotas, lo rechazo.", "lo rechazo", "reclamo", False),
        ("No me sirve ese plan", "No me sirve ese plan", "amenaza_legal", False),
        ("Voy a consultar con mi abogado", "consultar con mi abogado", "amenaza_legal", True),
        # Promoted from evals/blind (ADR-010, addendum 2) with the quote the live classifier
        # returned. They need the model, so the offline canonical suite cannot hold them.
        (
            "Cortá con la grabación. Decime el nombre de la persona que me atiende en serio y que "
            "tenga potestad para resolver esto ahora.",
            "Decime el nombre de la persona que me atiende en serio y que tenga potestad para "
            "resolver esto ahora.",
            "pedido_explicito",
            True,
        ),
        (
            "Atendeme: quiero que un supervisor tome mi caso y me dé una solución concreta ahora. "
            "Las respuestas enlatadas ya no sirven.",
            "\u201cAtendeme\u201d y \u201cun supervisor tome mi caso\u201d",
            "pedido_explicito",
            True,
        ),
        (
            "Hola, me operaron hace un mes y estoy de licencia sin sueldo. Cobro una prestación "
            "mínima que no alcanza para remedios y alquiler, me llegó este débito y no lo puedo "
            "afrontar. ¿Se puede prorrogar el vencimiento o armar cuotas muy chicas hasta que "
            "vuelva a trabajar? No doy más con los gastos médicos.",
            "me operaron hace un mes y estoy de licencia sin sueldo. Cobro una prestación mínima "
            "que no alcanza para remedios y alquiler. no lo puedo afrontar.",
            "vulnerabilidad",
            True,
        ),
    ],
)
def test_escalation_evidence_must_be_quoted_and_name_a_cause(
    text: str, evidence: str, signal: str, holds: bool
) -> None:
    assert escalation_evidence_holds(text, evidence, signal) is holds


async def test_classifier_request_for_a_role_outside_the_lexicon_is_derived() -> None:
    signal = GuardModelResult(
        escalation_signal="pedido_explicito", escalation_evidence="que me atienda un supervisor"
    )
    async with agent_runtime(guard_classifier=ScriptedLLM([signal])) as runtime:
        await _say(runtime, "Necesito que me atienda un supervisor hoy")
        assert [c.arguments for c in runtime.recorder.tool_calls] == [
            {"motivo": "pedido_explicito"}
        ]
        assert any(
            e["type"] == "escalation_signal_from_classifier" for e in runtime.recorder.events
        )


async def test_classifier_vulnerability_without_a_cause_keeps_the_negotiation() -> None:
    # Live k=5 regression: gpt-5-nano labelled payment difficulty as vulnerability (N-01, N-04).
    signal = GuardModelResult(
        escalation_signal="vulnerabilidad", escalation_evidence="no llego con el total"
    )
    async with agent_runtime(guard_classifier=ScriptedLLM([signal])) as runtime:
        await _say(runtime, "No llego con el total, ¿qué opciones hay?")
        assert not any(c.name == "request_human" for c in runtime.recorder.tool_calls)
        assert any(e["type"] == "escalation_signal_ungrounded" for e in runtime.recorder.events)


async def test_classifier_signal_never_changes_or_overrides_protocols() -> None:
    # An existing deterministic derivation keeps its reason.
    reclamo = GuardModelResult(escalation_signal="reclamo")
    async with agent_runtime(guard_classifier=ScriptedLLM([reclamo])) as runtime:
        await _say(runtime, "Quiero hablar con un asesor")
        assert [c.arguments for c in runtime.recorder.tool_calls] == [
            {"motivo": "pedido_explicito"}
        ]
    # A turn the guard restricts or deflects never takes a model-only derivation.
    attack = GuardModelResult(label="injection", confidence=1.0, escalation_signal="reclamo")
    async with agent_runtime(guard_classifier=ScriptedLLM([attack])) as runtime:
        await _say(runtime, "texto cualquiera")
        assert runtime.recorder.tool_calls == []
    # An ungrounded model signal cannot discard a pending draft.
    pending = GuardModelResult(escalation_signal="vulnerabilidad")
    async with agent_runtime(guard_classifier=ScriptedLLM([pending])) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        await runtime.seed(conversation, {"pending_draft": await fixture_draft(runtime)})
        result = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "eso lo vemos",
            context=runtime.context,
        )
        assert not any(c.name == "request_human" for c in runtime.recorder.tool_calls)
    assert result.state["pending_draft"] is not None


@pytest.mark.parametrize(
    ("message", "motivo", "expected"),
    [
        (
            "No puedo seguir con esto: me echaron del trabajo y no sé qué hacer.",
            "vulnerabilidad",
            "ya te derivé con prioridad",
        ),
        ("Esa deuda no es mía, quiero reclamarla.", "reclamo", "ya te derivé"),
        ("Quiero hablar con una persona antes de confirmar.", "pedido_explicito", "ya te derivé"),
        ("Mi abogado va a iniciar una demanda.", "amenaza_legal", "ya te derivé"),
    ],
)
async def test_escalation_during_confirmation_cancels_draft_and_outranks_protocol(
    message: str, motivo: str, expected: str
) -> None:
    async with agent_runtime() as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        await runtime.seed(conversation, {"pending_draft": await fixture_draft(runtime)})
        result = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            message,
            context=runtime.context,
        )
        transfers = [
            call.arguments["motivo"]
            for call in runtime.recorder.tool_calls
            if call.name == "request_human"
        ]
        assert transfers == [motivo]
        assert expected in result.text
        assert "¿Confirmás este acuerdo?" not in result.text
        assert result.state["pending_draft"] is None


async def test_grounded_model_escalation_during_confirmation_outranks_protocol() -> None:
    signal = GuardModelResult(
        escalation_signal="vulnerabilidad", escalation_evidence="retuvo todo mi sueldo"
    )
    async with agent_runtime(guard_classifier=ScriptedLLM([signal])) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        await runtime.seed(conversation, {"pending_draft": await fixture_draft(runtime)})
        result = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "Mi empleador retuvo todo mi sueldo; no puedo decidir sobre el plan.",
            context=runtime.context,
        )
        assert [
            call.arguments["motivo"]
            for call in runtime.recorder.tool_calls
            if call.name == "request_human"
        ] == ["vulnerabilidad"]
        assert result.state["pending_draft"] is None


async def test_list_policy_extract_keeps_every_item() -> None:
    # Local chat regression: a 3-sentence cap dropped debit and transfer from PAY-MET-001.
    async with agent_runtime(retriever=StaticRetriever([corpus_chunk("PAY-MET-001")])) as runtime:
        result = await _say(runtime, "¿Qué medios de pago puedo usar?")
    for method in ("Débito automático", "Transferencia bancaria", "Tarjeta de crédito", "Cupón"):
        assert method in result.text
    assert result.text.endswith("[PAY-MET-001]")


async def test_after_an_agreement_the_first_installment_comes_from_the_plan() -> None:
    # Local chat regression: the question listed the arrears instead of the new plan's date.
    await idempotency_store.reset()  # the in-process mock keeps agreements from earlier tests
    async with agent_runtime() as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")

        async def say(text: str) -> Any:
            return await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )

        await say("Quiero la opción de 3 cuotas")
        registered = await say("sí")
        assert "quedó registrado" in registered.text
        plan = await say("¿Cuándo vence la primera cuota?")
        arrears = await say("¿Cuáles son mis vencimientos?")
    assert "Tu plan de 3 cuotas de $61.500 (total $184.500)" in plan.text
    assert "20/09/2026" in plan.text and registered.state["agreement_id"] in plan.text
    assert "vencimientos impagos" in arrears.text


async def test_after_a_transfer_the_agent_answers_but_no_longer_negotiates() -> None:
    # Local chat regression: after deriving a prejudicial account it asked how much could be paid.
    async with agent_runtime("CUST-00212") as runtime:
        conversation = await runtime.service.create_conversation("CUST-00212")

        async def say(text: str) -> Any:
            return await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )

        derived = await say("Quiero pagarlo en cuotas")
        amount = await say("Quiero pagar lo que pueda")
        option = await say("Quiero la opción de 3 cuotas")
        balance = await say("¿Cuánto debo?")
        menu = await say("mmm")
        person = await say("Quiero hablar con una persona")
        vulnerable = await say("Me quedé sin trabajo y no puedo pagar")
        motivos = [
            c.arguments["motivo"] for c in runtime.recorder.tool_calls if c.name == "request_human"
        ]
    assert "Ya te derivé" in derived.text
    assert "ya quedó derivado" in amount.text and "ya quedó derivado" in option.text
    assert "$1.240.000" in balance.text
    assert "veamos alternativas" not in balance.text and not balance.state["offered_next_step"]
    assert "alternativas" not in menu.text and "asesor" in menu.text
    # Asking for a person once a person owns the case is not a second transfer; a vulnerability
    # signal is, because it changes the case's priority.
    assert "ya quedó derivado" in person.text
    assert "con prioridad" in vulnerable.text
    assert motivos == ["fuera_de_politica", "vulnerabilidad"]


async def test_zero_debt_customer_gets_no_plan_nor_transfer() -> None:
    async with agent_runtime("CUST-00450") as runtime:
        conversation = await runtime.service.create_conversation("CUST-00450")
        texts = []
        for text in (
            "Quiero pagarlo en cuotas",
            "Quiero pagar lo que pueda",
            "Quiero la opción de 3 cuotas",
        ):
            result = await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )
            texts.append(result.text)
        assert all(text.startswith("No registrás deuda vigente") for text in texts)
        assert not any(c.name == "request_human" for c in runtime.recorder.tool_calls)


async def test_short_yes_to_our_confirmation_is_not_a_request_for_a_person() -> None:
    # gpt-5-nano occasionally reads "ok mandale" as pedido_explicito: an allow-listed answer to our
    # own confirmation question keeps the confirmation protocol instead of transferring.
    await idempotency_store.reset()
    signal = GuardModelResult(
        escalation_signal="pedido_explicito", escalation_evidence="ok mandale"
    )
    async with agent_runtime(guard_classifier=ScriptedLLM([signal])) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        await runtime.seed(conversation, {"pending_draft": await fixture_draft(runtime)})
        result = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "ok mandale",
            context=runtime.context,
        )
        assert not any(c.name == "request_human" for c in runtime.recorder.tool_calls)
    assert "quedó registrado" in result.text


async def test_grounded_reason_corrects_a_model_routed_derivation() -> None:
    # Promoted from evals/blind E-62:b4: the model router derived with "pedido_explicito" while the
    # classifier quoted a dispute. A model-made reason is unquoted; the grounded one wins.
    text = (
        "Esto no es mio. Me vienen cargando $19.800 por un servicio que jamas suscribí. Corrijan "
        "eso YA y confirmen por mensaje que lo eliminaron."
    )
    router = ScriptedLLM(
        [RouteResult(intent="pedido_humano", escalation_motivo="pedido_explicito")]
    )
    signal = GuardModelResult(
        escalation_signal="reclamo",
        escalation_evidence="Me vienen cargando $19.800 por un servicio que jamas suscribí.",
    )
    async with agent_runtime(llm=router, guard_classifier=ScriptedLLM([signal])) as runtime:
        await _say(runtime, text)
        assert [c.arguments for c in runtime.recorder.tool_calls] == [{"motivo": "reclamo"}]


@pytest.mark.parametrize(
    ("text", "reply"),
    [
        ("sí", "accept"),
        ("si me sirve", "accept"),
        ("dale, esa", "accept"),
        ("me interesa", "accept"),
        ("okey", "accept"),
        ("de una", "accept"),
        ("nop", "reject"),
        ("sí, no hay problema", "accept"),
        ("mejor no", "reject"),
        ("esa no", "reject"),
        ("no me sirve", "reject"),
        ("prefiero ver las otras", "reject"),
        ("lo pienso", None),
        ("¿y cuánto es el anticipo de esa opción?", None),
        ("", None),
    ],
)
def test_short_replies_to_a_proposed_option(text: str, reply: str | None) -> None:
    assert proposal_reply(text) == reply


async def test_a_bare_yes_or_no_answers_the_offer_just_made() -> None:
    # Local chat regressions: after "¿Te sirve esa?" or "¿Querés que veamos alternativas?", a bare
    # "si me sirve", "mejor no" or "no" fell back to the generic menu.
    await idempotency_store.reset()
    async with agent_runtime() as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")

        async def say(text: str) -> Any:
            return await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )

        proposed = await say("¿Puedo pagar en 9 cuotas?")
        assert proposed.state["proposed_option_id"] == "OPT-9C"
        rejected = await say("mejor no")
        assert "puedo ofrecerte" in rejected.text and not rejected.state["proposed_option_id"]
        balance = await say("¿Cuánto debo?")
        assert balance.state["offered_next_step"] == "options"
        declined = await say("no")
        assert declined.text.startswith("Entendido.")
        changed_mind = await say("si quiero")
        assert "puedo ofrecerte" in changed_mind.text
        await say("¿Cuánto debo?")
        options = await say("sí")
        assert "puedo ofrecerte" in options.text
        await say("¿Puedo pagar en 9 cuotas?")
        accepted = await say("si me sirve")
        assert (
            "9 cuotas de $21.402" in accepted.text and "¿Confirmás este acuerdo?" in accepted.text
        )
        assert _writes_of(runtime) == []  # selecting is not registering: "sí" is still required
        assert not accepted.state["proposed_option_id"]  # nothing is proposed during a draft


async def test_yes_to_an_offered_transfer_derives() -> None:
    async with agent_runtime(retriever=StaticRetriever([])) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        offer = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "¿Cuánto tarda la acreditación?",
            context=runtime.context,
        )
        assert offer.state["offered_next_step"] == "human"
        await runtime.service.send_message(
            conversation.conversation_id, conversation.customer_id, "dale", context=runtime.context
        )
        assert [c.arguments for c in runtime.recorder.tool_calls if c.name == "request_human"] == [
            {"motivo": "pedido_explicito"}
        ]


def _writes_of(runtime: AgentRuntime) -> list[tuple[str, str]]:
    return [r for r in runtime.transport.requests if r[1] == "/payment-agreement"]


def test_an_empty_option_list_offers_a_transfer() -> None:
    from app.graph.nodes.respond import _offered_next_step

    # "No hay opciones automáticas habilitadas. Te puedo derivar." makes a bare "sí" a transfer.
    assert (
        _offered_next_step(ResponsePlan(kind="negotiation", template_id="options"), {}) == "human"
    )
    assert _offered_next_step(ResponsePlan(kind="direct", template_id="greeting"), {}) == ""


async def test_the_model_reads_short_replies_the_lexicon_misses() -> None:
    from app.graph.nodes.guards import OfferReply

    async with agent_runtime(
        llm=ScriptedLLM(
            [OfferReply(reply="accept"), OfferReply(reply="other"), RuntimeError("down")]
        )
    ) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")

        async def say(text: str) -> Any:
            return await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )

        await say("¿Cuánto debo?")
        options = await say("puede ser")
        assert "puedo ofrecerte" in options.text
        await say("¿Cuánto debo?")
        other = await say("hmm capaz")
        assert other.text.startswith("¿Querés consultar el saldo")
        await say("¿Cuánto debo?")
        await say("mmm veremos")  # classifier down: falls back to the normal routing
        assert "offer_reply_classifier_unavailable" in [e["type"] for e in runtime.recorder.events]
        # A message with its own intent is never read as an answer to the offer.
        await say("¿Cuánto debo?")
        person = await say("quiero hablar con una persona")
        assert "derivé" in person.text


@pytest.mark.parametrize(
    ("text", "choice"),
    [
        ("la 4", ("index", 4)),
        ("opción 2", ("index", 2)),
        ("me quedo con la cuarta", ("index", 4)),
        ("la de 6", ("installments", 6)),
        ("3 cuotas", None),
        ("la 4 o la 3, no sé", None),
    ],
)
def test_listed_choice_reads_positions_and_installments(text: str, choice: object) -> None:
    assert listed_choice(text) == choice


async def test_choosing_from_the_list_just_shown_builds_a_draft() -> None:
    # Local chat regression: "elijo las 9 cuotas" after the list was answered as a question.
    for reply, installments in (
        ("elijo las 9 cuotas", 9),
        ("la 2", 3),
        ("la de 6", 6),
        ("la 9", None),
    ):
        await idempotency_store.reset()
        async with agent_runtime() as runtime:
            conversation = await runtime.service.create_conversation("CUST-00125")

            async def say(
                text: str, conversation: Any = conversation, runtime: AgentRuntime = runtime
            ) -> Any:
                return await runtime.service.send_message(
                    conversation.conversation_id,
                    conversation.customer_id,
                    text,
                    context=runtime.context,
                )

            listed = await say("No puedo pagar todo este mes, ¿qué opciones tengo?")
            assert listed.state["offered_next_step"] == "choose"
            result = await say(reply)
            if installments is None:  # there is no fifth... ninth option: no draft is invented
                assert result.state.get("pending_draft") is None
            else:
                assert f"{installments} cuotas" in result.text
                assert "¿Confirmás este acuerdo?" in result.text
            assert _writes_of(runtime) == []


def test_no_quiero_is_not_a_choice() -> None:
    assert route_turn("no quiero las 3 cuotas").intent != "aceptar_opcion"
    assert route_turn("elijo las 9 cuotas").intent == "aceptar_opcion"


@pytest.mark.parametrize(
    ("text", "amount"),
    [
        ("1000", 1000),
        ("$ 30.000", 30000),
        ("unos 30 mil", 30000),
        ("mil pesos", 1000),
        ("en 6 cuotas", None),
        ("no sé", None),
    ],
)
def test_monthly_amount_reads_short_answers(text: str, amount: int | None) -> None:
    assert monthly_amount(text) == amount


def test_cannot_pay_asks_for_alternatives() -> None:
    assert route_turn("no puedo pagar todo").intent == "negociacion"
    assert route_turn("no me alcanza para la cuota").intent == "negociacion"
    assert route_turn("quiero pagar mi deuda pero se me hace dificil").intent == "negociacion"


async def test_an_amount_answer_proposes_what_fits_or_offers_a_person() -> None:
    # Local chat regression: "1000" after "¿cuánto podrías pagar?" fell back to the generic menu.
    async with agent_runtime() as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")

        async def say(text: str) -> Any:
            return await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )

        asked = await say("Quiero pagar lo que pueda")
        assert asked.state["offered_next_step"] == "amount"
        below = await say("1000")
        assert "la cuota más baja" in below.text and "1000" not in below.text
        assert below.state["offered_next_step"] == "human"
        await say("Quiero pagar lo que pueda")
        fits = await say("unos 70 mil")
        assert "3 cuotas de $61.500" in fits.text
        confirm = await say("sí")
        assert "¿Confirmás este acuerdo?" in confirm.text
        await say("no")  # cancels the draft
        await say("Quiero pagar lo que pueda")
        await say("1000")
        derived = await say("sí")
    assert "derivé" in derived.text


async def test_offer_edges_fall_back_safely() -> None:
    from app.graph.nodes.respond import _template_text
    from app.graph.recorder import TurnBudgetExceeded

    async with agent_runtime() as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "¿Qué opciones tengo?",
            context=runtime.context,
        )
        # "sí" to "¿Alguna te sirve?" names no option: the list is shown again.
        again = await runtime.service.send_message(
            conversation.conversation_id, conversation.customer_id, "sí", context=runtime.context
        )
        assert "puedo ofrecerte" in again.text
    # A budget overrun while reading the reply is the controlled budget path, never swallowed.
    async with agent_runtime(llm=ScriptedLLM([TurnBudgetExceeded("llm", 3)])) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        for text in ("¿Cuánto debo?", "puede ser"):
            result = await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )
        assert "límite seguro" in result.text
    # No allowed option at all: the amount answer offers a person.
    below = _template_text(ResponsePlan(kind="negotiation", template_id="amount_below_options"), {})
    assert below.startswith("No hay opciones automáticas")


def test_closing_and_reopening_helpers() -> None:
    assert bare_installments("9 cuotas") == 9 and bare_installments("las 6 cuotas") == 6
    assert bare_installments("¿puedo pagar en 9 cuotas?") is None
    assert reopens_offer("sí, quiero") and reopens_offer("mostrame") and not reopens_offer("bueno")


async def test_closings_and_bare_installments_after_a_list() -> None:
    # Local chat regressions: "9 cuotas" after the list re-proposed instead of choosing,
    # "bueno muchas gracias" greeted again, and "bueno" after declining reopened the options.
    await idempotency_store.reset()
    async with agent_runtime() as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")

        async def say(text: str) -> Any:
            return await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )

        await say("¿Qué opciones tengo?")
        question = await say("¿puedo pagar en 9 cuotas?")
        assert "¿Te sirve esa" in question.text
        await say("¿Qué opciones tengo?")
        chosen = await say("9 cuotas")
        assert "¿Confirmás este acuerdo?" in chosen.text
        registered = await say("sí")
        assert "quedó registrado" in registered.text
        thanks = await say("bueno muchas gracias")
        assert thanks.text.startswith("De nada.")
        await say("¿Cuánto debo?")
        await say("no")
        ack = await say("bueno")
        assert ack.text.startswith("Perfecto.") and "puedo ofrecerte" not in ack.text
        bye = await say("chau")
        assert bye.text.startswith("Hasta pronto.")
        hello = await say("hola")
        assert hello.text.startswith("Hola, soy")


async def test_confirmation_lists_every_committed_figure() -> None:
    # Local chat regression: the 9-installment summary omitted the $18.450 advance.
    from datetime import date as _date
    from decimal import Decimal as _Decimal

    from app.graph.nodes.respond import _confirmation_text
    from app.tools.schemas import AgreementDraft

    await idempotency_store.reset()
    async with agent_runtime() as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")

        async def say(text: str) -> Any:
            return await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )

        summary = await say("Quiero la opción de 9 cuotas")
        assert "anticipo de $18.450 y 9 cuotas de $21.402, total $211.068" in summary.text
        await say("sí")
        plan = await say("¿Cuándo vence la primera cuota?")
        assert "Incluye un anticipo de $18.450." in plan.text
    single = AgreementDraft(
        draft_id="draft-single",
        opcion_id="OPT-1P",
        monto_total=_Decimal("178000"),
        cuotas=1,
        monto_cuota=_Decimal("178000"),
        fecha_primer_vencimiento=_date(2026, 9, 20),
        medio_pago="debito_automatico",
        debt_fingerprint="a" * 64,
        expires_at=REFERENCE_NOW,
        policy_refs=[],
    )
    assert "un pago único de $178.000, con vencimiento el 20/09/2026" in _confirmation_text(single)


def test_plan_asks_for_alternatives_unless_off_topic() -> None:
    assert route_turn("quiero un plan").intent == "negociacion"
    assert route_turn("quiero armar un plan de pagos").intent == "negociacion"
    assert route_turn("¿me recomendás un plan de ahorro?").intent == "fuera_de_dominio"
    # A dispute that mentions a plan is not a request for one (evals/blind E-62:b3).
    assert route_turn("Me cobran un plan que nunca firmé").intent != "negociacion"


async def test_balance_of_an_account_that_needs_an_advisor_offers_the_advisor() -> None:
    # Local chat regression: CUST-00377 was offered alternatives, then told it needed an advisor.
    async with agent_runtime("CUST-00377") as runtime:
        conversation = await runtime.service.create_conversation("CUST-00377")

        async def say(text: str) -> Any:
            return await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )

        balance = await say("¿Cuánto debo?")
        assert "tiene que revisar un asesor" in balance.text
        assert "veamos alternativas" not in balance.text
        derived = await say("dale")
        motivos = [
            c.arguments["motivo"] for c in runtime.recorder.tool_calls if c.name == "request_human"
        ]
    assert "derivé" in derived.text and motivos == ["identidad_no_verificada"]


async def test_zero_debt_mentions_the_last_credited_payment() -> None:
    async with agent_runtime("CUST-00450") as runtime:
        conversation = await runtime.service.create_conversation("CUST-00450")
        result = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "¿Tengo deuda?",
            context=runtime.context,
        )
    assert result.text.startswith("No registrás deuda vigente: tu último pago, de $74.200")


def test_short_whole_statements_and_grouped_claims_are_supported() -> None:
    # Live evaluation: correct model answers citing table rows were rejected as unsupported.
    from app.guards.grounding import GroundedReply, verify_grounded_reply

    source = {"POL-NEG-003": corpus_chunk("POL-NEG-003").chunk.content}
    reply = GroundedReply.model_validate(
        {
            "text": "",
            "claims": [
                {
                    "sentence": "Un plan en cuotas no acumula quita.",
                    "section_id": "POL-NEG-003",
                    "quote": "Un plan en cuotas no acumula quita.",
                },
                {
                    "sentence": "Mora temprana: 0 %. Mora media: 20 %. Mora tardía: 40 %.",
                    "section_id": "POL-NEG-003",
                    "quote": "Mora temprana: 0 %. Mora media: 20 %. Mora tardía: 40 %.",
                },
                {
                    "sentence": "Prejudicial: requiere operador.",
                    "section_id": "POL-NEG-003",
                    "quote": "Prejudicial: requiere operador.",
                },
            ],
        }
    )
    composed = reply.model_copy(update={"text": " ".join(claim.sentence for claim in reply.claims)})
    assert verify_grounded_reply(composed, source) == ()
    # A short fragment that is not a whole statement is still not evidence.
    fragment = GroundedReply.model_validate(
        {
            "text": "Requiere operador.",
            "claims": [
                {
                    "sentence": "Requiere operador.",
                    "section_id": "POL-NEG-003",
                    "quote": "requiere operador",
                }
            ],
        }
    )
    assert "quote_not_in_source" in verify_grounded_reply(fragment, source)
