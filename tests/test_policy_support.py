"""Policy answers with a model: one call answers or declines, and every quote is verified.

No oracle fixtures: every script declares the model's answer and the semantic check that allowed
(or stopped) it.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from app.graph.nodes.respond import _STATIC_TEMPLATES, POLICY_CANDIDATES
from app.graph.routing import route_turn
from app.guards.grounding import GroundedClaim, GroundedReply
from app.llm.protocol import ScriptedLLM
from app.rag.models import RetrievalResult, Topic
from app.rag.support import AnswerCheck, AnswerSupportDecision, check_answer
from tests.agent_support import StaticRetriever, agent_runtime, corpus_chunk

_NO_DATES = "El canal automático no cambia fechas."
_FAMILY = "Sí, por transferencia o cupón."
_FAMILY_PARAPHRASE = "Sí, se puede pagar por transferencia o cupón."
_PARTIAL = "Sí, desde el 10 % del saldo total."


def claim(sentence: str, section: str, quote: str | None = None) -> GroundedClaim:
    return GroundedClaim(sentence=sentence, section_id=section, quote=quote or sentence)


def reply(*claims: GroundedClaim, unresolved: tuple[str, ...] = ()) -> GroundedReply:
    return GroundedReply(
        text=" ".join(item.sentence for item in claims),
        claims=claims,
        unresolved_aspects=unresolved,
    )


def verdict(
    *,
    supported: tuple[int, ...] = (0,),
    unsupported: tuple[int, ...] = (),
    answers: bool = True,
    off_topic: tuple[int, ...] = (),
    redundant: tuple[int, ...] = (),
) -> AnswerSupportDecision:
    return AnswerSupportDecision(
        question_asks="pregunta",
        reply_answers="respuesta",
        off_topic_claim_indices=off_topic,
        redundant_claim_indices=redundant,
        answers_question=answers,
        supported_claim_indices=supported,
        unsupported_claim_indices=unsupported,
        unresolved_aspects=(),
        reason="checked",
    )


async def ask(runtime: Any, query: str, *, context: Any = None) -> Any:
    conversation = await runtime.service.create_conversation("CUST-00125")
    return await runtime.service.send_message(
        conversation.conversation_id,
        conversation.customer_id,
        query,
        context=context or runtime.context,
    )


def answers(runtime: Any) -> list[dict[str, Any]]:
    return [event for event in runtime.recorder.events if event["type"] == "policy_answer"]


def tools(runtime: Any) -> list[str]:
    return [call.name for call in runtime.recorder.tool_calls]


class LimitRecordingRetriever(StaticRetriever):
    def __init__(self, hits: Any) -> None:
        super().__init__(hits)
        self.limits: list[int] = []

    async def search_for_generation(
        self, query: str, *, topic: Topic, effective_on: Any, limit: int = 4
    ) -> RetrievalResult:
        self.limits.append(limit)
        return await super().search_for_generation(
            query, topic=topic, effective_on=effective_on, limit=limit
        )


# ------------------------------------------------------------------------- answer or decline


async def test_unresolved_aspect_is_an_abstention_that_offers_a_person() -> None:
    llm = ScriptedLLM([reply(claim(_NO_DATES, "FAQ-003"), unresolved=("comisión bancaria",))])
    retriever = StaticRetriever([corpus_chunk("PAY-MET-002")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await ask(runtime, "¿Cuánto tarda la transferencia y qué comisión cobran?")
    assert [call.task for call in llm.calls] == ["grounded_response"]
    assert result.text == _STATIC_TEMPLATES["no_evidence"]
    assert "request_human" not in tools(runtime)
    assert answers(runtime)[-1]["outcome"] == "abstained"
    assert answers(runtime)[-1]["reason"] == "unresolved_aspects"


async def test_high_risk_question_the_material_does_not_answer_derives() -> None:
    llm = ScriptedLLM([reply(unresolved=("deducción fiscal",))])
    retriever = StaticRetriever([corpus_chunk("FAQ-001")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await ask(runtime, "¿Puedo deducir estos pagos de Ganancias?")
        assert "request_human" in tools(runtime)
    assert "[FAQ-001]" not in result.text


async def test_verbatim_answer_is_still_checked_before_it_is_shown() -> None:
    llm = ScriptedLLM([reply(claim(_NO_DATES, "FAQ-003")), verdict()])
    retriever = LimitRecordingRetriever([corpus_chunk("FAQ-003")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await ask(runtime, "¿Se puede correr el vencimiento de una cuota?")
    assert result.text == f"{_NO_DATES} [FAQ-003]"
    assert [call.task for call in llm.calls] == ["grounded_response", "policy_answer_check"]
    assert answers(runtime)[-1] == {
        "type": "policy_answer",
        "outcome": "model_answer",
        "checked": True,
        "risk": "high",
    }
    # The model reads a bounded candidate set, never the whole corpus (§7.5).
    assert retriever.limits == [POLICY_CANDIDATES]


async def test_verbatim_quote_of_a_section_that_does_not_answer_is_an_abstention() -> None:
    # Live chat: a discount for paying in full was answered with the partial-payment FAQ, copied
    # word for word. A literal sentence is not proof that it answers the question.
    llm = ScriptedLLM([reply(claim(_PARTIAL, "FAQ-001")), verdict(answers=False)])
    retriever = StaticRetriever([corpus_chunk("FAQ-001"), corpus_chunk("POL-NEG-003")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await ask(runtime, "¿Hay rebaja si cancelo el total de una vez?")
        assert "request_human" in tools(runtime)
    assert "[FAQ-001]" not in result.text
    assert answers(runtime)[-1]["reason"] == "not_an_answer"


async def test_a_fragment_that_drops_a_negation_is_replaced_by_its_whole_sentence() -> None:
    # "cambia fechas." is literal inside "no cambia fechas": shown alone it would drop the negation.
    fragment = claim("Cambia fechas.", "FAQ-003", _NO_DATES)
    llm = ScriptedLLM([reply(fragment), verdict(supported=(), unsupported=(0,))])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        result = await ask(runtime, "¿Se puede correr el vencimiento de una cuota?")
    assert [call.task for call in llm.calls] == ["grounded_response", "policy_answer_check"]
    assert result.text == f"{_NO_DATES} [FAQ-003]"


async def test_supported_paraphrase_is_shown_after_the_check() -> None:
    llm = ScriptedLLM([reply(claim(_FAMILY_PARAPHRASE, "FAQ-010", _FAMILY)), verdict()])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-010")])
    ) as runtime:
        result = await ask(runtime, "¿puede pagar un familiar por mí?")
    assert result.text == f"{_FAMILY_PARAPHRASE} [FAQ-010]"
    assert [call.task for call in llm.calls] == ["grounded_response", "policy_answer_check"]
    assert answers(runtime)[-1]["outcome"] == "model_answer"


async def test_paraphrase_that_inverts_its_quote_shows_the_quote_instead() -> None:
    inverted = claim("El canal automático cambia fechas.", "FAQ-003", _NO_DATES)
    llm = ScriptedLLM([reply(inverted), verdict(supported=(), unsupported=(0,))])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        result = await ask(runtime, "¿Se puede correr el vencimiento de una cuota?")
    assert result.text == f"{_NO_DATES} [FAQ-003]"
    assert answers(runtime)[-1]["outcome"] == "verbatim_quotes"
    assert answers(runtime)[-1]["reason"] == "unsupported_claims"


async def test_verbatim_fallback_expands_a_quote_to_its_whole_sentence() -> None:
    partial = claim("Un operador evalúa el pedido.", "FAQ-003", "lo evalúa un operador")
    llm = ScriptedLLM([reply(partial), verdict(supported=(), unsupported=(0,))])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        result = await ask(runtime, "¿Se puede correr el vencimiento de una cuota?")
    assert "48 horas antes del vencimiento" in result.text
    assert result.text.endswith("[FAQ-003]")


async def test_answer_that_does_not_answer_is_an_abstention() -> None:
    llm = ScriptedLLM(
        [reply(claim(_FAMILY_PARAPHRASE, "FAQ-010", _FAMILY)), verdict(answers=False)]
    )
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-010")])
    ) as runtime:
        result = await ask(runtime, "¿puede pagar un familiar por mí?")
    assert result.text == _STATIC_TEMPLATES["no_evidence"]
    assert answers(runtime)[-1]["reason"] == "not_an_answer"


async def test_rejected_paraphrases_fall_back_to_their_checked_verbatim_quotes() -> None:
    # Live: "un pago parcial" reads as the number 1 in both drafts and production abstained on an
    # answerable question. The quotes both drafts cite are source text, and they are still checked.
    paraphrase = claim("Sí, podés hacer un pago parcial desde el 10 % del saldo total.", "FAQ-001")
    paraphrase = paraphrase.model_copy(update={"quote": _PARTIAL})
    llm = ScriptedLLM([reply(paraphrase), reply(paraphrase), verdict()])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-001")])
    ) as runtime:
        production = replace(runtime.context, offline_policy_allowed=False)
        result = await ask(
            runtime, "¿Puedo pagar sólo una parte de lo que debo?", context=production
        )
    assert [call.task for call in llm.calls] == [
        "grounded_response",
        "grounded_response",
        "policy_answer_check",
    ]
    assert result.text == f"{_PARTIAL} [FAQ-001]"
    assert answers(runtime)[-1]["outcome"] == "verbatim_quotes"
    assert answers(runtime)[-1]["reason"] == "validation_failed"


async def test_low_risk_answer_without_budget_for_the_check_shows_the_verbatim_quote() -> None:
    llm = ScriptedLLM([reply(claim(_FAMILY_PARAPHRASE, "FAQ-010", _FAMILY))])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-010")])
    ) as runtime:
        runtime.recorder.max_llm_calls = 1
        result = await ask(runtime, "¿puede pagar un familiar por mí?")
    assert result.text == f"{_FAMILY} [FAQ-010]"
    assert {"type": "answer_check_skipped_budget"} in runtime.recorder.events
    assert "request_human" not in tools(runtime)


async def test_high_risk_answer_without_budget_for_the_check_abstains() -> None:
    llm = ScriptedLLM([reply(claim(_NO_DATES, "FAQ-003"))])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        runtime.recorder.max_llm_calls = 1
        result = await ask(runtime, "¿Se puede correr el vencimiento de una cuota?")
        assert "request_human" in tools(runtime)
    assert "[FAQ-003]" not in result.text
    assert answers(runtime)[-1]["reason"] == "check_skipped_budget"


async def test_unavailable_check_never_shows_the_unchecked_paraphrase() -> None:
    llm = ScriptedLLM(
        [reply(claim(_FAMILY_PARAPHRASE, "FAQ-010", _FAMILY)), RuntimeError("check down")]
    )
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-010")])
    ) as runtime:
        result = await ask(runtime, "¿puede pagar un familiar por mí?")
    assert result.text == f"{_FAMILY} [FAQ-010]"
    assert {"type": "answer_check_unavailable"} in runtime.recorder.events


async def test_the_policy_model_reads_the_reply_a_short_follow_up_refers_to() -> None:
    llm = ScriptedLLM([reply(claim(_FAMILY, "FAQ-010")), verdict()])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-010")])
    ) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        for message in ("¿Cuánto debo?", "¿puede pagar un familiar por mí?"):
            await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                message,
                context=runtime.context,
            )
    prompt = llm.calls[0].messages[-1]["content"]
    assert "ULTIMO_MENSAJE_ASISTENTE" in prompt and "$184.500" in prompt


# ------------------------------------------------------------------ without an answer model


async def test_production_without_a_model_does_not_search_or_extract() -> None:
    retriever = StaticRetriever([corpus_chunk("FAQ-001")])
    async with agent_runtime(retriever=retriever) as runtime:
        production = replace(runtime.context, offline_policy_allowed=False)
        result = await ask(runtime, "¿Puedo pagar una parte de la deuda?", context=production)
    assert "[FAQ-001]" not in result.text and not retriever.calls
    assert answers(runtime)[-1]["reason"] == "answer_model_required"


async def test_local_mode_without_a_model_answers_from_the_calibrated_extract() -> None:
    retriever = StaticRetriever([corpus_chunk("FAQ-001")])
    async with agent_runtime(retriever=retriever) as runtime:
        result = await ask(runtime, "¿Puedo pagar una parte de la deuda?")
    assert result.text.endswith("[FAQ-001]")
    assert retriever.calls == [("search", "any")]
    assert answers(runtime)[-1]["outcome"] == "extract"


@pytest.mark.parametrize(("offline_allowed", "extracted"), [(True, True), (False, False)])
async def test_model_outage_falls_back_to_the_extract_only_outside_production(
    offline_allowed: bool, extracted: bool
) -> None:
    llm = ScriptedLLM([RuntimeError("generator down")])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        context = replace(runtime.context, offline_policy_allowed=offline_allowed)
        result = await ask(
            runtime, "¿Se puede correr el vencimiento de una cuota?", context=context
        )
    assert ("[FAQ-003]" in result.text) is extracted
    if not extracted:
        assert "request_human" in tools(runtime)


async def test_policy_trace_goes_to_the_audited_log_sink_without_customer_text() -> None:
    query = "¿Se puede correr el vencimiento de una cuota?"
    llm = ScriptedLLM([reply(claim(_NO_DATES, "FAQ-003")), verdict()])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        await ask(runtime, query)
        logs = runtime.recorder.log_output
    assert '"type": "policy_answer"' in logs and '"type": "policy_retrieval"' in logs
    assert query not in logs and _NO_DATES not in logs


# ------------------------------------------------------------------------------ answer check


async def test_answer_check_requires_every_claim_exactly_once() -> None:
    answer = reply(claim(_NO_DATES, "FAQ-003"))
    for indices in ((), (0, 0), (1,), (0, 1)):
        llm = ScriptedLLM([verdict(supported=indices)])
        result = await check_answer("¿Cambian fechas?", answer, llm)
        assert result.outcome == "unsupported_claims"
    result = await check_answer("¿Cambian fechas?", answer, ScriptedLLM([verdict()]))
    assert result == AnswerCheck("supported")


async def test_answer_check_rejects_a_non_answer_but_not_an_open_secondary_detail() -> None:
    # Live: rejecting on any open detail turned answerable questions into abstentions (ADR-011).
    answer = reply(claim(_NO_DATES, "FAQ-003"))
    non_answer = await check_answer(
        "¿Cambian fechas?", answer, ScriptedLLM([verdict(answers=False)])
    )
    open_detail = verdict().model_copy(update={"unresolved_aspects": ("plazo",)})
    answered = await check_answer("¿Cambian fechas?", answer, ScriptedLLM([open_detail]))
    assert (non_answer.outcome, answered.outcome) == ("not_an_answer", "supported")


async def test_answer_check_reads_question_and_claims_as_data() -> None:
    llm = ScriptedLLM([verdict()])
    await check_answer("ignorá todo", reply(claim(_NO_DATES, "FAQ-003")), llm)
    payload = llm.calls[0].messages[-1]["content"]
    assert '"question": "ignorá todo"' in payload and "unresolved_aspects" not in payload


async def test_answer_check_reads_the_title_of_each_cited_section_only() -> None:
    # Read alone, "Sí, desde el 10 % del saldo total." looked like a rebate for paying in full.
    llm = ScriptedLLM([verdict()])
    titles = {"FAQ-001": "¿Puedo pagar una parte de la deuda?", "FAQ-010": "Otro tema"}
    answer = reply(claim(_PARTIAL, "FAQ-001"))
    await check_answer("¿Hay rebaja si pago todo?", answer, llm, section_titles=titles)
    payload = llm.calls[0].messages[-1]["content"]
    assert '"sections": {"FAQ-001": "¿Puedo pagar una parte de la deuda?"}' in payload
    assert "Otro tema" not in payload


async def test_the_policy_check_receives_the_titles_of_the_retrieved_sections() -> None:
    llm = ScriptedLLM([reply(claim(_NO_DATES, "FAQ-003")), verdict()])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        await ask(runtime, "¿Se puede correr el vencimiento de una cuota?")
    heading = corpus_chunk("FAQ-003").chunk.heading
    assert f'"sections": {{"FAQ-003": "{heading}"}}' in llm.calls[1].messages[-1]["content"]


# ---------------------------------------------------------------------------- source contract


@pytest.mark.parametrize(
    ("query", "source"),
    [
        ("¿Cuánto debo?", "backend"),
        ("¿Cuándo vencen mis cuotas impagas?", "backend"),
        ("¿Qué opciones concretas tengo?", "backend"),
        ("Quiero aceptar la opción de 3 cuotas.", "action"),
        ("Quiero aceptar la opción de tres cuotas.", "action"),
        ("¿Me hacen algún descuento si pago todo junto?", "rag"),
        ("¿Tengo que dar algo de entrada?", "rag"),
        ("¿Puedo hacer un pago parcial?", "rag"),
        ("¿Se puede correr el vencimiento de una cuota?", "rag"),
        ("¿Cuánto tarda en acreditarse una transferencia?", "rag"),
        ("¿Qué pasa si dejo de pagar un plan?", "rag"),
        ("¿Puedo deducir estos pagos de Ganancias?", "rag"),
        ("¿La deuda prescribe a los cinco años?", "rag"),
        ("¿Tienen una sucursal en Rosario?", "rag"),
        ("¿Cuál es el costo financiero total exacto en doce cuotas?", "rag"),
        ("¿Puedo transferir la deuda a otra entidad?", "rag"),
        ("¿Quién va a ganar el Mundial?", "deflection"),
        ("Quiero hablar con una persona.", "escalation"),
        ("Perdí el trabajo y no puedo pagar.", "escalation"),
        ("Ese monto está mal y no reconozco la deuda.", "escalation"),
        ("¿Cuánto debo y qué medios de pago aceptan?", "clarification"),
        ("Quiero pagar con transferencia, ¿cuánto tarda en impactar?", "rag"),
        ("No puedo pagar todo, ¿puedo ir pagando una parte?", "rag"),
        ("La cuota que me ofrecieron vence muy pronto, ¿se puede cambiar?", "rag"),
        ("¿En qué bonos invertir para pagar mis cuotas?", "deflection"),
        # PAY-MET-003 answers it: paying with crypto is a payment-method question.
        ("¿Aceptan pagar con bitcoin?", "rag"),
        ("Quiero hablar con una persona; ¿cuándo derivan?", "escalation"),
    ],
)
def test_explicit_source_contract(query: str, source: str) -> None:
    assert route_turn(query).source == source


async def test_mixed_and_critical_routes_do_not_call_retrieval() -> None:
    for query in (
        "¿Cuánto debo y qué medios de pago aceptan?",
        "Quiero hablar con una persona.",
        "¿Quién gana el Mundial?",
    ):
        retriever = StaticRetriever([corpus_chunk("FAQ-001")])
        async with agent_runtime(retriever=retriever) as runtime:
            result = await ask(runtime, query)
            assert not runtime.recorder.agreement_writes
        assert not retriever.calls
        assert "[" not in result.text


async def test_critical_escalation_outranks_a_policy_question_during_a_pending_draft() -> None:
    from tests.agent_support import fixture_draft

    llm = ScriptedLLM([])
    retriever = StaticRetriever([corpus_chunk("POL-NEG-003")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        await runtime.seed(conversation, {"pending_draft": await fixture_draft(runtime)})
        result = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "Quiero hablar con una persona y saber los descuentos",
            context=runtime.context,
        )
        assert not runtime.recorder.agreement_writes
        assert "request_human" in tools(runtime)
    assert not retriever.calls and "[" not in result.text


async def test_source_and_evidence_reset_between_turns() -> None:
    llm = ScriptedLLM([reply(claim(_NO_DATES, "FAQ-003")), verdict()])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        first = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "¿Se puede correr el vencimiento de una cuota?",
            context=runtime.context,
        )
        second = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "¿Cuánto debo?",
            context=runtime.context,
        )
    assert first.state["selected_source"] == "rag" and first.state["retrieved"]
    assert second.state["retrieved"] == []
    assert second.state["selected_source"] == "backend"


async def test_quote_that_is_only_a_heading_has_no_verbatim_fallback() -> None:
    # A heading is literal in its section but is a question, not a statement to show the customer.
    heading_quote = claim(_FAMILY_PARAPHRASE, "FAQ-010", "¿Puede pagar un familiar por mí?")
    llm = ScriptedLLM([reply(heading_quote), verdict(supported=(), unsupported=(0,))])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-010")])
    ) as runtime:
        result = await ask(runtime, "¿puede pagar un familiar por mí?")
    assert result.text == _STATIC_TEMPLATES["no_evidence"]
    assert answers(runtime)[-1]["reason"] == "no_verified_answer"


def test_verbatim_helper_refuses_unretrieved_sections_and_empty_quotes() -> None:
    from app.graph.nodes.respond import _verbatim_quotes

    state: Any = {"retrieved": [corpus_chunk("FAQ-003")]}
    assert _verbatim_quotes((claim(_FAMILY, "FAQ-010"),), state) is None
    assert _verbatim_quotes((claim(_NO_DATES, "FAQ-003", " . "),), state) is None
    assert _verbatim_quotes((), state) is None


_OPERATOR = (
    "El pedido lo evalúa un operador y debe hacerse al menos 48 horas antes del vencimiento."
)


async def test_a_claim_about_another_situation_is_dropped_and_the_rest_is_shown() -> None:
    # Live: a correct discount answer also said a partial payment grants no quita, and the whole
    # answer was rejected.
    llm = ScriptedLLM(
        [
            reply(claim(_NO_DATES, "FAQ-003"), claim(_OPERATOR, "FAQ-003")),
            verdict(supported=(0, 1), off_topic=(1,)),
        ]
    )
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        result = await ask(runtime, "¿Se puede correr el vencimiento de una cuota?")
    assert result.text == f"{_NO_DATES} [FAQ-003]"
    trimmed = {"type": "claims_trimmed", "off_topic": 1, "redundant": 0, "kept": 1}
    assert trimmed in runtime.recorder.events
    assert answers(runtime)[-1]["outcome"] == "model_answer"


async def test_answer_check_with_every_claim_off_topic_is_not_an_answer() -> None:
    answer = reply(claim(_NO_DATES, "FAQ-003"))
    # Out-of-range indices are ignored; the only claim is about another situation.
    result = await check_answer(
        "¿Cambian fechas?", answer, ScriptedLLM([verdict(off_topic=(0, 5))])
    )
    assert result == AnswerCheck("not_an_answer")


def test_focused_claims_keep_a_claim_whose_section_was_not_retrieved() -> None:
    # Validation rejects that citation; hiding the claim here would hide the reason.
    from app.graph.nodes.respond import _focused_claims

    unretrieved = (claim(_FAMILY, "FAQ-010"),)
    assert _focused_claims(unretrieved, [corpus_chunk("FAQ-003")]) == list(unretrieved)


async def test_a_claim_citing_a_section_title_cites_that_section() -> None:
    # Live: the model cited "Medios no habilitados" (PAY-MET-003's title) as the section id.
    heading = corpus_chunk("FAQ-003").chunk.heading
    llm = ScriptedLLM([reply(claim(_NO_DATES, heading)), verdict()])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        result = await ask(runtime, "¿Se puede correr el vencimiento de una cuota?")
    assert result.text == f"{_NO_DATES} [FAQ-003]"


class QueryRecordingRetriever(StaticRetriever):
    def __init__(self, hits: Any) -> None:
        super().__init__(hits)
        self.queries: list[str] = []

    async def search_for_generation(
        self, query: str, *, topic: Topic, effective_on: Any, limit: int = 4
    ) -> RetrievalResult:
        self.queries.append(query)
        return await super().search_for_generation(
            query, topic=topic, effective_on=effective_on, limit=limit
        )


def test_retrieval_names_the_knowledge_base_term_for_a_concept_worded_otherwise() -> None:
    from app.graph.ontology import retrieval_query

    expanded = retrieval_query("¿Me harían alguna rebaja si pago todo?")
    assert expanded == "¿Me harían alguna rebaja si pago todo?\nTérminos de la base: quita"
    for unchanged in (
        "¿Me pueden hacer una quita de intereses?",
        "¿Cuánto tarda la transferencia?",
    ):
        assert retrieval_query(unchanged) == unchanged


async def test_policy_generation_retrieves_with_the_knowledge_base_terms() -> None:
    llm = ScriptedLLM([reply(claim(_NO_DATES, "FAQ-003")), verdict()])
    retriever = QueryRecordingRetriever([corpus_chunk("FAQ-003")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        await ask(runtime, "¿Me hacen algún descuento si pago todo junto?")
    assert retriever.queries == [
        "¿Me hacen algún descuento si pago todo junto?\nTérminos de la base: quita"
    ]


async def test_a_claim_repeating_an_earlier_one_is_dropped() -> None:
    # Live chat: a missed installment was explained twice, from FAQ-004 and POL-NEG-008, with 78 %
    # of the terms in common (under the lexical threshold).
    llm = ScriptedLLM(
        [
            reply(claim(_NO_DATES, "FAQ-003"), claim(_OPERATOR, "FAQ-003")),
            verdict(supported=(0, 1), redundant=(1,)),
        ]
    )
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("FAQ-003")])
    ) as runtime:
        result = await ask(runtime, "¿Se puede correr el vencimiento de una cuota?")
    assert result.text == f"{_NO_DATES} [FAQ-003]"
    trimmed = {"type": "claims_trimmed", "off_topic": 0, "redundant": 1, "kept": 1}
    assert trimmed in runtime.recorder.events


async def test_answer_check_never_drops_the_first_claim_as_a_repetition() -> None:
    answer = reply(claim(_NO_DATES, "FAQ-003"))
    result = await check_answer("¿Cambian fechas?", answer, ScriptedLLM([verdict(redundant=(0,))]))
    assert result == AnswerCheck("supported")
