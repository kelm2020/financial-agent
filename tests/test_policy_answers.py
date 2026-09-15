"""Policy answers with a model: verified quotes answer, and nothing verified means abstaining."""

from __future__ import annotations

from typing import Any

from app.graph.nodes.respond import _STATIC_TEMPLATES
from app.graph.recorder import TurnBudgetExceeded
from app.guards.grounding import (
    GroundedReply,
    echoed_terms,
    sentence_supported,
    verify_grounded_reply,
)
from app.llm.protocol import ScriptedLLM
from app.rag.models import RetrievalResult
from tests.agent_support import StaticRetriever, agent_runtime, corpus_chunk, supported_check

_EMPTY = GroundedReply(text="", claims=())


class BelowGateRetriever(StaticRetriever):
    """Max-recall hits whose evidence did not clear the calibrated gate."""

    def _result(self) -> RetrievalResult:
        return super()._result().model_copy(update={"evidence_gate_passed": False})


async def _ask(runtime: Any, text: str) -> Any:
    conversation = await runtime.service.create_conversation("CUST-00125")
    return await runtime.service.send_message(
        conversation.conversation_id, conversation.customer_id, text, context=runtime.context
    )


async def test_model_abstention_above_the_gate_answers_with_the_extract() -> None:
    # Local chat: "¿Puedo cambiar la fecha de vencimiento de una cuota?" (evidence 0.70) got
    # "No encontré esa información" when the model returned an empty reply.
    llm = ScriptedLLM([_EMPTY])
    retriever = StaticRetriever([corpus_chunk("FAQ-003")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await _ask(runtime, "hola ¿Puedo cambiar la fecha de vencimiento de una cuota?")
    assert [call.task for call in llm.calls] == ["grounded_response"]
    assert result.text.endswith("[FAQ-003]") and "48 horas" in result.text
    assert {"type": "policy_model_abstained"} in runtime.recorder.events


async def test_model_abstention_offers_a_person_for_a_low_risk_question() -> None:
    llm = ScriptedLLM([_EMPTY])
    retriever = BelowGateRetriever([corpus_chunk("PAY-MET-001")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await _ask(runtime, "¿puedo pagar con tarjeta?")
    # Max recall: the route's topic sets the risk but never filters the sections searched.
    assert retriever.calls[0] == ("search_for_generation", "any")
    assert [call.task for call in llm.calls] == ["grounded_response"]
    assert result.text == _STATIC_TEMPLATES["no_evidence"]
    assert result.state["offered_next_step"] == "human"
    assert "request_human" not in [call.name for call in runtime.recorder.tool_calls]


async def test_model_abstention_derives_a_high_risk_question() -> None:
    llm = ScriptedLLM([_EMPTY])
    retriever = BelowGateRetriever([corpus_chunk("POL-NEG-003")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await _ask(runtime, "¿Me pueden hacer una quita de intereses?")
    assert "[POL-NEG-003]" not in result.text
    assert "request_human" in [call.name for call in runtime.recorder.tool_calls]


async def test_rejected_answers_below_the_gate_abstain_instead_of_extracting() -> None:
    unsupported = GroundedReply(text="Se puede pagar en 99 cuotas sin recargo.", claims=())
    llm = ScriptedLLM([unsupported, unsupported])
    retriever = BelowGateRetriever([corpus_chunk("PAY-MET-001")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await _ask(runtime, "¿puedo pagar con tarjeta?")
    assert [call.task for call in llm.calls] == ["grounded_response", "grounded_response"]
    assert result.text == _STATIC_TEMPLATES["no_evidence"]
    assert "99" not in result.text and "[PAY-MET-001]" not in result.text


async def test_sentence_restating_the_question_with_a_real_quote_is_rejected() -> None:
    # Live answerability run: "La comisión del asesor es del 10 % del saldo total. [FAQ-001]"
    # passed with a real quote because low-risk claims were not verified.
    sentence = "La comisión del asesor es del 10 % del saldo total."
    fabricated = GroundedReply.model_validate(
        {
            "text": sentence,
            "claims": [
                {
                    "sentence": sentence,
                    "section_id": "FAQ-001",
                    "quote": "desde el 10 % del saldo total",
                }
            ],
        }
    )
    llm = ScriptedLLM([fabricated, fabricated])
    retriever = BelowGateRetriever([corpus_chunk("FAQ-001")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await _ask(runtime, "¿puedo pagarle una comisión al asesor?")
    assert "comisión" not in result.text
    # A fee question is financiero_legal: high risk, so the abstention derives.
    assert result.text == _STATIC_TEMPLATES["no_evidence_high_risk"]
    trimmed = [event for event in runtime.recorder.events if event["type"] == "claims_trimmed"]
    assert trimmed[0]["unsupported"] == 1


def test_echoed_terms_find_question_words_the_source_lacks() -> None:
    source = "Atención de operadores: lunes a viernes de 9 a 18."
    assert echoed_terms(
        "La comisión del asesor es de 9.", "¿cuánto cobra de comisión el asesor?", source
    ) == {"comis", "aseso"}
    assert echoed_terms(
        "Los operadores atienden de lunes a viernes.", "¿qué horario atienden?", source
    ) == {"atien"}


async def test_affirmative_answer_about_a_term_the_source_never_mentions_is_rejected() -> None:
    # Answerability run 2: "¿tienen descuentos especiales para jubilados?" got "Sí, hay descuentos
    # por transferencia o cupón. [FAQ-010]" with one echoed term.
    source = "Un familiar puede pagar por transferencia o cupón."
    assert echoed_terms(
        "Sí, hay descuentos por transferencia o cupón.",
        "¿tienen descuentos especiales para jubilados?",
        source,
    ) == {"descu"}


async def test_regeneration_without_budget_falls_back_to_the_extract() -> None:
    # Answerability run 2: guard, router and a rejected answer used the budget, and the
    # regeneration ended a policy question in "Alcancé el límite seguro de operaciones".
    unsupported = GroundedReply(text="Se puede pagar en 99 cuotas sin recargo.", claims=())
    llm = ScriptedLLM([unsupported, TurnBudgetExceeded("llm", 3)])
    retriever = StaticRetriever([corpus_chunk("PAY-MET-001")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await _ask(runtime, "¿puedo pagar con tarjeta?")
    assert result.text.endswith("[PAY-MET-001]") and "99" not in result.text
    assert {"type": "regeneration_skipped_budget"} in runtime.recorder.events


async def test_a_faithful_answer_may_use_the_words_of_its_section_heading() -> None:
    # "pagar" is only in FAQ-010's heading ("¿Puede pagar un familiar por mí?").
    sentence = "Sí, se puede pagar por transferencia o cupón."
    reply = GroundedReply.model_validate(
        {
            "text": sentence,
            "claims": [
                {
                    "sentence": sentence,
                    "section_id": "FAQ-010",
                    "quote": "Sí, por transferencia o cupón.",
                }
            ],
        }
    )
    from app.rag.support import AnswerSupportDecision

    # A paraphrase is shown only after the semantic check supports it (ADR-011).
    check = AnswerSupportDecision(
        question_asks="quién puede pagar",
        reply_answers="quién puede pagar",
        off_topic_claim_indices=(),
        redundant_claim_indices=(),
        answers_question=True,
        supported_claim_indices=(0,),
        unsupported_claim_indices=(),
        unresolved_aspects=(),
        reason="supported",
    )
    llm = ScriptedLLM([reply, check])
    retriever = BelowGateRetriever([corpus_chunk("FAQ-010")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await _ask(runtime, "¿puede pagar un familiar por mí?")
    assert result.text == f"{sentence} [FAQ-010]"


async def test_restated_claims_are_dropped_from_the_answer() -> None:
    first = "Sí, desde el 10 % del saldo total."
    restated = "Un pago parcial a cuenta se acepta desde el 10 % del saldo total."
    other = (
        "El pago parcial no suspende la gestión de cobranza, no otorga quita y no reemplaza un "
        "acuerdo."
    )
    reply = GroundedReply.model_validate(
        {
            "text": "",
            "claims": [
                {"sentence": first, "section_id": "FAQ-001", "quote": first},
                {"sentence": restated, "section_id": "POL-NEG-006", "quote": restated},
                {"sentence": other, "section_id": "POL-NEG-006", "quote": other},
            ],
        }
    )
    llm = ScriptedLLM([reply, supported_check(2)])
    retriever = StaticRetriever([corpus_chunk("FAQ-001"), corpus_chunk("POL-NEG-006")])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await _ask(runtime, "¿Puedo pagar una parte de la deuda?")
    assert result.text == f"{first} {other} [FAQ-001] [POL-NEG-006]"


def _claims(*items: tuple[str, str, str]) -> GroundedReply:
    return GroundedReply.model_validate(
        {
            "text": "",
            "claims": [
                {"sentence": sentence, "section_id": section, "quote": quote}
                for sentence, section, quote in items
            ],
        }
    )


async def test_answer_keeps_the_answering_section_and_drops_unrelated_ones() -> None:
    # Local chat: "¿Puedo cambiar la fecha de vencimiento de una cuota?" answered FAQ-003 plus the
    # payment-method change (PAY-MET-005) and "…fuera de los límites de este documento…"
    # (POL-NEG-009, a sentence about the policy document). The check marks both as another
    # situation, and only they are dropped (ADR-011).
    first = "El canal automático no cambia fechas."
    second = (
        "El pedido lo evalúa un operador y debe hacerse al menos 48 horas antes del vencimiento."
    )
    unrelated = (
        "El medio de pago de un plan vigente se puede cambiar hasta 48 horas antes del próximo "
        "vencimiento."
    )
    internal = "Cualquier condición fuera de los límites de este documento es una excepción."
    reply = _claims(
        (first, "FAQ-003", first),
        (unrelated, "PAY-MET-005", unrelated),
        (internal, "POL-NEG-009", "Cualquier condición fuera de los límites de este documento"),
        (second, "FAQ-003", second),
    )
    retriever = StaticRetriever(
        [corpus_chunk("FAQ-003"), corpus_chunk("PAY-MET-005"), corpus_chunk("POL-NEG-009")]
    )
    check = supported_check(4).model_copy(update={"off_topic_claim_indices": (1, 2)})
    llm = ScriptedLLM([reply, check])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await _ask(runtime, "¿Puedo cambiar la fecha de vencimiento de una cuota?")
    assert result.text == f"{first} {second} [FAQ-003]"
    trimmed = {"type": "claims_trimmed", "off_topic": 2, "redundant": 0, "kept": 2}
    assert trimmed in runtime.recorder.events


async def test_without_the_check_a_low_risk_answer_keeps_its_best_ranked_section() -> None:
    # Nothing semantic dropped the unrelated claim, so the lexical fallback does.
    family = "Sí, por transferencia o cupón."
    unrelated = (
        "El medio de pago de un plan vigente se puede cambiar hasta 48 horas antes del próximo "
        "vencimiento."
    )
    reply = _claims((family, "FAQ-010", family), (unrelated, "PAY-MET-005", unrelated))
    retriever = StaticRetriever([corpus_chunk("FAQ-010"), corpus_chunk("PAY-MET-005")])
    async with agent_runtime(llm=ScriptedLLM([reply]), retriever=retriever) as runtime:
        runtime.recorder.max_llm_calls = 1
        result = await _ask(runtime, "¿puede pagar un familiar por mí?")
    assert result.text == f"{family} [FAQ-010]"
    assert {"type": "answer_check_skipped_budget"} in runtime.recorder.events


async def test_a_reply_with_only_internal_claims_is_an_abstention() -> None:
    internal = "El agente no aprueba excepciones."
    reply = _claims(
        (internal, "POL-NEG-009", "El agente no la aprueba, no la anticipa como probable")
    )
    retriever = BelowGateRetriever([corpus_chunk("POL-NEG-009")])
    async with agent_runtime(llm=ScriptedLLM([reply]), retriever=retriever) as runtime:
        result = await _ask(runtime, "¿se puede pedir una excepción?")
    assert "agente" not in result.text
    assert {"type": "policy_model_abstained"} in runtime.recorder.events


async def test_extract_speaks_of_the_policies_not_of_the_agent_or_the_document() -> None:
    # POL-NEG-009 keeps its customer-facing rule; "este documento" becomes "estas políticas" and
    # the sentence addressed to the agent is never shown.
    retriever = StaticRetriever([corpus_chunk("POL-NEG-009")])
    async with agent_runtime(retriever=retriever) as runtime:
        result = await _ask(runtime, "¿se puede pedir una excepción?")
    assert "fuera de los límites de estas políticas" in result.text
    assert result.text.endswith("[POL-NEG-009]")
    assert "este documento" not in result.text and "agente" not in result.text


def test_claim_sentence_must_say_what_its_cited_section_says() -> None:
    source = (
        "¿Puedo cambiar la fecha de vencimiento de una cuota?\n"
        "El canal automático no cambia fechas. El pedido lo evalúa un operador y debe hacerse al "
        "menos 48 horas antes del vencimiento."
    )
    borrowed = (
        "Podés cambiar el medio de pago de un plan vigente hasta 48 horas antes del próximo "
        "vencimiento."
    )
    paraphrase = "El canal automático no puede cambiar fechas; el pedido lo evalúa un operador."
    assert not sentence_supported(borrowed, source)
    assert sentence_supported(paraphrase, source)
    assert sentence_supported("Sí, se puede.", source)


async def test_a_claim_saying_what_another_section_says_is_dropped_not_blocked() -> None:
    # Local chat: "Puedes cambiar el medio de pago…" (PAY-MET-005) was shown under a real FAQ-003
    # quote. The claim is dropped and the rest of the answer stands.
    good = "El canal automático no cambia fechas."
    borrowed = (
        "Podés cambiar el medio de pago de un plan vigente hasta 48 horas antes del próximo "
        "vencimiento."
    )
    reply = _claims((good, "FAQ-003", good), (borrowed, "FAQ-003", good))
    retriever = StaticRetriever([corpus_chunk("FAQ-003")])
    llm = ScriptedLLM([reply, supported_check()])
    async with agent_runtime(llm=llm, retriever=retriever) as runtime:
        result = await _ask(runtime, "¿Puedo cambiar la fecha de vencimiento de una cuota?")
    assert result.text == f"{good} [FAQ-003]"
    trimmed = [event for event in runtime.recorder.events if event["type"] == "claims_trimmed"]
    assert trimmed[0]["unsupported"] == 1


def test_verifier_rejects_a_sentence_echoing_the_question() -> None:
    sentence = "La comisión del asesor es del 10 % del saldo total."
    reply = GroundedReply.model_validate(
        {
            "text": sentence,
            "claims": [
                {
                    "sentence": sentence,
                    "section_id": "FAQ-001",
                    "quote": "desde el 10 % del saldo total",
                }
            ],
        }
    )
    sources = {"FAQ-001": "Sí, desde el 10 % del saldo total."}
    assert verify_grounded_reply(reply, sources) == ()
    question = "¿cuánto cobra de comisión el asesor?"
    assert verify_grounded_reply(reply, sources, question=question) == ("unsupported_sentence",)
