"""Policy answers with a model: verified quotes answer, and nothing verified means abstaining."""

from __future__ import annotations

from typing import Any

from app.graph.nodes.respond import _STATIC_TEMPLATES
from app.graph.recorder import TurnBudgetExceeded
from app.guards.grounding import GroundedReply, echoed_terms
from app.llm.protocol import ScriptedLLM
from app.rag.models import RetrievalResult
from tests.agent_support import StaticRetriever, agent_runtime, corpus_chunk

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


async def test_model_abstention_offers_a_person_for_a_low_risk_question() -> None:
    llm = ScriptedLLM([_EMPTY])
    retriever = StaticRetriever([corpus_chunk("PAY-MET-001")])
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
    retriever = StaticRetriever([corpus_chunk("POL-NEG-003")])
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
    assert result.text == _STATIC_TEMPLATES["no_evidence"]
    assert "unsupported_sentence" in result.state["guard_flags"]


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
