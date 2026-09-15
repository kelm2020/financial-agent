"""Policy questions go to the knowledge base even when they mention the debt (local chat)."""

from __future__ import annotations

from typing import Any

import pytest

from app.graph.routing import asks_policy, route_turn
from app.guards.grounding import GroundedReply, plain_text, split_sentences
from app.llm.protocol import ScriptedLLM
from tests.agent_support import StaticRetriever, agent_runtime, corpus_chunk


@pytest.mark.parametrize(
    ("text", "intent", "topic"),
    [
        ("¿Puedo pagar una parte de la deuda?", "consulta_general", "any"),
        ("hola ¿Puedo cambiar la fecha de vencimiento de una cuota?", "consulta_general", "any"),
        ("¿Qué pasa si no llego a pagar una cuota?", "consulta_general", "any"),
        ("ya pagué y me sigue apareciendo la deuda", "consulta_general", "any"),
        ("¿puedo pagar con transferencia y cuánto tarda?", "consulta_general", "any"),
        # Answerability run 2026-09-14: these reached the balance, the options or a clarification.
        ("¿aceptan que abone sólo un porcentaje del total?", "consulta_general", "any"),
        ("si arranco un plan y después no llego a pagarlo, ¿qué pasa?", "consulta_general", "any"),
        ("hice el pago ayer y todavía figura la deuda, ¿es normal?", "consulta_general", "any"),
        ("¿me puedo arrepentir de un plan que ya acepté?", "consulta_general", "any"),
        ("¿cómo es el procedimiento para objetar una deuda?", "consulta_general", "any"),
        ("¿qué recargo tiene financiar la deuda?", "consulta_general", "any"),
        ("¿cuántos días después de aceptar vence la primera cuota?", "consulta_general", "any"),
        (
            "¿qué hacen cuando un cliente cuenta que atraviesa una situación delicada?",
            "consulta_general",
            "any",
        ),
        ("¿cuánto de los intereses me pueden perdonar?", "consulta_general", "negociacion"),
        ("perdón, ¿cuánto debo?", "consulta_deuda", "any"),
        # Their own routes are kept.
        ("¿Puedo pagar en 9 cuotas?", "negociacion", "any"),
        ("¿Me pueden hacer una quita de intereses?", "consulta_general", "negociacion"),
        ("¿Por qué me cobran intereses?", "consulta_deuda", "faq"),
        ("¿Puedo saber cuánto debo?", "consulta_deuda", "any"),
        ("¿Puedo invertir en cripto?", "fuera_de_dominio", "any"),
        ("¿Cuándo vencían mis cuotas?", "consulta_deuda", "faq"),
        ("hola quiero pagar", "consulta_deuda", "any"),
        ("quiero pagar con tarjeta", "consulta_general", "medios_pago"),
        ("no puedo pagar todo", "negociacion", "any"),
    ],
)
def test_policy_questions_and_payment_intent_routes(text: str, intent: str, topic: str) -> None:
    route = route_turn(text)
    assert (route.intent, route.topic) == (intent, topic)


def test_policy_question_helper_edges() -> None:
    assert asks_policy("¿puedo hacer un pago parcial?")
    assert not asks_policy("¿puedo pagar en cuotas?")
    assert not asks_policy("¿puedo ver otra opción?")
    assert not asks_policy("¿puedo pagar en 3 cuotas?", installments=3)


async def _ask(runtime: Any, text: str) -> Any:
    conversation = await runtime.service.create_conversation("CUST-00125")
    return await runtime.service.send_message(
        conversation.conversation_id, conversation.customer_id, text, context=runtime.context
    )


async def test_partial_payment_question_is_answered_from_the_faq() -> None:
    # Local chat regression: "¿Puedo pagar una parte de la deuda?" returned the balance.
    async with agent_runtime(retriever=StaticRetriever([corpus_chunk("FAQ-001")])) as runtime:
        result = await _ask(runtime, "¿Puedo pagar una parte de la deuda?")
    assert result.text.endswith("[FAQ-001]") and "10 %" in result.text
    # "(POL-NEG-006)" links sections inside the knowledge base; the customer reads the citation.
    assert "(POL-NEG-006)" not in result.text
    assert [call.name for call in runtime.recorder.tool_calls] == ["search_policies"]


async def test_policy_question_takes_the_risk_of_the_answering_section() -> None:
    source = plain_text(corpus_chunk("POL-NEG-008").chunk.content)
    sentence = next(item for item in split_sentences(source) if len(item.split()) >= 6)
    reply = GroundedReply.model_validate(
        {
            "text": sentence,
            "claims": [{"sentence": sentence, "section_id": "POL-NEG-008", "quote": sentence}],
        }
    )
    llm = ScriptedLLM([reply])
    async with agent_runtime(
        llm=llm, retriever=StaticRetriever([corpus_chunk("POL-NEG-008")])
    ) as runtime:
        result = await _ask(runtime, "¿Qué pasa si no llego a pagar una cuota?")
    assert [call.task for call in llm.calls] == ["grounded_response"]
    assert result.text.endswith("[POL-NEG-008]")
