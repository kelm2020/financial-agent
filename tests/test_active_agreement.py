"""An active agreement is reported, never offered again, in this and later conversations."""

from __future__ import annotations

import re
from typing import Any

from tests.agent_support import BackendFault, agent_runtime


async def test_active_agreement_is_reported_instead_of_new_plans() -> None:
    # Local chat regression: after registering a plan the balance still offered alternatives, and
    # a new conversation listed options until the backend answered AGREEMENT_EXISTS.
    async with agent_runtime() as runtime:

        async def say(conversation: Any, text: str) -> Any:
            return await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )

        first = await runtime.service.create_conversation("CUST-00125")
        await say(first, "Quiero la opción de 3 cuotas")
        registered = await say(first, "sí")
        balance = await say(first, "¿cuánto debo?")
        options = await say(first, "¿qué opciones tengo?")

        second = await runtime.service.create_conversation("CUST-00125")
        later_balance = await say(second, "hola, ¿cuánto debo?")
        later_options = await say(second, "¿qué opciones tengo?")
        choice = await say(second, "Quiero la opción de 6 cuotas")
        handoff = await say(second, "sí")

    match = re.search(r"AGR-[0-9A-F]+", registered.text)
    assert match is not None
    number = match.group()
    assert f"compromiso N° {number}" in balance.text and "alternativas" not in balance.text
    assert options.text.startswith("Ya tenés un acuerdo activo") and number in options.text
    assert "acuerdo de pago activo" in later_balance.text
    assert "alternativas" not in later_balance.text and "N°" not in later_balance.text
    assert later_options.text.startswith("Ya tenés un acuerdo activo")
    assert choice.text.startswith("Ya tenés un acuerdo activo") and "Confirmás" not in choice.text
    assert "derivé" in handoff.text
    writes = [
        request for request in runtime.transport.requests if request[1] == "/payment-agreement"
    ]
    assert len(writes) == 1


# ------------------------------------------------------- escalation blocks negotiation


async def test_derivation_for_missing_evidence_kills_the_pending_draft() -> None:
    # Camino A (audit): the high-risk question without evidence derives, and the frozen draft
    # must not survive it. A later bare "sí" answers nothing.
    async with agent_runtime(retriever=None) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        for text in ("Quiero la opción de 3 cuotas", "¿Qué quita existe?", "sí"):
            turn = await runtime.service.send_message(
                conversation.conversation_id,
                conversation.customer_id,
                text,
                context=runtime.context,
            )
        assert turn.state["handoff_motivo"] == "fuera_de_politica"
        assert turn.state["pending_draft"] is None
        assert runtime.recorder.agreement_writes == []
        assert any(
            event["type"] == "agreement_draft_cancelled" or not runtime.recorder.agreement_writes
            for event in runtime.recorder.events
        )


async def test_failed_transfer_keeps_the_negotiation_blocked() -> None:
    # Camino B (audit): the vulnerability escalates, the transfer POST fails with 500, and the
    # block survives the failure: no new draft, no confirmation, no write on a later "sí".
    faults = (BackendFault(method="POST", path_prefix="/transfer", mode="500"),)
    async with agent_runtime(faults=faults) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        vulnerable = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "Me quedé sin trabajo",
            context=runtime.context,
        )
        assert "no pude completar la derivación" in vulnerable.text
        assert vulnerable.state["handoff_motivo"] == "vulnerabilidad"
        offered = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "Quiero la opción de 3 cuotas",
            context=runtime.context,
        )
        assert "asesor" in offered.text
        assert "confirmá" not in offered.text
        confirmed = await runtime.service.send_message(
            conversation.conversation_id,
            conversation.customer_id,
            "sí",
            context=runtime.context,
        )
        assert runtime.recorder.agreement_writes == []
        assert confirmed.state["handoff_motivo"] == "vulnerabilidad"
