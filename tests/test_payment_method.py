"""The customer chooses how to pay, and a change of method is confirmed again (INV-4)."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from decimal import Decimal
from typing import Any

import pytest

from app.graph.nodes.respond import _STATIC_TEMPLATES
from app.graph.routing import requested_payment_method
from tests.agent_support import BackendFault, agent_runtime, fixture_draft

_ALLOWED = "podés pagar por débito automático, transferencia y cupón de pago."


@pytest.mark.parametrize(
    ("text", "method"),
    [
        ("Prefiero pagar por transferencia", "transferencia"),
        ("prefiero transferencia", "transferencia"),
        ("sí, pero por transferencia", "transferencia"),
        ("Quiero la opción de 3 cuotas por débito automático", "debito_automatico"),
        ("No quiero pagar con débito, prefiero en efectivo", "cupon"),
        ("¿Y si pago con tarjeta cambia algo?", None),
        ("Contame si la de 3 cuotas cambia con tarjeta.", None),
        ("No puedo pagar en efectivo", None),
        ("sí, confirmo", None),
        ("Quiero aceptar la opción de pago que me ofreciste.", None),
    ],
)
def test_requested_payment_method(text: str, method: str | None) -> None:
    assert requested_payment_method(text) == method


async def _chat(runtime: Any) -> Callable[[str], Awaitable[Any]]:
    conversation = await runtime.service.create_conversation("CUST-00125")

    async def say(text: str) -> Any:
        return await runtime.service.send_message(
            conversation.conversation_id, conversation.customer_id, text, context=runtime.context
        )

    return say


def _written_methods(runtime: Any) -> list[str]:
    return [
        call.arguments["medio_pago"]
        for call in runtime.recorder.tool_calls
        if call.name == "create_payment_agreement"
    ]


async def test_the_method_named_with_the_choice_is_the_one_confirmed_and_written() -> None:
    async with agent_runtime() as runtime:
        say = await _chat(runtime)
        summary = await say("Quiero la opción de 3 cuotas por transferencia")
        assert "por transferencia. ¿Confirmás este acuerdo?" in summary.text
        await say("sí")
        assert _written_methods(runtime) == ["transferencia"]


async def test_a_new_method_while_confirming_freezes_a_new_draft_and_asks_again() -> None:
    async with agent_runtime() as runtime:
        say = await _chat(runtime)
        first = await say("Quiero la opción de 3 cuotas")
        changed = await say("sí, pero por transferencia")
        assert "por transferencia. ¿Confirmás este acuerdo?" in changed.text
        assert changed.state["pending_draft"].draft_id != first.state["pending_draft"].draft_id
        assert not _written_methods(runtime)
        await say("sí")
        assert _written_methods(runtime) == ["transferencia"]


async def test_a_method_the_option_does_not_admit_is_explained_and_changes_nothing() -> None:
    async with agent_runtime() as runtime:
        say = await _chat(runtime)
        chosen = await say("Quiero la opción de 3 cuotas con tarjeta")
        assert chosen.text.startswith(
            f"El pago con tarjeta no está habilitado para esta opción; {_ALLOWED}"
        )
        assert "por débito automático. ¿Confirmás este acuerdo?" in chosen.text
        refused = await say("Prefiero pagar con tarjeta")
        assert refused.text.startswith("El pago con tarjeta no está habilitado")
        assert refused.state["pending_draft"].draft_id == chosen.state["pending_draft"].draft_id
        assert not _written_methods(runtime)


async def test_a_method_change_revalidates_the_offer_first() -> None:
    async with agent_runtime() as runtime:
        stale = await fixture_draft(runtime, monto_total=Decimal("1"))
        valid = await fixture_draft(runtime)
        conversation = await runtime.service.create_conversation("CUST-00125")
        await runtime.seed(conversation, {"pending_draft": stale})
        invalidated = await runtime.service.send_message(
            conversation.conversation_id,
            "CUST-00125",
            "prefiero transferencia",
            context=runtime.context,
        )
        assert invalidated.state["pending_draft"] is None and not _written_methods(runtime)
    async with agent_runtime(faults=[BackendFault("GET", "/debt/", "timeout")]) as runtime:
        conversation = await runtime.service.create_conversation("CUST-00125")
        await runtime.seed(conversation, {"pending_draft": valid})
        unavailable = await runtime.service.send_message(
            conversation.conversation_id,
            "CUST-00125",
            "prefiero transferencia",
            context=runtime.context,
        )
    assert unavailable.text == _STATIC_TEMPLATES["data_unavailable"]
    assert unavailable.state["pending_draft"] == valid
