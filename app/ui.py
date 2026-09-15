"""Web chat against the agent API, in Froneus blue.

``make ui`` serves a Gradio page that is one more client of the agent, like ``make chat``: it asks
the local mock backend for a customer's short-lived token (the mock issuer exists only for local
development), opens a conversation and streams the validated SSE clauses into the chat as they
arrive. Nothing here decides what the customer reads; the agent's output frontier already did.
Messages written by this page itself (connection problems, a lost conversation) render as notices,
apart from the agent's words.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import gradio as gr
import httpx

from app.cli import issue_local_token

TIMEOUT_SECONDS = 60

Message = dict[str, Any]
History = list[Message]

# Fixture customers of the mock (mock_api/fixtures/customers.json) and what each one exercises.
CUSTOMERS: dict[str, tuple[str, str]] = {
    "CUST-00125": (
        "María Gómez",
        "Mora media, tres vencimientos impagos. Probá saldo, opciones, un acuerdo con "
        "confirmación, quita y medios de pago.",
    ),
    "CUST-00212": ("Jorge Ferreyra", "Prejudicial. Un pedido de plan se deriva a un operador."),
    "CUST-00377": (
        "Luciana Paz",
        "Mora temprana con identidad sin verificar. Un plan requiere validar identidad con un "
        "asesor.",
    ),
    "CUST-00450": ("Diego Sosa", "Sin deuda vigente. «¿Cuánto debo?» no inventa una deuda."),
}
DEFAULT_CUSTOMER = "CUST-00125"

# The six scenarios of the challenge statement, in the order the README walks through them.
SCRIPT = (
    "¿Cuánto debo?",
    "No puedo pagar todo este mes. ¿Qué opciones tengo?",
    "Quiero la opción de 3 cuotas",
    "Sí",
    "Quiero pagar lo que pueda",
    "¿Quién va a ganar el Mundial?",
    "Quiero hablar con una persona",
)

# Brand tokens published in froneus.com's stylesheet: --brand, --brand-hover and --brand-solid
# for the light theme, and their dark-theme counterparts.
BRAND = "#1565c0"
BRAND_HOVER = "#0d4d9c"
BRAND_DARK = "#4f9cf9"
BRAND_SOLID_DARK = "#4270bb"
BRAND_SOLID_HOVER_DARK = "#3560a8"
FRONEUS_BLUE = gr.themes.Color(
    c50="#eef4fb",
    c100="#d7e6f7",
    c200="#b0ccef",
    c300="#82ade3",
    c400="#4f8fd6",
    c500=BRAND,
    c600=BRAND_HOVER,
    c700="#0b4285",
    c800="#0a376e",
    c900="#082b56",
    c950="#051b37",
    name="froneus",
)

CSS = f"""
.froneus-header {{
  display: flex; align-items: center; gap: 0.9rem; flex-wrap: wrap;
  padding: 0.7rem 1.2rem; border-radius: 12px; color: #fff;
  background: linear-gradient(120deg, {BRAND_HOVER} 0%, {BRAND} 55%, {BRAND_DARK} 130%);
  box-shadow: 0 4px 16px rgba(13, 77, 156, 0.22);
}}
.froneus-mark {{
  display: grid; place-items: center; width: 2.2rem; height: 2.2rem; border-radius: 9px;
  background: rgba(255, 255, 255, 0.16); border: 1px solid rgba(255, 255, 255, 0.35);
  font-weight: 800; font-size: 1.15rem; letter-spacing: -0.02em; color: #fff;
}}
.froneus-header h1 {{ margin: 0; font-size: 1.15rem; font-weight: 700; color: #fff; }}
.froneus-header p {{ margin: 0; font-size: 0.82rem; color: rgba(255, 255, 255, 0.82); }}
.froneus-brand {{ font-size: 0.68rem; letter-spacing: 0.14em; text-transform: uppercase;
  color: rgba(255, 255, 255, 0.75); }}
/* Desktop: the page is exactly one viewport tall. The chat grows into whatever height is left, so
   the input box is always on screen, and the sidebar scrolls on its own when it does not fit. */
@media (min-width: 768px) {{
  .gradio-container {{ height: 100dvh; max-height: 100dvh; }}
  /* Gradio's flex ancestors default to min-height: auto and would grow to the sidebar's content. */
  .gradio-container .app, .gradio-container .app .wrap, .gradio-container main.contain,
  .gradio-container main.contain > .column {{ min-height: 0 !important; }}
  .froneus-main {{ flex: 1 1 0; min-height: 0; flex-wrap: nowrap; }}
  .froneus-sidebar {{ min-height: 0; overflow-y: auto; }}
  .froneus-chat {{ min-height: 0; }}
  .froneus-chat > .froneus-chatbot {{ flex: 1 1 0 !important; height: auto !important; }}
}}
/* At any width: a wrapping column would push the input box into a second, off-screen column. */
.froneus-sidebar, .froneus-chat {{ flex-wrap: nowrap !important; }}
/* Gradio wraps the textbox in a .form with an inline flex-grow; only the chat takes the space. */
.froneus-chat > .form {{ flex: 0 0 auto !important; }}
.froneus-script button {{ justify-content: flex-start; text-align: left; }}
.froneus-script button:hover {{ border-color: {BRAND}; color: {BRAND}; }}
.froneus-input button.submit-button {{ background: {BRAND}; color: #fff; }}
.froneus-input button.submit-button:hover {{ background: {BRAND_HOVER}; }}
.dark .froneus-input button.submit-button {{ background: {BRAND_SOLID_DARK}; }}
.dark .froneus-script button:hover {{ border-color: {BRAND_DARK}; color: {BRAND_DARK}; }}
footer {{ display: none !important; }}
"""

HEADER = """
<div class="froneus-header">
  <div class="froneus-mark" aria-hidden="true">F</div>
  <div>
    <div class="froneus-brand">Froneus</div>
    <h1>Asistente de cobranzas</h1>
    <p>Entorno local de prueba: cada respuesta pasa por la validación del agente.</p>
  </div>
</div>
"""


@dataclass(frozen=True, slots=True)
class Endpoints:
    agent_url: str
    mock_url: str


@dataclass(slots=True)
class ChatSession:
    customer_id: str
    token: str = ""
    conversation_id: str = ""


class TurnRejected(Exception):
    """The agent refused the message before reading it (the status arrived with the headers)."""

    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"{status_code}: {body}")
        self.status_code = status_code
        self.body = body


def froneus_theme() -> gr.themes.Base:
    return gr.themes.Soft(
        primary_hue=FRONEUS_BLUE,
        secondary_hue=FRONEUS_BLUE,
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    ).set(
        body_background_fill="#f6f7f9",
        body_background_fill_dark="#111214",
        button_primary_background_fill=BRAND,
        button_primary_background_fill_hover=BRAND_HOVER,
        button_primary_background_fill_dark=BRAND_SOLID_DARK,
        button_primary_background_fill_hover_dark=BRAND_SOLID_HOVER_DARK,
        button_primary_text_color="#ffffff",
        button_primary_text_color_dark="#ffffff",
        color_accent=BRAND,
        color_accent_soft="*primary_100",
        color_accent_soft_dark="*primary_900",
        link_text_color=BRAND,
        link_text_color_dark=BRAND_DARK,
        input_border_color_focus=BRAND,
        input_border_color_focus_dark=BRAND_DARK,
    )


def describe_customer(customer_id: str) -> str:
    name, scenario = CUSTOMERS[customer_id]
    return f"**{name}** · `{customer_id}`\n\n{scenario}"


def _agent(text: str) -> Message:
    return {"role": "assistant", "content": text}


def _notice(text: str) -> Message:
    return {"role": "assistant", "content": "", "metadata": {"title": f"⚠️ {text}"}}


def _unreachable(endpoints: Endpoints) -> Message:
    return _notice(
        f"No pude conectar con el agente ({endpoints.agent_url}) o con el mock "
        f"({endpoints.mock_url}). Levantalos con make run y make mock y volvé a intentar."
    )


def _turn_messages(filler: str | None, clauses: list[str], *, finished: bool) -> History:
    messages: History = []
    if filler is not None:
        status = "done" if clauses or finished else "pending"
        messages.append(
            {"role": "assistant", "content": "", "metadata": {"title": filler, "status": status}}
        )
    if clauses:
        messages.append(_agent(" ".join(clauses)))
    elif finished:
        messages.append(_notice("El agente terminó el turno sin enviar una respuesta."))
    return messages


def _open(client: httpx.Client, session: ChatSession, endpoints: Endpoints) -> str:
    """Token and conversation for the session; returns the disclosure the agent opens with."""
    session.token = issue_local_token(endpoints.mock_url, session.customer_id)
    client.headers["Authorization"] = f"Bearer {session.token}"
    created = client.post("/conversations", json={"channel": "chat"})
    created.raise_for_status()
    session.conversation_id = str(created.json()["conversation_id"])
    return str(created.json()["message"])


def stream_events(
    client: httpx.Client, conversation_id: str, message: str
) -> Iterator[tuple[str, str]]:
    """Filler and validated clauses as they arrive; an error status raises before any of them."""
    with client.stream(
        "POST", f"/conversations/{conversation_id}/messages", json={"message": message}
    ) as response:
        if response.status_code >= 400:
            raise TurnRejected(
                response.status_code, response.read().decode("utf-8", errors="replace")
            )
        event = ""
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: ") and event in {"filler", "validated_clause"}:
                yield event, str(json.loads(line.removeprefix("data: ")))


def _answer(
    client: httpx.Client, conversation_id: str, message: str, shown: History
) -> Iterator[History]:
    filler: str | None = None
    clauses: list[str] = []
    for event, text in stream_events(client, conversation_id, message):
        if event == "filler":
            filler = text
        else:
            clauses.append(text)
        yield [*shown, *_turn_messages(filler, clauses, finished=False)]
    yield [*shown, *_turn_messages(filler, clauses, finished=True)]


def _rejection(
    rejected: TurnRejected, client: httpx.Client, session: ChatSession, endpoints: Endpoints
) -> History:
    if rejected.status_code == 404:
        # `make run` reloads on code changes and conversations live in memory. The message is not
        # resent: a short reply such as "sí" means nothing in a conversation that never saw it.
        greeting = _open(client, session, endpoints)
        return [
            _notice(
                "La conversación ya no existía en el servidor (¿se reinició el agente?). "
                "Empecé una nueva: repetí tu mensaje."
            ),
            _agent(greeting),
        ]
    if rejected.status_code == 409:
        return [
            _notice("El agente todavía está respondiendo tu mensaje anterior. Esperá y reenvialo.")
        ]
    return [
        _notice(
            f"El agente rechazó el mensaje (error {rejected.status_code}): {rejected.body[:200]}"
        )
    ]


def _exchange(
    client: httpx.Client, message: str, shown: History, session: ChatSession, endpoints: Endpoints
) -> Iterator[History]:
    if not session.conversation_id:
        # The page could not open the conversation on load: the disclosure goes before the message.
        greeting = _open(client, session, endpoints)
        shown = [*shown[:-1], _agent(greeting), shown[-1]]
        yield shown
    client.headers["Authorization"] = f"Bearer {session.token}"
    for attempt in (1, 2):
        try:
            yield from _answer(client, session.conversation_id, message, shown)
            return
        except TurnRejected as rejected:
            if rejected.status_code == 401 and attempt == 1:
                # Local tokens last five minutes. The server rejected the message before reading
                # it, so sending it again with a fresh token cannot duplicate anything.
                session.token = issue_local_token(endpoints.mock_url, session.customer_id)
                client.headers["Authorization"] = f"Bearer {session.token}"
                continue
            yield [*shown, *_rejection(rejected, client, session, endpoints)]
            return


def start_conversation(customer_id: str, *, endpoints: Endpoints) -> tuple[History, ChatSession]:
    session = ChatSession(customer_id=customer_id)
    try:
        with httpx.Client(base_url=endpoints.agent_url, timeout=TIMEOUT_SECONDS) as client:
            return [_agent(_open(client, session, endpoints))], session
    except httpx.HTTPError:
        return [_unreachable(endpoints)], ChatSession(customer_id=customer_id)


def respond(
    message: str,
    history: History,
    session: ChatSession | None,
    customer_id: str,
    *,
    endpoints: Endpoints,
) -> Iterator[tuple[History, ChatSession, str]]:
    """Chat updates, the session and the input box (cleared once the message is taken)."""
    if session is None or session.customer_id != customer_id:
        session = ChatSession(customer_id=customer_id)
    text = message.strip()
    if not text:
        yield history, session, message
        return
    shown: History = [*history, {"role": "user", "content": text}]
    yield shown, session, ""
    try:
        with httpx.Client(base_url=endpoints.agent_url, timeout=TIMEOUT_SECONDS) as client:
            for update in _exchange(client, text, shown, session, endpoints):
                shown = update
                yield shown, session, ""
    except httpx.HTTPError:
        # Whatever the agent already streamed stays on screen; the notice goes after it.
        yield [*shown, _unreachable(endpoints)], session, ""


def build_ui(endpoints: Endpoints) -> gr.Blocks:
    open_chat = functools.partial(start_conversation, endpoints=endpoints)
    send = functools.partial(respond, endpoints=endpoints)
    demo = gr.Blocks(
        title="Froneus · Asistente de cobranzas", analytics_enabled=False, fill_height=True
    )
    with demo:
        gr.HTML(HEADER)
        session = gr.State(None)
        with gr.Row(scale=1, equal_height=True, elem_classes="froneus-main"):
            with gr.Column(scale=1, min_width=260, elem_classes="froneus-sidebar"):
                customer = gr.Dropdown(
                    choices=[(f"{name} · {cid}", cid) for cid, (name, _) in CUSTOMERS.items()],
                    value=DEFAULT_CUSTOMER,
                    label="Cliente de prueba",
                )
                scenario = gr.Markdown(describe_customer(DEFAULT_CUSTOMER))
                restart = gr.Button("Nueva conversación", variant="primary")
                gr.Markdown("**Guion del challenge**")
                with gr.Column(elem_classes="froneus-script"):
                    script = [
                        gr.Button(phrase, size="sm", variant="secondary") for phrase in SCRIPT
                    ]
            with gr.Column(scale=3, min_width=320, elem_classes="froneus-chat"):
                chatbot = gr.Chatbot(
                    label="Conversación",
                    show_label=False,
                    scale=1,
                    height="100%",
                    min_height=320,
                    elem_classes="froneus-chatbot",
                )
                message = gr.Textbox(
                    placeholder="Escribí tu mensaje…",
                    show_label=False,
                    submit_btn="Enviar",
                    autofocus=True,
                    elem_classes="froneus-input",
                )
        demo.load(open_chat, [customer], [chatbot, session])
        customer.change(open_chat, [customer], [chatbot, session])
        customer.change(describe_customer, [customer], [scenario])
        restart.click(open_chat, [customer], [chatbot, session])
        message.submit(send, [message, chatbot, session, customer], [chatbot, session, message])
        for button in script:
            button.click(send, [button, chatbot, session, customer], [chatbot, session, message])
    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Web chat with the collections agent")
    parser.add_argument("--agent-url", default="http://localhost:8000")
    parser.add_argument(
        "--mock-url", default=os.environ.get("MOCK_API_URL", "http://localhost:8001")
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    demo = build_ui(Endpoints(agent_url=args.agent_url, mock_url=args.mock_url))
    demo.queue(default_concurrency_limit=None).launch(
        server_name=args.host, server_port=args.port, theme=froneus_theme(), css=CSS
    )


if __name__ == "__main__":  # pragma: no cover - exercised through `make ui`
    main()
