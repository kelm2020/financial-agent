from __future__ import annotations

import json
import sys
from collections.abc import Iterator
from typing import Any

import gradio as gr
import httpx
import pytest

from app import ui
from app.ui import ChatSession, Endpoints, respond, start_conversation

ENDPOINTS = Endpoints(agent_url="http://agent", mock_url="http://mock")
DISCLOSURE = "Soy un asistente virtual de cobranzas."
STREAM = (
    "event: filler\n"
    'data: "Dejame revisar la política, un segundo."\n\n'
    "event: validated_clause\n"
    'data: "Primera cláusula."\n\n'
    "event: validated_clause\n"
    'data: "Segunda cláusula."\n\n'
    "event: done\n"
    "data: {}\n\n"
)


class FakeBackend:
    """Mock issuer and agent API behind an in-process transport."""

    def __init__(self) -> None:
        self.tokens = 0
        self.conversations = 0
        self.sent: list[tuple[str, str, str]] = []
        self.statuses: list[int] = []
        self.stream: bytes | Iterator[bytes] = STREAM.encode()
        self.down = False

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.down:
            raise httpx.ConnectError("connection refused", request=request)
        if request.url.path == "/auth/token":
            self.tokens += 1
            return httpx.Response(200, json={"access_token": f"token-{self.tokens}"})
        if request.url.path == "/conversations":
            self.conversations += 1
            conversation = {"conversation_id": f"c-{self.conversations}", "message": DISCLOSURE}
            return httpx.Response(201, json=conversation)
        self.sent.append(
            (
                request.url.path,
                request.headers["Authorization"],
                json.loads(request.content)["message"],
            )
        )
        status = self.statuses.pop(0) if self.statuses else 200
        if status >= 400:
            return httpx.Response(status, json={"detail": {"code": "REJECTED"}})
        return httpx.Response(200, content=self.stream)


@pytest.fixture
def backend(monkeypatch: pytest.MonkeyPatch) -> FakeBackend:
    fake = FakeBackend()
    real_client = httpx.Client

    def client(**kwargs: Any) -> httpx.Client:
        return real_client(transport=httpx.MockTransport(fake), **kwargs)

    monkeypatch.setattr(httpx, "Client", client)
    return fake


def _opened(backend: FakeBackend) -> tuple[list[dict[str, Any]], ChatSession]:
    return start_conversation("CUST-00125", endpoints=ENDPOINTS)


def _last(updates: Iterator[tuple[list[dict[str, Any]], ChatSession, str]]) -> Any:
    collected = list(updates)
    return collected, collected[-1][0], collected[-1][1]


def test_start_conversation_shows_the_disclosure_and_keeps_the_session(
    backend: FakeBackend,
) -> None:
    history, session = _opened(backend)

    assert history == [{"role": "assistant", "content": DISCLOSURE}]
    assert session == ChatSession("CUST-00125", token="token-1", conversation_id="c-1")


def test_start_conversation_reports_a_backend_that_is_not_running(backend: FakeBackend) -> None:
    backend.down = True

    history, session = _opened(backend)

    assert "make run" in history[0]["metadata"]["title"] and history[0]["content"] == ""
    assert session == ChatSession("CUST-00125")


def test_respond_streams_the_filler_then_the_validated_clauses(backend: FakeBackend) -> None:
    history, session = _opened(backend)

    updates, final, _ = _last(
        respond("  hola  ", history, session, "CUST-00125", endpoints=ENDPOINTS)
    )

    assert updates[0][0][-1] == {"role": "user", "content": "hola"}
    assert all(box == "" for _, _, box in updates)
    assert updates[1][0][-1]["metadata"]["status"] == "pending"
    assert final[2]["metadata"] == {
        "title": "Dejame revisar la política, un segundo.",
        "status": "done",
    }
    assert final[3] == {"role": "assistant", "content": "Primera cláusula. Segunda cláusula."}
    assert len(final) == 4
    assert backend.sent == [("/conversations/c-1/messages", "Bearer token-1", "hola")]


def test_respond_ignores_a_blank_message(backend: FakeBackend) -> None:
    history, session = _opened(backend)

    assert list(respond("   ", history, session, "CUST-00125", endpoints=ENDPOINTS)) == [
        (history, session, "   ")
    ]
    assert backend.sent == []


def test_respond_opens_the_conversation_when_the_page_could_not(backend: FakeBackend) -> None:
    stale = ChatSession("CUST-00125", token="token-0", conversation_id="c-0")

    _, history, session = _last(
        respond("¿Cuánto debo?", [], stale, "CUST-00212", endpoints=ENDPOINTS)
    )

    assert session == ChatSession("CUST-00212", token="token-1", conversation_id="c-1")
    assert history[0] == {"role": "assistant", "content": DISCLOSURE}
    assert history[1] == {"role": "user", "content": "¿Cuánto debo?"}
    assert history[-1]["content"] == "Primera cláusula. Segunda cláusula."
    assert backend.sent[0][2] == "¿Cuánto debo?"


def test_respond_renews_an_expired_token_and_resends_once(backend: FakeBackend) -> None:
    history, session = _opened(backend)
    backend.statuses = [401]

    _, final, session = _last(respond("bueno", history, session, "CUST-00125", endpoints=ENDPOINTS))

    assert [auth for _, auth, _ in backend.sent] == ["Bearer token-1", "Bearer token-2"]
    assert session.token == "token-2"
    assert final[-1]["content"] == "Primera cláusula. Segunda cláusula."


def test_respond_starts_a_new_conversation_without_resending(backend: FakeBackend) -> None:
    history, session = _opened(backend)
    backend.statuses = [404]

    _, final, session = _last(respond("sí", history, session, "CUST-00125", endpoints=ENDPOINTS))

    assert len(backend.sent) == 1 and session.conversation_id == "c-2"
    assert "repetí tu mensaje" in final[-2]["metadata"]["title"]
    assert final[-1] == {"role": "assistant", "content": DISCLOSURE}


@pytest.mark.parametrize(
    ("statuses", "expected"),
    [([409], "todavía está respondiendo"), ([401, 401], "error 401"), ([422], "error 422")],
)
def test_respond_explains_a_rejected_message(
    backend: FakeBackend, statuses: list[int], expected: str
) -> None:
    history, session = _opened(backend)
    backend.statuses = statuses

    _, final, _ = _last(respond("hola", history, session, "CUST-00125", endpoints=ENDPOINTS))

    assert expected in final[-1]["metadata"]["title"]
    assert final[-2] == {"role": "user", "content": "hola"}


def test_respond_keeps_what_arrived_when_the_stream_breaks(backend: FakeBackend) -> None:
    history, session = _opened(backend)

    def broken() -> Iterator[bytes]:
        yield 'event: validated_clause\ndata: "Primera cláusula."\n\n'.encode()
        raise httpx.ReadError("connection reset")

    backend.stream = broken()

    _, final, _ = _last(respond("hola", history, session, "CUST-00125", endpoints=ENDPOINTS))

    assert final[-2] == {"role": "assistant", "content": "Primera cláusula."}
    assert "make run" in final[-1]["metadata"]["title"]


def test_respond_flags_a_turn_without_an_answer(backend: FakeBackend) -> None:
    history, session = _opened(backend)
    backend.stream = b"event: done\ndata: {}\n\n"

    _, final, _ = _last(respond("hola", history, session, "CUST-00125", endpoints=ENDPOINTS))

    assert "sin enviar una respuesta" in final[-1]["metadata"]["title"]


def test_every_fixture_customer_has_a_scenario() -> None:
    fixtures = json.loads(open("mock_api/fixtures/customers.json", encoding="utf-8").read())

    assert {customer["customer_id"] for customer in fixtures} == set(ui.CUSTOMERS)
    assert "Prejudicial" in ui.describe_customer("CUST-00212")


def test_entrypoint_launches_the_themed_page(monkeypatch: pytest.MonkeyPatch) -> None:
    launched: dict[str, Any] = {}

    def launch(self: gr.Blocks, **kwargs: Any) -> None:
        launched.update(kwargs)

    monkeypatch.setattr(gr.Blocks, "launch", launch)
    monkeypatch.setattr(sys, "argv", ["app.ui", "--port", "7861", "--agent-url", "http://a:1"])

    ui.main()

    assert launched["server_port"] == 7861 and launched["server_name"] == "127.0.0.1"
    assert isinstance(launched["theme"], gr.themes.Base) and ui.BRAND in launched["css"]
