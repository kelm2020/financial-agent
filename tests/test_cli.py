from __future__ import annotations

import builtins
import sys
from collections.abc import Iterator
from typing import Any, ClassVar

from app.cli import main, run

STREAM = (
    "event: filler",
    'data: "Dejame revisar la política, un segundo."',
    "event: validated_clause",
    'data: "Primera cláusula."',
    "event: validated_clause",
    'data: "Segunda cláusula."',
    "event: done",
    "data: {}",
)


class FakeResponse:
    def __init__(self, *, status_code: int = 200, payload: dict[str, str] | None = None) -> None:
        self.status_code = status_code
        self.payload = payload or {
            "conversation_id": "conversation-1",
            "message": "Asistente listo",
        }

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return self.payload

    def read(self) -> bytes:
        return b'{"detail":{"code":"CONVERSATION_BUSY"}}'

    def iter_lines(self) -> Iterator[str]:
        yield from STREAM


class FakeStream:
    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def __enter__(self) -> FakeResponse:
        return FakeResponse(status_code=self.status_code)

    def __exit__(self, *args: Any) -> None:
        return None


class FakeClient:
    stream_status: ClassVar[int] = 200
    stream_statuses: ClassVar[list[int]] = []  # consumed first, one per message
    token_requests: ClassVar[list[dict[str, Any]]] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.headers = dict(kwargs.get("headers", {}))

    def __enter__(self) -> FakeClient:
        return self

    def __exit__(self, *args: Any) -> None:
        return None

    def post(self, path: str, **kwargs: Any) -> FakeResponse:
        if path == "/auth/token":
            FakeClient.token_requests.append(kwargs["json"])
            return FakeResponse(payload={"access_token": "issued-token"})
        return FakeResponse()

    def stream(self, *args: Any, **kwargs: Any) -> FakeStream:
        queued = FakeClient.stream_statuses
        return FakeStream(queued.pop(0) if queued else FakeClient.stream_status)


def _typing(monkeypatch: Any, *entries: str) -> None:
    replies = iter(entries)
    monkeypatch.setattr("app.cli.httpx.Client", FakeClient)
    monkeypatch.setattr(FakeClient, "token_requests", [])
    monkeypatch.setattr(builtins, "input", lambda prompt: next(replies))


def test_cli_prints_filler_and_joins_validated_clauses(monkeypatch: Any, capsys: Any) -> None:
    _typing(monkeypatch, "hola", "salir")

    run("http://agent", "token")

    output = capsys.readouterr().out
    assert "Asistente listo" in output
    assert "agente> (Dejame revisar la política, un segundo.)" in output
    assert "agente> Primera cláusula. Segunda cláusula." in output
    assert "{}" not in output and '"' not in output


def test_cli_reports_http_errors_and_keeps_the_session(monkeypatch: Any, capsys: Any) -> None:
    _typing(monkeypatch, "hola", "salir")
    monkeypatch.setattr(FakeClient, "stream_status", 409)

    run("http://agent", "token")

    assert "[error 409]" in capsys.readouterr().out


def test_cli_run_handles_eof(monkeypatch: Any) -> None:
    def eof(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr("app.cli.httpx.Client", FakeClient)
    monkeypatch.setattr(builtins, "input", eof)
    run("http://agent", "token")


def test_cli_entrypoint_requests_a_local_token(monkeypatch: Any) -> None:
    _typing(monkeypatch, "salir")
    monkeypatch.setattr(sys, "argv", ["app.cli", "--customer", "CUST-00212"])

    main()

    assert FakeClient.token_requests == [{"customer_id": "CUST-00212"}]


def test_cli_starts_a_new_conversation_when_the_server_lost_it(
    monkeypatch: Any, capsys: Any
) -> None:
    _typing(monkeypatch, "hola", "salir")
    monkeypatch.setattr(FakeClient, "stream_status", 404)

    run("http://agent", "token")

    output = capsys.readouterr().out
    assert "empiezo una nueva" in output and output.count("Asistente listo") == 2


def test_cli_renews_an_expired_local_token_and_resends(monkeypatch: Any, capsys: Any) -> None:
    # Local chat regression: after five minutes every message failed with 401 INVALID_TOKEN.
    _typing(monkeypatch, "bueno", "salir")
    monkeypatch.setattr(FakeClient, "stream_statuses", [401, 200])
    monkeypatch.setattr(sys, "argv", ["app.cli", "--customer", "CUST-00125"])

    main()

    output = capsys.readouterr().out
    assert len(FakeClient.token_requests) == 2
    assert "[error 401]" not in output and "Primera cláusula." in output


def test_cli_with_a_manual_token_reports_401(monkeypatch: Any, capsys: Any) -> None:
    _typing(monkeypatch, "hola", "salir")
    monkeypatch.setattr(FakeClient, "stream_statuses", [401])

    run("http://agent", "token")

    assert "[error 401]" in capsys.readouterr().out


def test_cli_with_a_manual_token_and_ctrl_c_exits_cleanly(monkeypatch: Any, capsys: Any) -> None:
    def interrupt(prompt: str) -> str:
        raise KeyboardInterrupt

    monkeypatch.setattr("app.cli.httpx.Client", FakeClient)
    monkeypatch.setattr(builtins, "input", interrupt)
    monkeypatch.setattr(sys, "argv", ["app.cli", "--token", "manual-token"])

    main()

    assert "[chat terminado]" in capsys.readouterr().out
    assert FakeClient.token_requests == []  # a manual token is never replaced
