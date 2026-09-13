from __future__ import annotations

import builtins
import sys
from collections.abc import AsyncIterator
from typing import Any

from app.cli import main, run


class FakeResponse:
    def __init__(self, *, created: bool = False) -> None:
        self.created = created

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, str]:
        return {
            "conversation_id": "conversation-1",
            "message": "Asistente listo",
        }

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in ("event: validated_clause", 'data: "Respuesta"', "data: {}"):
            yield line


class FakeStream:
    async def __aenter__(self) -> FakeResponse:
        return FakeResponse()

    async def __aexit__(self, *args: Any) -> None:
        return None


class FakeAsyncClient:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args
        self.kwargs = kwargs

    async def __aenter__(self) -> FakeAsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        return None

    async def post(self, *args: Any, **kwargs: Any) -> FakeResponse:
        return FakeResponse(created=True)

    def stream(self, *args: Any, **kwargs: Any) -> FakeStream:
        return FakeStream()


async def test_cli_run_streams_only_data_and_stops(monkeypatch: Any, capsys: Any) -> None:
    entries = iter(("hola", "salir"))
    monkeypatch.setattr("app.cli.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(builtins, "input", lambda prompt: next(entries))

    await run("http://agent", "token")

    output = capsys.readouterr().out
    assert "Asistente listo" in output
    assert 'agente> "Respuesta"' in output
    assert "data: {}" not in output


async def test_cli_run_handles_eof(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.cli.httpx.AsyncClient", FakeAsyncClient)

    def eof(prompt: str) -> str:
        raise EOFError

    monkeypatch.setattr(builtins, "input", eof)
    await run("http://agent", "token")


def test_cli_entrypoint(monkeypatch: Any) -> None:
    monkeypatch.setattr("app.cli.httpx.AsyncClient", FakeAsyncClient)
    monkeypatch.setattr(builtins, "input", lambda prompt: "salir")
    monkeypatch.setattr(sys, "argv", ["app.cli", "--base-url", "http://agent", "--token", "token"])

    main()
