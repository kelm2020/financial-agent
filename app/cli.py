"""Terminal chat against the agent API.

``make chat`` asks the local mock backend for a short-lived token of one customer (the mock issuer
exists only for local development), renews it when it expires and streams the validated SSE
clauses as they arrive. A terminal chat is sequential, so the client is synchronous: reading input
never blocks an event loop and Ctrl+C stops the chat at once.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
from collections.abc import Callable

import httpx


def issue_local_token(mock_url: str, customer_id: str) -> str:
    with httpx.Client(base_url=mock_url, timeout=10) as client:
        response = client.post("/auth/token", json={"customer_id": customer_id})
        response.raise_for_status()
        return str(response.json()["access_token"])


def _start_conversation(client: httpx.Client) -> str:
    created = client.post("/conversations", json={"channel": "chat"})
    created.raise_for_status()
    print(created.json()["message"])
    return str(created.json()["conversation_id"])


def _send(client: httpx.Client, conversation_id: str, message: str) -> tuple[int, str]:
    """Stream one answer. An error status is returned, not printed, so the caller can recover."""
    with client.stream(
        "POST", f"/conversations/{conversation_id}/messages", json={"message": message}
    ) as response:
        if response.status_code >= 400:
            return response.status_code, response.read().decode("utf-8", errors="replace")
        event = ""
        answered = False
        for line in response.iter_lines():
            if line.startswith("event: "):
                event = line.removeprefix("event: ")
            elif line.startswith("data: ") and event in {"filler", "validated_clause"}:
                text = json.loads(line.removeprefix("data: "))
                if event == "filler":
                    print(f"agente> ({text})")
                    continue
                print("agente> " if not answered else "", end="", flush=True)
                print(text, end=" ", flush=True)
                answered = True
        print()
        return response.status_code, ""


def run(base_url: str, token: str, *, refresh_token: Callable[[], str] | None = None) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    with httpx.Client(base_url=base_url, headers=headers, timeout=60) as client:
        conversation_id = _start_conversation(client)
        while True:
            try:
                message = input("vos> ").strip()
            except EOFError:
                break
            if not message or message.casefold() in {"salir", "exit", "quit"}:
                break
            status, body = _send(client, conversation_id, message)
            if status == 401 and refresh_token is not None:
                # Local tokens last five minutes. The server rejected the message before reading
                # it, so sending it again with a fresh token cannot duplicate anything.
                client.headers["Authorization"] = f"Bearer {refresh_token()}"
                status, body = _send(client, conversation_id, message)
            if status == 404:
                # `make run` reloads on code changes and conversations live in memory.
                print("[la conversación ya no existe en el servidor: empiezo una nueva]")
                conversation_id = _start_conversation(client)
            elif status >= 400:
                print(f"[error {status}] {body}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the collections agent")
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--token", help="Bearer token; omit it to request one from the mock")
    parser.add_argument("--customer", default="CUST-00125")
    parser.add_argument(
        "--mock-url", default=os.environ.get("MOCK_API_URL", "http://localhost:8001")
    )
    args = parser.parse_args()
    refresh = (
        None if args.token else functools.partial(issue_local_token, args.mock_url, args.customer)
    )
    token = args.token or issue_local_token(args.mock_url, args.customer)
    try:
        run(args.base_url, token, refresh_token=refresh)
    except KeyboardInterrupt:
        print("\n[chat terminado]")


if __name__ == "__main__":  # pragma: no cover - exercised by the installed console entrypoint
    main()
