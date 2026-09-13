from __future__ import annotations

import argparse
import asyncio

import httpx


async def run(base_url: str, token: str) -> None:
    headers = {"Authorization": f"Bearer {token}"}
    async with httpx.AsyncClient(base_url=base_url, headers=headers, timeout=30) as client:
        created = await client.post("/conversations", json={"channel": "chat"})
        created.raise_for_status()
        conversation_id = created.json()["conversation_id"]
        print(created.json()["message"])
        while True:
            try:
                message = (await asyncio.to_thread(input, "vos> ")).strip()
            except EOFError:
                break
            if not message or message.casefold() in {"salir", "exit", "quit"}:
                break
            async with client.stream(
                "POST",
                f"/conversations/{conversation_id}/messages",
                json={"message": message},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line.startswith("data: ") and line != "data: {}":
                        print(f"agente> {line.removeprefix('data: ')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--token", required=True)
    args = parser.parse_args()
    asyncio.run(run(args.base_url, args.token))


if __name__ == "__main__":  # pragma: no cover - exercised by the installed console entrypoint
    main()
