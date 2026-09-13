"""Level-A hermeticity (§0.11): no developer .env, no provider keys, no outbound network."""

from __future__ import annotations

import os
from collections.abc import Iterator

import httpx
import pytest

from config.settings import Settings

# Settings must never read the developer's .env inside the suite: a local key would silently
# turn level-A tests into paid, networked calls and make coverage depend on secrets.
Settings.model_config["env_file"] = None
for _key in ("OPENAI_API_KEY", "COHERE_API_KEY", "SYSTEM_PROMPT_CANARY", "APP_ENV"):
    os.environ.pop(_key, None)


@pytest.fixture(autouse=True)
def no_outbound_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """In-process transports (ASGI, Mock) still work; a real socket transport fails loudly."""

    async def blocked_async(
        self: httpx.AsyncHTTPTransport, request: httpx.Request
    ) -> httpx.Response:
        raise RuntimeError(f"level-A tests must not reach the network ({request.url.host})")

    def blocked(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise RuntimeError(f"level-A tests must not reach the network ({request.url.host})")

    monkeypatch.setattr(httpx.AsyncHTTPTransport, "handle_async_request", blocked_async)
    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", blocked)
    yield
