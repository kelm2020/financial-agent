from __future__ import annotations

import json

import httpx
import pytest

from app.graph.state import GeneratedReply
from app.guards.injection import GuardModelResult
from app.llm.openai_responses import (
    OpenAIIncompleteError,
    OpenAIRefusalError,
    OpenAIResponseError,
    OpenAIResponsesLLM,
    _strict_json_schema,
)


async def test_openai_responses_adapter_requests_strict_schema_and_parses_output() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"text":"respuesta segura"}'}],
                    }
                ]
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.test/v1"
    )
    adapter = OpenAIResponsesLLM(api_key="sk-test", model="model-test", client=client)
    try:
        result = await adapter.complete(
            task="response",
            messages=(
                {"role": "system", "content": "reglas"},
                {"role": "user", "content": "consulta"},
            ),
            response_model=GeneratedReply,
        )
        await adapter.aclose()
    finally:
        await client.aclose()

    assert result.text == "respuesta segura"
    assert captured["instructions"] == "reglas"
    assert captured["store"] is False
    assert captured["text"]["format"]["type"] == "json_schema"  # type: ignore[index]
    assert captured["text"]["format"]["strict"] is True  # type: ignore[index]
    schema = captured["text"]["format"]["schema"]  # type: ignore[index]
    assert schema["required"] == ["text"]
    assert schema["additionalProperties"] is False


async def test_openai_responses_adapter_requires_key_and_output_text() -> None:
    with pytest.raises(ValueError, match="key"):
        OpenAIResponsesLLM(api_key=" ", model="test")

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"output": []})),
        base_url="https://api.openai.test/v1",
    )
    adapter = OpenAIResponsesLLM(api_key="sk-test", model="model-test", client=client)
    try:
        with pytest.raises(ValueError, match="structured output"):
            await adapter.complete(
                task="response",
                messages=({"role": "user", "content": "consulta"},),
                response_model=GeneratedReply,
            )
    finally:
        await client.aclose()


async def test_openai_responses_adapter_closes_owned_client() -> None:
    adapter = OpenAIResponsesLLM(api_key="sk-test", model="model-test")
    await adapter.aclose()
    assert adapter._client.is_closed


@pytest.mark.parametrize(
    ("payload", "error"),
    [
        (
            {"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"}},
            OpenAIIncompleteError,
        ),
        (
            {
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}],
            },
            OpenAIRefusalError,
        ),
        ({"status": "failed", "error": {"message": "provider failure"}}, OpenAIResponseError),
        ({"status": "cancelled"}, OpenAIResponseError),
    ],
)
async def test_openai_responses_adapter_handles_terminal_states(
    payload: dict[str, object], error: type[OpenAIResponseError]
) -> None:
    client = httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)),
        base_url="https://api.openai.test/v1",
    )
    adapter = OpenAIResponsesLLM(api_key="sk-test", model="model-test", client=client)
    try:
        with pytest.raises(error):
            await adapter.complete(
                task="response",
                messages=({"role": "user", "content": "consulta"},),
                response_model=GeneratedReply,
            )
    finally:
        await client.aclose()


def test_model_facing_guard_schema_avoids_unsupported_numeric_constraints() -> None:
    confidence = GuardModelResult.model_json_schema()["properties"]["confidence"]
    assert "minimum" not in confidence
    assert "maximum" not in confidence
    with pytest.raises(ValueError, match="confidence"):
        GuardModelResult(label="benign", confidence=2)


def test_strict_schema_recurses_and_drops_defaults() -> None:
    schema = _strict_json_schema(
        {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"value": {"type": "string", "default": "x"}},
                    },
                }
            },
        }
    )
    assert schema["required"] == ["items"]
    nested = schema["properties"]["items"]["items"]
    assert nested["required"] == ["value"]
    assert "default" not in nested["properties"]["value"]
