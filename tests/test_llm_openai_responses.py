from __future__ import annotations

import asyncio
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
    build_agent_llm,
    collect_usage,
)


async def test_openai_responses_adapter_requests_strict_schema_and_parses_output() -> None:
    captured: dict[str, object] = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 15,
                    "input_tokens_details": {"cached_tokens": 80},
                },
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"text":"respuesta segura"}'}],
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.test/v1"
    )
    adapter = OpenAIResponsesLLM(
        api_key="sk-test", model="model-test", reasoning_effort="low", client=client
    )
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
    assert adapter.usage_records[0].input_tokens == 120
    assert adapter.usage_records[0].output_tokens == 15
    assert adapter.usage_records[0].cached_tokens == 80
    assert captured["instructions"] == "reglas"
    assert captured["reasoning"] == {"effort": "low"}
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
    assert adapter.usage_records == ()


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


async def test_a_task_model_serves_its_task_and_is_recorded_in_usage() -> None:
    models: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        models.append(json.loads(request.content)["model"])
        return httpx.Response(
            200,
            json={
                "usage": {"input_tokens": 1, "output_tokens": 1},
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"text":"ok"}'}],
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.test/v1"
    )
    adapter = OpenAIResponsesLLM(
        api_key="sk-test",
        model="agent-model",
        client=client,
        task_models={"policy_answer_check": "check-model"},
    )
    try:
        for task in ("grounded_response", "policy_answer_check"):
            await adapter.complete(
                task=task,
                messages=({"role": "user", "content": "consulta"},),
                response_model=GeneratedReply,
            )
    finally:
        await client.aclose()
    assert models == ["agent-model", "check-model"]
    assert [record.model for record in adapter.usage_records] == models
    assert adapter.model == "agent-model"


async def test_every_agent_entry_point_pairs_the_turn_model_with_the_check_model() -> None:
    adapter = build_agent_llm(api_key="sk-test", model="gpt-5-nano", check_model="gpt-5-mini")
    try:
        assert adapter.model == "gpt-5-nano"
        assert adapter._task_models == {"policy_answer_check": "gpt-5-mini"}
        assert adapter._reasoning == {"effort": "low"}
    finally:
        await adapter.aclose()


async def test_usage_is_attributed_per_scope_when_calls_overlap() -> None:
    """Each scope is billed only for its own calls, even while other scopes are mid-flight.

    The evaluation runner used to attribute tokens by slicing the client's shared record list from
    an index taken before the case started. That is only correct while exactly one case runs at a
    time: with cases in parallel the slice picks up whatever the others appended. A context-local
    sink is what makes the cost per case survive concurrency.
    """
    started = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        tokens = int(payload["instructions"])
        # Force overlap: the first call in flight only finishes once another has begun.
        if tokens == 1:
            started.set()
        else:
            await started.wait()
        return httpx.Response(
            200,
            json={
                "usage": {"input_tokens": tokens, "output_tokens": tokens},
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"text":"ok"}'}],
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.test/v1"
    )
    adapter = OpenAIResponsesLLM(api_key="sk-test", model="model-test", client=client)

    async def scoped(tokens: int) -> list[int]:
        with collect_usage() as sink:
            await adapter.complete(
                task="response",
                messages=({"role": "system", "content": str(tokens)},),
                response_model=GeneratedReply,
            )
            return [record.input_tokens for record in sink]

    try:
        async with asyncio.TaskGroup() as group:
            first = group.create_task(scoped(1))
            second = group.create_task(scoped(2))
    finally:
        await client.aclose()

    # Neither scope saw the other's call, and the client still holds both.
    assert first.result() == [1]
    assert second.result() == [2]
    assert sorted(record.input_tokens for record in adapter.usage_records) == [1, 2]


async def test_usage_outside_a_scope_still_reaches_the_client() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        del request
        return httpx.Response(
            200,
            json={
                "usage": {"input_tokens": 7, "output_tokens": 3},
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": '{"text":"x"}'}],
                    }
                ],
            },
        )

    client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="https://api.openai.test/v1"
    )
    adapter = OpenAIResponsesLLM(api_key="sk-test", model="model-test", client=client)
    try:
        await adapter.complete(
            task="response",
            messages=({"role": "system", "content": "s"},),
            response_model=GeneratedReply,
        )
    finally:
        await client.aclose()

    assert [record.input_tokens for record in adapter.usage_records] == [7]
