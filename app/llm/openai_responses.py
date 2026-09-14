from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import httpx
from pydantic import BaseModel

ReasoningEffort = Literal["minimal", "low", "medium", "high"]


class OpenAIResponseError(RuntimeError):
    """The Responses API completed without usable structured output."""


class OpenAIRefusalError(OpenAIResponseError):
    pass


class OpenAIIncompleteError(OpenAIResponseError):
    pass


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    task: str
    model: str
    input_tokens: int
    output_tokens: int
    cached_tokens: int


def _strict_json_schema(value: Any) -> Any:
    if isinstance(value, list):
        return [_strict_json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    transformed = {
        key: _strict_json_schema(item) for key, item in value.items() if key != "default"
    }
    properties = transformed.get("properties")
    if isinstance(properties, dict):
        transformed["required"] = list(properties)
        transformed["additionalProperties"] = False
    return transformed


class OpenAIResponsesLLM:
    """Structured-output implementation of ``LLMClient`` over the Responses API."""

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        max_output_tokens: int = 800,
        timeout_seconds: float = 20,
        reasoning_effort: ReasoningEffort | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        if not api_key.strip():
            raise ValueError("OpenAI API key is required")
        self._model = model
        self._max_output_tokens = max_output_tokens
        # Reasoning tokens count against max_output_tokens. Without an explicit effort gpt-5-nano
        # spent 560-700 tokens on a guard classification and ~10 % of live turns came back
        # incomplete; "low" keeps classifications near 250 tokens.
        self._reasoning = {"effort": reasoning_effort} if reasoning_effort else None
        self._owns_client = client is None
        self._usage_records: list[ProviderUsage] = []
        self._client = client or httpx.AsyncClient(
            base_url="https://api.openai.com/v1",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=timeout_seconds,
        )

    async def complete[T: BaseModel](
        self,
        *,
        task: str,
        messages: Sequence[Mapping[str, str]],
        response_model: type[T],
    ) -> T:
        instructions = "\n\n".join(
            message["content"] for message in messages if message.get("role") == "system"
        )
        model_input = [
            {"role": message["role"], "content": message["content"]}
            for message in messages
            if message.get("role") != "system"
        ]
        schema_name = re.sub(r"[^a-zA-Z0-9_-]", "_", f"{task}_{response_model.__name__}")[:64]
        response = await self._client.post(
            "/responses",
            json={
                "model": self._model,
                "instructions": instructions or None,
                "input": model_input,
                "max_output_tokens": self._max_output_tokens,
                **({"reasoning": self._reasoning} if self._reasoning else {}),
                "store": False,
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": _strict_json_schema(response_model.model_json_schema()),
                        "strict": True,
                    }
                },
            },
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
        usage = payload.get("usage")
        if isinstance(usage, dict):
            details = usage.get("input_tokens_details")
            cached = details.get("cached_tokens", 0) if isinstance(details, dict) else 0
            self._usage_records.append(
                ProviderUsage(
                    task=task,
                    model=self._model,
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                    cached_tokens=int(cached),
                )
            )
        status = payload.get("status")
        if status == "incomplete":
            details = payload.get("incomplete_details")
            reason = details.get("reason") if isinstance(details, dict) else None
            raise OpenAIIncompleteError(
                f"OpenAI response was incomplete ({reason or 'reason unavailable'})"
            )
        if status in {"failed", "cancelled"}:
            raise OpenAIResponseError(f"OpenAI response ended with status {status}")
        refused = any(
            content.get("type") == "refusal"
            for item in payload.get("output", [])
            if isinstance(item, dict) and item.get("type") == "message"
            for content in item.get("content", [])
            if isinstance(content, dict)
        )
        if refused:
            raise OpenAIRefusalError("OpenAI response was refused")
        output_text = next(
            (
                content.get("text")
                for item in payload.get("output", [])
                if item.get("type") == "message"
                for content in item.get("content", [])
                if content.get("type") == "output_text" and content.get("text")
            ),
            None,
        )
        if not isinstance(output_text, str):
            raise ValueError("OpenAI response did not contain structured output text")
        return response_model.model_validate(json.loads(output_text))

    @property
    def usage_records(self) -> tuple[ProviderUsage, ...]:
        return tuple(self._usage_records)

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
