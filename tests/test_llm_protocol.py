from __future__ import annotations

import pytest
from pydantic import BaseModel

from app.llm.protocol import ScriptedLLM, ScriptExhaustedError


class Result(BaseModel):
    value: str


async def test_scripted_llm_validates_records_and_exhausts_responses() -> None:
    llm = ScriptedLLM([{"value": "first"}, Result(value="second")])

    first = await llm.complete(
        task="route",
        messages=[{"role": "user", "content": "hola"}],
        response_model=Result,
    )
    second = await llm.complete(task="render", messages=(), response_model=Result)

    assert first == Result(value="first")
    assert second == Result(value="second")
    assert llm.remaining == 0
    assert [call.task for call in llm.calls] == ["route", "render"]
    assert llm.calls[0].messages == ({"role": "user", "content": "hola"},)
    assert llm.calls[0].response_model is Result

    with pytest.raises(ScriptExhaustedError, match="no-script"):
        await llm.complete(task="no-script", messages=(), response_model=Result)


async def test_scripted_llm_can_raise_a_declared_provider_failure() -> None:
    llm = ScriptedLLM([TimeoutError("provider timeout")])
    with pytest.raises(TimeoutError, match="provider timeout"):
        await llm.complete(task="route", messages=(), response_model=Result)
