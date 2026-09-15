"""Blind policy questions: labels come from the situation, text from a writer that sees none."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml  # type: ignore[import-untyped]

from app.llm.protocol import ScriptedLLM
from scripts import generate_policy_phrasings as policy_phrasings
from scripts.generate_blind_phrasings import Phrasings


def _replies(per_situation: int, *, prefix: str = "pregunta") -> list[Phrasings]:
    return [
        Phrasings(
            phrases=[
                f"{prefix} {situation.case_id} número {index}" for index in range(per_situation + 2)
            ]
        )
        for situation in policy_phrasings.SITUATIONS
    ]


async def test_generate_skips_known_questions_and_never_shows_the_labels() -> None:
    llm = ScriptedLLM(_replies(2))
    known = ["pregunta S-61 número 0"]
    phrasings = await policy_phrasings.generate(llm, per_situation=2, existing=known)
    assert phrasings["S-61"] == ["pregunta S-61 número 1", "pregunta S-61 número 2"]
    prompt = llm.calls[0].messages[-1]["content"]
    assert "pregunta S-61 número 0" in prompt
    assert "POL-NEG" not in prompt and "answerable" not in prompt

    payload = policy_phrasings.case_file(phrasings)
    rows = {case["id"]: case for case in payload["cases"]}
    assert rows["S-61:b1"]["sections"] == ["POL-NEG-003"] and rows["S-61:b1"]["answerable"]
    assert rows["N-61:b2"]["answerable"] is False and rows["N-61:b2"]["sections"] == []
    rendered = policy_phrasings.render(
        payload, model="writer-test", generated_at=datetime(2026, 9, 14, tzinfo=UTC)
    )
    assert rendered.startswith("# Preguntas de política CIEGAS")
    assert yaml.safe_load(rendered)["cases"][0]["query"] == "pregunta S-61 número 1"


async def test_generate_requires_enough_new_questions() -> None:
    repeated = [Phrasings(phrases=["la misma", "la misma"])]
    with pytest.raises(ValueError, match="S-61"):
        await policy_phrasings.generate(ScriptedLLM(repeated), per_situation=2, existing=[])


def test_known_questions_read_the_regression_sets(tmp_path: Path) -> None:
    questions = policy_phrasings.known_questions()
    assert "¿tengo que dar algo de entrada?" in questions
    assert policy_phrasings.known_questions([tmp_path / "missing.yaml"]) == []


def test_run_writes_a_new_file_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripted = ScriptedLLM(_replies(1, prefix="ciega"))

    class Writer:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def complete(self, **kwargs: Any) -> Any:
            return await scripted.complete(**kwargs)

        async def aclose(self) -> None:
            return None

    monkeypatch.setattr(policy_phrasings, "_writer", lambda model: ("sk-test", "writer-test"))
    monkeypatch.setattr(policy_phrasings, "OpenAIResponsesLLM", Writer)
    output = tmp_path / "policy_blind.yaml"
    policy_phrasings.main(["--output", str(output), "--per-situation", "1"])
    content = output.read_text(encoding="utf-8")
    assert "Modelo: writer-test" in content
    assert len(yaml.safe_load(content)["cases"]) == len(policy_phrasings.SITUATIONS)
    with pytest.raises(FileExistsError):
        policy_phrasings.main(["--output", str(output)])
