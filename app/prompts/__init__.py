"""Prompt files. They are identified by content hash (published with every evaluation), not by a
version in the filename, so there is always exactly one current prompt."""

from __future__ import annotations

from pathlib import Path

PROMPTS_DIR = Path(__file__).parent
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system.md"
JUDGE_PROMPT_PATH = PROMPTS_DIR / "judge.md"


def load_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")
