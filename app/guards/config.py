from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).parents[2]
GUARDRAILS_PATH = ROOT / "config" / "guardrails.yaml"
CONTACT_ALLOWLIST_PATH = ROOT / "config" / "contact_allowlist.yaml"


class GuardrailConfig(BaseModel):
    """Single source of the guardrail limits and thresholds (config/guardrails.yaml)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    message_max_characters: int = Field(ge=1, le=100_000)
    conversation_rate_limit: int = Field(ge=1)
    conversation_rate_window_seconds: float = Field(gt=0)
    repeated_deflect_close_after: int = Field(ge=1)
    confirmation_other_cancel_after: int = Field(ge=1)
    confirmation_window_minutes: int = Field(ge=1, le=60)
    classifier_medium_confidence: float = Field(ge=0, le=1)
    classifier_high_confidence: float = Field(ge=0, le=1)
    classifier_solo_deflect_confidence: float = Field(ge=0, le=1)
    grounding_min_quote_words: int = Field(ge=1)

    @model_validator(mode="after")
    def ordered_thresholds(self) -> GuardrailConfig:
        if not (
            self.classifier_medium_confidence
            <= self.classifier_high_confidence
            <= self.classifier_solo_deflect_confidence
        ):
            raise ValueError("classifier thresholds must be medium <= high <= solo deflect")
        return self


def load_guardrail_config(path: Path = GUARDRAILS_PATH) -> GuardrailConfig:
    return GuardrailConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


@lru_cache(maxsize=1)
def guardrail_config() -> GuardrailConfig:
    return load_guardrail_config()


class ContactAllowlist(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    contacts: tuple[str, ...] = ()


def load_contact_allowlist(path: Path = CONTACT_ALLOWLIST_PATH) -> tuple[str, ...]:
    """Static allowlist; empty by default. Never learned from state, model or chunks."""
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ContactAllowlist.model_validate(payload).contacts
