from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field

MODELS_PATH = Path(__file__).parents[2] / "config" / "models.yaml"


class EmbeddingModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str
    dimensions: int = Field(gt=0)


class RerankerModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str
    model: str


def _tier(path: Path, name: str) -> Any:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    try:
        return raw["tiers"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"config/models.yaml no define tiers.{name}") from exc


@lru_cache(maxsize=1)
def load_embedding_config(path: Path = MODELS_PATH) -> EmbeddingModelConfig:
    return EmbeddingModelConfig.model_validate(_tier(path, "embedding"))


@lru_cache(maxsize=1)
def load_reranker_config(path: Path = MODELS_PATH) -> RerankerModelConfig:
    return RerankerModelConfig.model_validate(_tier(path, "reranker"))
