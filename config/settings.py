from functools import lru_cache
from typing import Literal

from pydantic import AnyHttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/collections"
    mock_api_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8001")
    openai_api_key: SecretStr | None = None
    openai_agent_model: str = "gpt-5-nano"
    cohere_api_key: SecretStr | None = None

    mock_token_secret: SecretStr = SecretStr("local-development-secret-change-me")
    mock_token_issuer: str = "financial-agent-local-idp"
    mock_token_audience: str = "collections-api"
    mock_token_ttl_seconds: int = 300

    tool_timeout_seconds: float = 0.25
    tool_retry_attempts: int = 3
    circuit_breaker_threshold: int = 3
    circuit_breaker_reset_seconds: float = 30.0

    # Message limits, rate limits and guard thresholds live in config/guardrails.yaml.
    conversation_lock_timeout_seconds: float = 10.0
    # Static per deployment. Production derives a stable fallback from MOCK_TOKEN_SECRET.
    system_prompt_canary: SecretStr | None = None

    # §7.4 fusion gate. The absolute dense gate is calibrated on evals/retrieval_dev.yaml only
    # (make calibrate-rag) and must never be tuned against the held-out test split.
    rag_min_rrf_score: float = 0.35
    rag_min_dense_score: float = 0.505  # make calibrate-rag (dev split), 2026-09-12
    rag_min_rerank_score: float = 0.209  # make calibrate-rerank (dev split), 2026-09-12
    # Chosen on dev: dense gate 24/32 + 9/10 vs rerank gate 22/32 + 9/10 (balanced accuracy).
    rag_evidence_gate: Literal["dense", "rerank"] = "dense"


@lru_cache
def get_settings() -> Settings:
    return Settings()
