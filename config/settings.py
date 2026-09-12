from functools import lru_cache

from pydantic import AnyHttpUrl, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_env: str = "local"
    database_url: str = "postgresql://postgres:postgres@localhost:5432/collections"
    mock_api_url: AnyHttpUrl = AnyHttpUrl("http://localhost:8001")

    mock_token_secret: SecretStr = SecretStr("local-development-secret-change-me")
    mock_token_issuer: str = "financial-agent-local-idp"
    mock_token_audience: str = "collections-api"
    mock_token_ttl_seconds: int = 300

    tool_timeout_seconds: float = 0.25
    tool_retry_attempts: int = 3
    circuit_breaker_threshold: int = 3
    circuit_breaker_reset_seconds: float = 30.0


@lru_cache
def get_settings() -> Settings:
    return Settings()
