from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: SecretStr
    openai_model: str = "gpt-4.1-mini"
    app_name: str = "Autonomous Data Detective"
    max_investigation_steps: int = 8


@lru_cache
def get_settings() -> Settings:
    return Settings()
