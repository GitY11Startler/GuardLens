"""Project settings, read from environment variables and an optional `.env` file."""

from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None  # for gated Hugging Face datasets
    configs_dir: Path = PROJECT_ROOT / "configs"
    data_dir: Path = PROJECT_ROOT / "data"
    results_dir: Path = PROJECT_ROOT / "results"


def get_settings() -> Settings:
    return Settings()
