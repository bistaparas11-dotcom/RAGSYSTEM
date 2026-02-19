from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    SEC_API_KEY: str
    OCR_API_URL: str

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"  # Ignore extra variables in .env
    )


@lru_cache()
def get_settings():
    """
    lru_cache to ensure we only read the .env file once,
    even if we call this function in multiple files.
    """
    return Settings()


settings = get_settings()
