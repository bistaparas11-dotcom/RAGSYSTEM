from pydantic import SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    SEC_API_KEY: str
    OCR_API_URL: str
    GROQ_API_KEY: str
    DEEPSEEK_API_URL: str
    NEO4J_URI: str
    NEO4J_USERNAME: str
    NEO4J_PASSWORD: str
    NVIDIA_NIM_API: str
    WEAVIATE_URL: str
    WEAVIATE_API_KEY: str
    MODEL: str = "llama-3.1-8b-instant"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache()
def get_settings():
    """
    lru_cache to ensure we only read the .env file once,
    even if we call this function in multiple files.
    """
    return Settings()


settings = get_settings()
