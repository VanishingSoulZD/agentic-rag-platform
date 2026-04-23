from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    FIREWORKS_MODEL: str = "accounts/fireworks/models/deepseek-v3p1"
    GEMINI_MODEL: str = "gemini-2.5-flash-lite"
    OPENROUTER_MODEL_FALLBACK: str = "openrouter/free"
    OPENROUTER_MODEL_CHAT: str = "meta-llama/llama-3.3-70b-instruct:free"
    OPENROUTER_MODEL_RAG: str = "google/gemma-4-31b-it:free"
    OPENROUTER_MODEL_CODE: str = "qwen/qwen3-coder:free"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


config = Config()
