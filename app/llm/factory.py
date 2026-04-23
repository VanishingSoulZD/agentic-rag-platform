from __future__ import annotations

import os

from app.llm.interfaces import AsyncLLMProvider
from app.llm.providers.fireworks import FireworksProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.openrouter import OpenRouterProvider
from config import config


def create_provider(
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 20.0,
    max_retries: int = 2,
) -> AsyncLLMProvider:
    selected_provider = (
        (provider or os.getenv("LLM_PROVIDER") or config.LLM_PROVIDER or "fireworks")
        .strip()
        .lower()
    )
    if selected_provider == "fireworks":
        return FireworksProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    if selected_provider == "openrouter":
        return OpenRouterProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    if selected_provider == "gemini":
        return GeminiProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    raise ValueError(f"Unsupported LLM provider: {selected_provider}")
