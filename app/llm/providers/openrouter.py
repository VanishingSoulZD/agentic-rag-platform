from __future__ import annotations

import os

from app.llm.providers.openai_compatible import OpenAICompatibleProvider
from config import config


class OpenRouterProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
    ):
        super().__init__(
            provider_name="openrouter",
            api_key=api_key or os.getenv("OPENROUTER_API_KEY"),
            base_url=base_url or os.getenv("OPENROUTER_BASE_URL"),
            model=model or config.OPENROUTER_MODEL_RAG,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
