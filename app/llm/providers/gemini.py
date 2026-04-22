from __future__ import annotations

import os

from config import config

from app.llm.providers.openai_compatible import OpenAICompatibleProvider


class GeminiProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
    ):
        super().__init__(
            provider_name="gemini",
            api_key=api_key or os.getenv("GEMINI_API_KEY"),
            base_url=base_url or os.getenv("GEMINI_BASE_URL"),
            model=model or config.GEMINI_MODEL,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
