from __future__ import annotations

import os

from app.llm.providers.openai_compatible import OpenAICompatibleProvider


class OpenAIProvider(OpenAICompatibleProvider):
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
    ):
        super().__init__(
            provider_name="openai",
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE_URL"),
            model=model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
