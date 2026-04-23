from __future__ import annotations

import os
from collections.abc import AsyncGenerator

from app.llm.factory import create_provider
from app.llm.types import LLMResult


class AsyncLLMClient:
    def __init__(
        self,
        provider: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float | None = None,
        max_retries: int | None = None,
    ):
        resolved_timeout_seconds = timeout_seconds
        if resolved_timeout_seconds is None:
            resolved_timeout_seconds = float(
                os.getenv(
                    "LLM_TIMEOUT_SECONDS", os.getenv("OPENAI_TIMEOUT_SECONDS", "20")
                )
            )
        resolved_max_retries = max_retries
        if resolved_max_retries is None:
            resolved_max_retries = int(
                os.getenv("LLM_MAX_RETRIES", os.getenv("OPENAI_MAX_RETRIES", "2"))
            )

        self.provider = create_provider(
            provider=provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=resolved_timeout_seconds,
            max_retries=resolved_max_retries,
        )

    async def chat(self, messages: list[dict[str, str]]) -> LLMResult:
        return await self.provider.chat(messages)

    async def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[dict, None]:
        async for event in self.provider.stream_chat(messages):
            yield event
