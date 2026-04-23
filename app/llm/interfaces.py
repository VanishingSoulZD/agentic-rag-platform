from __future__ import annotations

from collections.abc import AsyncGenerator
from typing import Protocol

from app.llm.types import LLMResult


class AsyncLLMProvider(Protocol):
    async def chat(self, messages: list[dict[str, str]]) -> LLMResult: ...

    async def stream_chat(
        self, messages: list[dict[str, str]]
    ) -> AsyncGenerator[dict, None]: ...
