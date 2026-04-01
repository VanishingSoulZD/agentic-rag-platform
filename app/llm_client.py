import asyncio
import logging
import os
from dataclasses import dataclass
from typing import AsyncGenerator

from openai import APIConnectionError, APIError, AsyncOpenAI, RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class LLMResult:
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    mock: bool


class AsyncLLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 20.0,
        max_retries: int = 2,
    ):
        self.api_key = api_key or os.getenv('OPENAI_API_KEY')
        self.base_url = base_url or os.getenv('OPENAI_API_BASE')
        self.model = model or os.getenv('OPENAI_MODEL', 'gpt-4.1-mini')
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.mock_mode = os.getenv('MOCK_LLM', '').lower() in {'1', 'true', 'yes'} or not self.api_key

        self.client = None
        if not self.mock_mode:
            self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _extract_latest_user_text(messages: list[dict[str, str]]) -> str:
        return next((m['content'] for m in reversed(messages) if m.get('role') == 'user'), '')

    async def chat(self, messages: list[dict[str, str]]) -> LLMResult:
        if self.mock_mode:
            user_text = self._extract_latest_user_text(messages)
            answer = f"[MOCK] 你问的是：{user_text}"
            prompt_text = '\n'.join(m.get('content', '') for m in messages)
            prompt_tokens = self._estimate_tokens(prompt_text)
            completion_tokens = self._estimate_tokens(answer)
            total_tokens = prompt_tokens + completion_tokens
            logger.info(
                'llm_use model=%s mock=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s',
                self.model,
                True,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )
            return LLMResult(answer, prompt_tokens, completion_tokens, total_tokens, self.model, True)

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                    ),
                    timeout=self.timeout_seconds,
                )

                answer = response.choices[0].message.content or ''
                usage = response.usage
                prompt_tokens = usage.prompt_tokens if usage else self._estimate_tokens(str(messages))
                completion_tokens = usage.completion_tokens if usage else self._estimate_tokens(answer)
                total_tokens = usage.total_tokens if usage else prompt_tokens + completion_tokens

                logger.info(
                    'llm_use model=%s mock=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s',
                    self.model,
                    False,
                    prompt_tokens,
                    completion_tokens,
                    total_tokens,
                )
                return LLMResult(answer, prompt_tokens, completion_tokens, total_tokens, self.model, False)
            except (TimeoutError, APIConnectionError, RateLimitError, APIError) as err:
                last_error = err
                if attempt >= self.max_retries:
                    break
                sleep_seconds = 0.5 * (2**attempt)
                logger.warning(
                    'llm_retry attempt=%s sleep_seconds=%.2f error=%s',
                    attempt + 1,
                    sleep_seconds,
                    repr(err),
                )
                await asyncio.sleep(sleep_seconds)

        raise RuntimeError(f'LLM request failed after retries: {last_error}')

    async def stream_chat(self, messages: list[dict[str, str]]) -> AsyncGenerator[dict, None]:
        if self.mock_mode:
            user_text = self._extract_latest_user_text(messages)
            answer = f"[MOCK] 你问的是：{user_text}"
            prompt_text = '\n'.join(m.get('content', '') for m in messages)
            prompt_tokens = self._estimate_tokens(prompt_text)
            completion_tokens = 0

            for ch in answer:
                completion_tokens += self._estimate_tokens(ch)
                yield {'type': 'token', 'content': ch}
                await asyncio.sleep(0.01)

            logger.info(
                'llm_use_stream model=%s mock=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s',
                self.model,
                True,
                prompt_tokens,
                completion_tokens,
                prompt_tokens + completion_tokens,
            )
            yield {
                'type': 'usage',
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
                'model': self.model,
                'mock': True,
            }
            return

        completion_text = ''
        prompt_tokens = self._estimate_tokens(str(messages))
        completion_tokens = 0

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=True,
            stream_options={'include_usage': True},
        )

        async for chunk in stream:
            if getattr(chunk, 'usage', None):
                prompt_tokens = chunk.usage.prompt_tokens or prompt_tokens
                completion_tokens = chunk.usage.completion_tokens or completion_tokens

            delta = chunk.choices[0].delta.content if chunk.choices else None
            if delta:
                completion_text += delta
                completion_tokens += self._estimate_tokens(delta)
                yield {'type': 'token', 'content': delta}

        total_tokens = prompt_tokens + completion_tokens
        logger.info(
            'llm_use_stream model=%s mock=%s prompt_tokens=%s completion_tokens=%s total_tokens=%s',
            self.model,
            False,
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )
        yield {
            'type': 'usage',
            'prompt_tokens': prompt_tokens,
            'completion_tokens': completion_tokens,
            'total_tokens': total_tokens,
            'model': self.model,
            'mock': False,
        }

