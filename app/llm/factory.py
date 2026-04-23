from __future__ import annotations

import os

from app.llm.interfaces import AsyncLLMProvider
from app.llm.providers.fireworks import FireworksProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.openai import OpenAIProvider
from app.llm.providers.openrouter import OpenRouterProvider

_SUPPORTED_PROVIDERS = {"fireworks", "openrouter", "gemini", "openai"}
_PROVIDER_KEY_ENV = {
    "fireworks": "FIREWORKS_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
}


def _get_provider_api_key(provider: str, api_key: str | None) -> str | None:
    return api_key or os.getenv(_PROVIDER_KEY_ENV[provider])


def _build_provider(
    selected_provider: str,
    api_key: str | None,
    base_url: str | None,
    model: str | None,
    timeout_seconds: float,
    max_retries: int,
) -> AsyncLLMProvider:
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
    if selected_provider == "openai":
        return OpenAIProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )
    raise ValueError(f"Unsupported LLM provider: {selected_provider}")


def create_provider(
    provider: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    timeout_seconds: float = 20.0,
    max_retries: int = 2,
) -> AsyncLLMProvider:
    explicit_provider = provider.strip().lower() if provider else None

    if explicit_provider:
        if explicit_provider not in _SUPPORTED_PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {explicit_provider}")
        if explicit_provider != "openai" and not _get_provider_api_key(explicit_provider, api_key):
            env_name = _PROVIDER_KEY_ENV[explicit_provider]
            raise ValueError(
                f"{explicit_provider} provider requires API key via `{env_name}` or `api_key` argument"
            )
        return _build_provider(
            selected_provider=explicit_provider,
            api_key=api_key,
            base_url=base_url,
            model=model,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    for candidate in ("fireworks", "openrouter", "gemini"):
        if _get_provider_api_key(candidate, api_key):
            return _build_provider(
                selected_provider=candidate,
                api_key=api_key,
                base_url=base_url,
                model=model,
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )

    return _build_provider(
        selected_provider="openai",
        api_key=api_key,
        base_url=base_url,
        model=model,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
    )
