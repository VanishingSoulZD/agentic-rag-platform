from __future__ import annotations

from app.llm.client import AsyncLLMClient
from app.llm.factory import create_provider
from app.llm.providers.fireworks import FireworksProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.openrouter import OpenRouterProvider


def test_factory_uses_fireworks_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    provider = create_provider()
    assert isinstance(provider, FireworksProvider)


def test_factory_can_select_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    provider = create_provider()
    assert isinstance(provider, OpenRouterProvider)


def test_factory_can_select_gemini(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    provider = create_provider()
    assert isinstance(provider, GeminiProvider)


def test_async_llm_client_delegates_to_factory_default(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    client = AsyncLLMClient()
    assert isinstance(client.provider, FireworksProvider)
