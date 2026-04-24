from __future__ import annotations

import asyncio
import os

import pytest

from app.llm.client import AsyncLLMClient
from app.llm.factory import create_provider
from app.llm.providers.fireworks import FireworksProvider
from app.llm.providers.gemini import GeminiProvider
from app.llm.providers.openai import OpenAIProvider
from app.llm.providers.openrouter import OpenRouterProvider


def test_factory_defaults_to_openai_when_only_openai_credentials_present(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1-mini")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provider = create_provider()

    assert isinstance(provider, OpenAIProvider)


def test_factory_defaults_to_openai_when_no_keys_and_mock_enabled(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("MOCK_LLM", "true")

    provider = create_provider()

    assert isinstance(provider, OpenAIProvider)
    assert provider.mock_mode is True


def test_factory_can_auto_detect_openrouter(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-openrouter-key")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)

    provider = create_provider()

    assert isinstance(provider, OpenRouterProvider)


def test_factory_can_auto_detect_gemini(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    provider = create_provider()

    assert isinstance(provider, GeminiProvider)


def test_factory_with_api_key_arg_only_defaults_to_openai(monkeypatch) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    provider = create_provider(api_key="test-openai-key")

    assert isinstance(provider, OpenAIProvider)


def test_factory_explicit_provider_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="requires API key"):
        create_provider(provider="fireworks")


def test_factory_explicit_openai_provider_allows_mock_without_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("MOCK_LLM", "true")

    provider = create_provider(provider="openai")

    assert isinstance(provider, OpenAIProvider)
    assert provider.mock_mode is True


def test_factory_explicit_provider_still_supported(monkeypatch) -> None:
    monkeypatch.setenv("FIREWORKS_API_KEY", "test-fireworks-key")

    provider = create_provider(provider="fireworks")

    assert isinstance(provider, FireworksProvider)


def test_async_llm_client_delegates_to_factory_default(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")
    monkeypatch.delenv("FIREWORKS_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    client = AsyncLLMClient()

    assert isinstance(client.provider, OpenAIProvider)


def _run_hello(provider_name: str) -> tuple[str, bool]:
    provider = create_provider(provider=provider_name)
    result = asyncio.run(provider.chat([{"role": "user", "content": "hello"}]))
    return result.answer.lower(), result.mock


@pytest.mark.integration
def test_openai_provider_hello_real_or_mock(monkeypatch) -> None:
    if not os.getenv("OPENAI_API_KEY"):
        monkeypatch.setenv("MOCK_LLM", "true")

    answer, is_mock = _run_hello("openai")

    assert "hello" in answer or is_mock


@pytest.mark.integration
def test_fireworks_provider_hello() -> None:
    if not os.getenv("FIREWORKS_API_KEY"):
        pytest.skip("FIREWORKS_API_KEY not configured; skip fireworks integration test")

    answer, is_mock = _run_hello("fireworks")

    assert not is_mock
    assert answer.strip()


@pytest.mark.integration
def test_openrouter_provider_hello() -> None:
    if not os.getenv("OPENROUTER_API_KEY"):
        pytest.skip(
            "OPENROUTER_API_KEY not configured; skip openrouter integration test"
        )

    answer, is_mock = _run_hello("openrouter")

    assert not is_mock
    assert answer.strip()


@pytest.mark.integration
def test_gemini_provider_hello() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not configured; skip gemini integration test")

    answer, is_mock = _run_hello("gemini")

    assert not is_mock
    assert answer.strip()
