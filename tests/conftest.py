import pytest


@pytest.fixture(autouse=True)
def auto_mock_llm(request, monkeypatch):
    if "integration" in request.keywords:
        monkeypatch.setenv("MOCK_LLM", "false")
    else:
        monkeypatch.setenv("MOCK_LLM", "true")
