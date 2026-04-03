from __future__ import annotations

import os

import pytest

from app.langchain_tools.calculator import calculate_expression


def test_calculator_rejects_unsafe_expression() -> None:
    with pytest.raises(ValueError):
        calculate_expression("__import__('os').system('echo unsafe')")


def test_calculator_tool_returns_expected_result() -> None:
    pytest.importorskip("langchain_core")
    from app.langchain_tools.registry import build_calculator_tool

    calculator_tool = build_calculator_tool()
    assert calculator_tool.invoke({"expression": "1 * 7 + 5"}) == "12"


@pytest.mark.integration
def test_agent_uses_calculator_tool_and_returns_result() -> None:
    pytest.importorskip("langchain_core")
    pytest.importorskip("langgraph")
    chat_openai = pytest.importorskip("langchain_openai")

    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        pytest.skip("OPENAI_API_KEY not configured; skip real-LLM tool-calling integration test")

    from app.langchain_tools.agent import build_calculator_agent, run_calculator_agent

    llm = chat_openai.ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
        api_key=openai_key,
        base_url=os.getenv("OPENAI_API_BASE"),
        temperature=0,
    )

    agent = build_calculator_agent(llm)
    result = run_calculator_agent(agent, "Please calculate 1 * 7 + 5")
    assert "12" in result
