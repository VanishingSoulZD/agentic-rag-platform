from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")
pytest.importorskip("langgraph")

from langchain_core.messages import AIMessage

try:
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
except Exception:  # pragma: no cover - compatibility fallback
    FakeMessagesListChatModel = None

from app.langchain_tools.agent import build_calculator_agent, run_calculator_agent
from app.langchain_tools.calculator import calculate_expression
from app.langchain_tools.registry import build_calculator_tool


@pytest.mark.skipif(FakeMessagesListChatModel is None, reason="FakeMessagesListChatModel not available")
def test_agent_uses_calculator_tool_and_returns_result() -> None:
    llm = FakeMessagesListChatModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "Calculator",
                        "args": {"expression": "1 * 7 + 5"},
                        "id": "call_1",
                    }
                ],
            ),
            AIMessage(content="12"),
        ]
    )
    agent = build_calculator_agent(llm)

    result = run_calculator_agent(agent, "帮我算一下 1 乘以 7 再加 5")

    assert result == "12"


def test_calculator_tool_returns_expected_result() -> None:
    calculator_tool = build_calculator_tool()
    assert calculator_tool.invoke({"expression": "1 * 7 + 5"}) == "12"


def test_calculator_rejects_unsafe_expression() -> None:
    with pytest.raises(ValueError):
        calculate_expression("__import__('os').system('echo unsafe')")
