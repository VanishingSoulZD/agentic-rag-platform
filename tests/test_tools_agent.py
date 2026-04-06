from __future__ import annotations

from pathlib import Path

import pytest

from app.langchain_tools.calculator import calculate_expression
from app.langchain_tools.db import initialize_local_user_db, query_local_user_db
from app.langchain_tools.weather import get_weather


def test_weather_tool_mock() -> None:
    assert get_weather("Taipei") == "Taipei: Sunny, 25°C"


def test_db_query_tool_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "users.db"
    initialize_local_user_db(db_path)

    result = query_local_user_db("SELECT name, city FROM users ORDER BY id LIMIT 2", db_path=db_path)
    assert "Alice" in result and "Bob" in result


def test_db_query_rejects_non_select(tmp_path: Path) -> None:
    db_path = tmp_path / "users.db"
    initialize_local_user_db(db_path)

    with pytest.raises(ValueError):
        query_local_user_db("DELETE FROM users", db_path=db_path)


def test_agent_can_call_weather_and_db_tools(tmp_path: Path) -> None:
    pytest.importorskip("langchain_core")
    pytest.importorskip("langgraph")

    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from app.langchain_tools.agent import build_agent, run_agent

    class RuleBasedToolModel(BaseChatModel):
        def __init__(self):
            super().__init__()
            self._tools = []

        @property
        def _llm_type(self) -> str:
            return "rule_based_tool_model"

        def bind_tools(self, tools, **kwargs):
            self._tools = tools
            return self

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            if any(isinstance(msg, ToolMessage) for msg in messages):
                tool_outputs = [msg.content for msg in messages if isinstance(msg, ToolMessage)]
                final = " | ".join(tool_outputs)
                return ChatResult(
                    generations=[ChatGeneration(message=AIMessage(content=f"Integrated result: {final}"))])

            user_text = ""
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_text = msg.content
                    break

            if "weather" in user_text.lower() and "alice" in user_text.lower():
                return ChatResult(
                    generations=[
                        ChatGeneration(
                            message=AIMessage(
                                content="",
                                tool_calls=[
                                    {"name": "UserDBQuery",
                                     "args": {"query": "SELECT city FROM users WHERE name='Alice'"}, "id": "db_1"},
                                    {"name": "WeatherAPI", "args": {"city": "Taipei"}, "id": "w_1"},
                                ],
                            )
                        )
                    ]
                )

            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="No tool needed"))])

    db_path = tmp_path / "users.db"
    initialize_local_user_db(db_path)

    agent = build_agent(RuleBasedToolModel(), db_path=db_path)
    answer = run_agent(agent, "Please tell me Alice city from DB and weather of that city")

    assert "Taipei" in answer
    assert "Alice" not in answer  # final answer should be integrated tool outputs only


def test_calculator_still_works() -> None:
    assert calculate_expression("1 * 7 + 5") == "12"
