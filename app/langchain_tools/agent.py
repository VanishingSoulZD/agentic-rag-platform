"""LangGraph-based agent wiring for tool calling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.langchain_tools.db import DEFAULT_DB_PATH
from app.langchain_tools.registry import build_calculator_tool, build_db_query_tool, build_weather_tool


def build_calculator_agent(llm: BaseChatModel):
    """Build a LangGraph ReAct agent that can call the Calculator tool."""

    return create_agent(
        model=llm,
        tools=[build_calculator_tool()],
        system_prompt="You are a careful math assistant. Use Calculator for arithmetic.",
    )


def build_agent(llm: BaseChatModel, db_path: Path = DEFAULT_DB_PATH):
    """Build agent with Calculator + Weather + SQLite tools."""

    return create_agent(
        model=llm,
        tools=[build_calculator_tool(), build_weather_tool(), build_db_query_tool(db_path=db_path)],
        system_prompt=(
            "You are a helpful assistant. Use tools when user asks for math, weather, or user DB information. "
            "When multiple tool results are needed, call each tool and provide an integrated final answer."
        ),
    )


def run_agent(agent: Any, question: str) -> str:
    """Execute the agent and return the last AI answer text."""

    result = agent.invoke({"messages": [("user", question)]})
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    raise RuntimeError("Agent did not return a final AI message.")


def run_calculator_agent(agent: Any, question: str) -> str:
    """Backward-compatible alias for calculator-only usage."""

    return run_agent(agent, question)
