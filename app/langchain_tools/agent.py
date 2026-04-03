"""LangGraph-based agent wiring for calculator tool calling."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

from app.langchain_tools.registry import build_calculator_tool


def build_calculator_agent(llm: BaseChatModel):
    """Build a LangGraph ReAct agent that can call the Calculator tool."""
    return create_agent(
        model=llm,
        tools=[build_calculator_tool()],
        system_prompt="You are a careful math assistant. Use Calculator for arithmetic.",
    )


def run_calculator_agent(agent: Any, question: str) -> str:
    """Execute the agent and return the last AI answer text."""
    result = agent.invoke({"messages": [("user", question)]})
    for message in reversed(result["messages"]):
        if isinstance(message, AIMessage) and message.content:
            return str(message.content)
    raise RuntimeError("Agent did not return a final AI message.")
