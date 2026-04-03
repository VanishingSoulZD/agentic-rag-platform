"""LangChain tool registry (new ecosystem: langchain-core tools)."""

from __future__ import annotations

from langchain_core.tools import StructuredTool

from app.langchain_tools.calculator import calculate_expression


def build_calculator_tool() -> StructuredTool:
    """Create the Calculator tool for arithmetic requests."""
    return StructuredTool.from_function(
        func=calculate_expression,
        name="Calculator",
        description=(
            "Use this tool for arithmetic calculations. "
            "Input must be a plain math expression such as '1 + 7 * 5'."
        ),
    )
