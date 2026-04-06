"""LangChain tool registry (new ecosystem: langchain-core tools)."""

from __future__ import annotations

from pathlib import Path

from langchain_core.tools import StructuredTool

from app.langchain_tools.calculator import calculate_expression
from app.langchain_tools.db import DEFAULT_DB_PATH, query_local_user_db
from app.langchain_tools.weather import get_weather


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


def build_weather_tool() -> StructuredTool:
    """Create mock weather API tool."""

    return StructuredTool.from_function(
        func=get_weather,
        name="WeatherAPI",
        description="Get weather by city name. Input example: 'Taipei'.",
    )


def build_db_query_tool(db_path: Path = DEFAULT_DB_PATH) -> StructuredTool:
    """Create local SQLite query tool (SELECT-only)."""

    def _query(query: str) -> str:
        return query_local_user_db(query=query, db_path=db_path)

    return StructuredTool.from_function(
        func=_query,
        name="UserDBQuery",
        description="Query local users sqlite DB with SELECT SQL.",
    )
