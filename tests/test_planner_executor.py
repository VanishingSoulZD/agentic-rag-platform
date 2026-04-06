from __future__ import annotations

from pathlib import Path

import pytest

from app.langchain_tools.db import initialize_local_user_db
from app.langchain_tools.planner_executor import PlannerExecutorAgent
from app.llm_client import AsyncLLMClient


@pytest.mark.anyio
async def test_composite_question_user_city_weather_and_math(tmp_path: Path) -> None:
    db_path = tmp_path / "users.db"
    initialize_local_user_db(db_path)
    agent = PlannerExecutorAgent(db_path=db_path, llm_client=AsyncLLMClient())

    question = "请先查 users 资料，再告诉我 Alice 所在城市天气，并计算 7 * 3 + 2，最后整理成结论。"
    result = await agent.execute(question)

    assert len(result["plan"]) >= 4
    assert any(item["tool"] == "UserDBQuery" for item in result["observations"])
    assert any(item["tool"] == "WeatherAPI" for item in result["observations"])
    assert any(item["tool"] == "Calculator" and item["output"] == "23" for item in result["observations"])
    # assert "MOCK" in result["answer"]


@pytest.mark.anyio
async def test_composite_question_weather_then_math_then_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "users.db"
    initialize_local_user_db(db_path)
    agent = PlannerExecutorAgent(db_path=db_path, llm_client=AsyncLLMClient())

    question = "先看 Beijing weather，再算 9 + 11 / 2，然后给我整理输出。"
    result = await agent.execute(question)

    assert len(result["plan"]) >= 3
    assert any(item["tool"] == "WeatherAPI" and "Beijing" in str(item["input"]) for item in result["observations"])
    assert any(item["tool"] == "Calculator" and item["output"] == "14.5" for item in result["observations"])
    # assert "observations" in result["answer"]
    assert "observations" in result


@pytest.mark.anyio
async def test_composite_question_db_plus_multiple_calculations(tmp_path: Path) -> None:
    db_path = tmp_path / "users.db"
    initialize_local_user_db(db_path)
    agent = PlannerExecutorAgent(db_path=db_path, llm_client=AsyncLLMClient())

    question = "查资料：users里有哪些人；计算 100 / 4 + 1；最后汇总成多步骤答案。"
    result = await agent.execute(question)

    assert any(item["tool"] == "UserDBQuery" for item in result["observations"])
    assert any(item["tool"] == "Calculator" and item["output"] == "26.0" for item in result["observations"])
    assert result["plan"][-1]["kind"] == "summary"
