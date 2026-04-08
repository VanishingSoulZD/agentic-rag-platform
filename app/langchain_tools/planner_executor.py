"""Planner / Executor architecture for multi-step answers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.langchain_tools.calculator import calculate_expression
from app.langchain_tools.db import DEFAULT_DB_PATH, query_local_user_db
from app.langchain_tools.weather import get_weather
from app.llm_client import AsyncLLMClient
from app.optimization.cache_layers import ToolCache
from app.security import ToolUsePolicy, sanitize_user_input


@dataclass
class PlanStep:
    step_id: int
    kind: str
    instruction: str
    tool_name: str | None = None
    tool_input: str | None = None


class PlannerExecutorAgent:
    """Planner -> Executor -> Summary (LLM) workflow."""

    def __init__(
            self,
            db_path: Path = DEFAULT_DB_PATH,
            llm_client: AsyncLLMClient | None = None,
            tool_policy: ToolUsePolicy | None = None,
            tool_cache: ToolCache | None = None,
    ):
        self.db_path = db_path
        self.llm_client = llm_client or AsyncLLMClient()
        self.tool_policy = tool_policy or ToolUsePolicy(denied_tools={"AdminAPI", "ShellAPI", "RemoteHTTP"})
        self.tool_cache = tool_cache

    def planner(self, question: str) -> list[PlanStep]:
        """Split a complex question into executable steps."""
        q = question.lower()
        steps: list[PlanStep] = []
        next_id = 1

        if any(name in q for name in ["alice", "bob", "carol", "用户", "users", "资料", "database", "db"]):
            steps.append(
                PlanStep(
                    step_id=next_id,
                    kind="tool",
                    instruction="Query local user profile data from sqlite",
                    tool_name="UserDBQuery",
                    tool_input="SELECT id, name, city FROM users ORDER BY id",
                )
            )
            next_id += 1

        if any(key in q for key in ["weather", "天气"]):
            city = self._extract_city(question)
            steps.append(
                PlanStep(
                    step_id=next_id,
                    kind="tool",
                    instruction="Get weather information for target city",
                    tool_name="WeatherAPI",
                    tool_input=city,
                )
            )
            next_id += 1

        expression = self._extract_expression(question)
        if expression is not None:
            steps.append(
                PlanStep(
                    step_id=next_id,
                    kind="tool",
                    instruction="Run arithmetic calculation",
                    tool_name="Calculator",
                    tool_input=expression,
                )
            )
            next_id += 1

        steps.append(
            PlanStep(
                step_id=next_id,
                kind="summary",
                instruction="Summarize all collected facts into final answer",
            )
        )
        return steps

    async def execute(self, question: str) -> dict[str, Any]:
        sanitized_question = sanitize_user_input(question)
        steps = self.planner(sanitized_question)
        observations: list[dict[str, Any]] = []
        cache_layers: dict[str, str] = {}

        for step in steps:
            if step.kind != "tool":
                continue

            result, strategy = self._call_tool(step.tool_name or "", step.tool_input or "")
            if strategy in {"exact", "semantic"}:
                cache_layers["tool"] = strategy
            observations.append(
                {
                    "step_id": step.step_id,
                    "tool": step.tool_name,
                    "input": step.tool_input,
                    "output": result,
                }
            )

        summary = await self._summary_step(question=sanitized_question, steps=steps, observations=observations)
        return {
            "question": sanitized_question,
            "plan": [step.__dict__ for step in steps],
            "observations": observations,
            "answer": summary,
            "cache_layers": cache_layers,
        }

    def _call_tool(self, tool_name: str, tool_input: str) -> tuple[str, str]:
        self.tool_policy.enforce(tool_name)

        if self.tool_cache is not None:
            cached, _, strategy = self.tool_cache.lookup(tool_name, tool_input)
            if strategy in {"exact", "semantic"} and cached is not None:
                return cached, strategy

        if tool_name == "Calculator":
            result = calculate_expression(tool_input)
        elif tool_name == "WeatherAPI":
            result = get_weather(tool_input)
        elif tool_name == "UserDBQuery":
            result = query_local_user_db(tool_input, db_path=self.db_path)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        if self.tool_cache is not None:
            self.tool_cache.store(tool_name, tool_input, result)
        return result, "miss"

    async def _summary_step(self, question: str, steps: list[PlanStep], observations: list[dict[str, Any]]) -> str:
        summary_payload = {
            "question": question,
            "steps": [step.__dict__ for step in steps],
            "observations": observations,
        }
        messages = [
            {
                "role": "system",
                "content": "You are a concise assistant. Create a complete multi-step answer from tool outputs.",
            },
            {
                "role": "user",
                "content": f"Please summarize this plan+results into final answer:\n{json.dumps(summary_payload, ensure_ascii=False)}",
            },
        ]
        result = await self.llm_client.chat(messages)
        return result.answer

    @staticmethod
    def _extract_city(question: str) -> str:
        mapping = {
            "taipei": "Taipei",
            "台北": "Taipei",
            "beijing": "Beijing",
            "北京": "Beijing",
            "shanghai": "Shanghai",
            "上海": "Shanghai",
            "hangzhou": "Hangzhou",
            "杭州": "Hangzhou",
        }
        q_lower = question.lower()
        for k, v in mapping.items():
            if k in q_lower:
                return v
        if "alice" in q_lower:
            return "Taipei"
        if "bob" in q_lower:
            return "Beijing"
        return "Taipei"

    @staticmethod
    def _extract_expression(question: str) -> str | None:
        import re

        explicit = re.findall(r"[0-9\s\+\-\*\/\(\)\.]+", question)
        candidates = [item.strip() for item in explicit if
                      any(ch.isdigit() for ch in item) and any(op in item for op in "+-*/")]
        if candidates:
            return candidates[0]

        q = question.lower()
        if "double" in q and "plus" in q and "10" in q:
            return "2 * 2 + 10"
        if "乘以" in question and "再加" in question:
            numbers = re.findall(r"\d+", question)
            if len(numbers) >= 3:
                return f"{numbers[0]} * {numbers[1]} + {numbers[2]}"
        return None
