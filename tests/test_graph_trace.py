from __future__ import annotations

import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.langchain_tools.graph_trace import (
    build_execution_graph,
    build_mermaid_html,
    save_execution_graph,
)
from app.main import app


def test_build_and_save_graph(tmp_path: Path) -> None:
    execution_result = {
        "question": "查资料+计算+整理",
        "plan": [
            {
                "step_id": 1,
                "kind": "tool",
                "instruction": "query db",
                "tool_name": "UserDBQuery",
            },
            {
                "step_id": 2,
                "kind": "tool",
                "instruction": "calculate",
                "tool_name": "Calculator",
            },
            {"step_id": 3, "kind": "summary", "instruction": "summarize"},
        ],
        "observations": [{"tool": "Calculator", "output": "12"}],
        "answer": "final summary",
    }

    graph = build_execution_graph(execution_result)
    trace_id, output_path = save_execution_graph(
        graph, output_dir=tmp_path, trace_id="trace123"
    )

    assert trace_id == "trace123"
    assert output_path.exists()
    loaded = json.loads(output_path.read_text(encoding="utf-8"))
    assert any(node["id"] == "step_1" for node in loaded["nodes"])
    assert any("LLM Summary" in node["label"] for node in loaded["nodes"])


def test_mermaid_html_contains_tool_and_llm_nodes() -> None:
    graph = {
        "meta": {"question": "test", "generated_at": "2026-01-01T00:00:00Z"},
        "nodes": [
            {"id": "start", "label": "User Question", "type": "input"},
            {"id": "step_1", "label": "UserDBQuery\\nquery db", "type": "tool"},
            {"id": "step_2", "label": "LLM Summary\\nsummary", "type": "summary"},
        ],
        "edges": [
            {"from": "start", "to": "step_1", "label": "next"},
            {"from": "step_1", "to": "step_2", "label": "next"},
        ],
        "observations": [],
        "answer": "done",
    }
    html = build_mermaid_html(graph)
    assert "UserDBQuery" in html
    assert "LLM Summary" in html
    assert "mermaid" in html


def test_agent_trace_endpoints(monkeypatch, tmp_path: Path) -> None:
    from app import main as main_module

    async def fake_execute(question: str):
        return {
            "question": question,
            "plan": [
                {
                    "step_id": 1,
                    "kind": "tool",
                    "instruction": "query db",
                    "tool_name": "UserDBQuery",
                },
                {"step_id": 2, "kind": "summary", "instruction": "summarize"},
            ],
            "observations": [
                {
                    "step_id": 1,
                    "tool": "UserDBQuery",
                    "input": "SELECT 1",
                    "output": "[(1,)]",
                }
            ],
            "answer": "summary answer",
        }

    monkeypatch.setattr(main_module.planner_executor_agent, "execute", fake_execute)
    monkeypatch.setattr(
        main_module,
        "save_execution_graph",
        lambda graph: ("trace999", tmp_path / "trace999.json"),
    )

    (tmp_path / "trace999.json").write_text(
        json.dumps(
            {
                "meta": {"question": "q", "generated_at": "now"},
                "nodes": [{"id": "step_1", "label": "UserDBQuery", "type": "tool"}],
                "edges": [],
                "observations": [],
                "answer": "ok",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        main_module,
        "load_execution_graph",
        lambda trace_id: json.loads(
            (tmp_path / "trace999.json").read_text(encoding="utf-8")
        ),
    )

    client = TestClient(app)
    create_resp = client.post("/agent/trace", json={"question": "test question"})
    assert create_resp.status_code == 200
    assert create_resp.json()["trace_id"] == "trace999"

    json_resp = client.get("/agent/trace/trace999")
    assert json_resp.status_code == 200
    assert json_resp.json()["nodes"][0]["label"] == "UserDBQuery"

    view_resp = client.get("/agent/trace/trace999/view")
    assert view_resp.status_code == 200
    assert "mermaid" in view_resp.text
