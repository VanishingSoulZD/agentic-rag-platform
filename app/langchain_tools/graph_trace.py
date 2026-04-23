"""Execution graph trace (JSON + Mermaid rendering)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

TRACE_OUTPUT_DIR = Path("reports/agent_traces")


def build_execution_graph(execution_result: dict) -> dict:
    """Convert planner/executor result to graph JSON format."""
    nodes = [{"id": "start", "label": "User Question", "type": "input"}]
    edges: list[dict[str, str]] = []

    previous_id = "start"
    for step in execution_result.get("plan", []):
        step_id = f"step_{step['step_id']}"
        label = step["instruction"]
        if step.get("tool_name"):
            label = f"{step['tool_name']}\\n{step['instruction']}"
        if step.get("kind") == "summary":
            label = f"LLM Summary\\n{step['instruction']}"

        nodes.append(
            {"id": step_id, "label": label, "type": step.get("kind", "unknown")}
        )
        edges.append({"from": previous_id, "to": step_id, "label": "next"})
        previous_id = step_id

    nodes.append({"id": "final_answer", "label": "Final Answer", "type": "output"})
    edges.append({"from": previous_id, "to": "final_answer", "label": "result"})

    return {
        "meta": {
            "question": execution_result.get("question", ""),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "nodes": nodes,
        "edges": edges,
        "observations": execution_result.get("observations", []),
        "answer": execution_result.get("answer", ""),
    }


def to_mermaid(graph: dict) -> str:
    """Build mermaid flowchart text from graph JSON."""
    lines = ["flowchart TD"]
    for node in graph["nodes"]:
        safe_label = str(node["label"]).replace('"', "'")
        lines.append(f'    {node["id"]}["{safe_label}"]')
    for edge in graph["edges"]:
        lines.append(f"    {edge['from']} -->|{edge['label']}| {edge['to']}")
    return "\n".join(lines)


def save_execution_graph(
    graph: dict, output_dir: Path = TRACE_OUTPUT_DIR, trace_id: str | None = None
) -> tuple[str, Path]:
    """Persist graph JSON file and return (trace_id, path)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_trace_id = trace_id or uuid4().hex[:12]
    output_file = output_dir / f"{resolved_trace_id}.json"
    output_file.write_text(
        json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resolved_trace_id, output_file


def load_execution_graph(trace_id: str, output_dir: Path = TRACE_OUTPUT_DIR) -> dict:
    """Load persisted graph JSON by trace id."""
    safe_trace_id = trace_id.replace("/", "").replace("..", "")
    path = output_dir / f"{safe_trace_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def build_mermaid_html(graph: dict) -> str:
    """Render an embeddable Mermaid HTML page for a graph."""
    mermaid_text = to_mermaid(graph)
    return f"""<!doctype html>
<html>
<head>
  <meta charset=\"utf-8\" />
  <title>Execution Graph</title>
  <script type=\"module\"> 
    import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@11/dist/mermaid.esm.min.mjs';
    mermaid.initialize({{ startOnLoad: true }});
  </script>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 0; padding: 16px; }}
    .meta {{ margin-bottom: 12px; color: #444; }}
    .mermaid {{ border: 1px solid #ddd; border-radius: 8px; padding: 12px; }}
    pre {{ background: #f6f8fa; padding: 12px; border-radius: 8px; overflow-x: auto; }}
  </style>
</head>
<body>
  <h2>Agent Execution Graph</h2>
  <div class=\"meta\"><strong>Question:</strong> {graph.get("meta", {}).get("question", "")}</div>
  <div class=\"meta\"><strong>Generated At:</strong> {graph.get("meta", {}).get("generated_at", "")}</div>
  <div class=\"mermaid\">{mermaid_text}</div>
  <h3>Observations</h3>
  <pre>{json.dumps(graph.get("observations", []), ensure_ascii=False, indent=2)}</pre>
  <h3>Final Answer</h3>
  <pre>{graph.get("answer", "")}</pre>
</body>
</html>"""
