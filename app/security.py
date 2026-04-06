"""Input sanitization and tool-use policy helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

DEFAULT_MAX_INPUT_CHARS = 1200

# 常见 prompt-injection 指令片段：统一替换为空白，降低模型遵从风险。
DANGEROUS_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions?",
    r"disregard\s+(all\s+)?(the\s+)?(above|previous)\s+instructions?",
    r"system\s+prompt",
    r"developer\s+message",
    r"reveal\s+(your\s+)?(hidden|internal)\s+instructions?",
    r"tool\s*:\s*",
    r"call\s+the\s+tool",
    r"execute\s+shell",
    r"curl\s+http",
]


@dataclass(frozen=True)
class ToolUsePolicy:
    """Simple allow/deny policy for agent tools."""

    denied_tools: set[str] = field(default_factory=set)

    def enforce(self, tool_name: str) -> None:
        if tool_name in self.denied_tools:
            raise PermissionError(f"Tool is blocked by policy: {tool_name}")


def sanitize_user_input(text: str, max_chars: int = DEFAULT_MAX_INPUT_CHARS) -> str:
    """Limit input length and remove common prompt-injection instructions."""

    if not text:
        return ""

    cleaned = text.strip()
    if len(cleaned) > max_chars:
        cleaned = cleaned[:max_chars]

    for pattern in DANGEROUS_PATTERNS:
        cleaned = re.sub(pattern, " ", cleaned, flags=re.IGNORECASE)

    # 压缩多空格，避免替换后产生噪音。
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned
