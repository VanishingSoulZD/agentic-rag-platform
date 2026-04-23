"""Path-related helpers for safe output file handling."""

from __future__ import annotations

from pathlib import Path


def ensure_output_parent(path: str | Path) -> Path:
    """Ensure the parent directory exists for a file path and return Path."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
