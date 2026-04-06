"""SQLite DB query tool utilities."""

from __future__ import annotations

import sqlite3
from pathlib import Path

DEFAULT_DB_PATH = Path("app/langchain_tools/local_users.db")


def initialize_local_user_db(db_path: Path = DEFAULT_DB_PATH) -> None:
    """Create local sqlite DB with demo users if it does not exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                city TEXT NOT NULL
            )
            """
        )

        cursor.execute("SELECT COUNT(*) FROM users")
        count = cursor.fetchone()[0]
        if count == 0:
            cursor.executemany(
                "INSERT INTO users(name, city) VALUES (?, ?)",
                [
                    ("Alice", "Taipei"),
                    ("Bob", "Beijing"),
                    ("Carol", "Shanghai"),
                ],
            )
        conn.commit()


def query_local_user_db(query: str, db_path: Path = DEFAULT_DB_PATH) -> str:
    """Execute read-only SQL query against local users DB."""
    sql = query.strip()
    if not sql:
        raise ValueError("SQL query must not be empty.")

    normalized = sql.lower()
    if not normalized.startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")

    initialize_local_user_db(db_path)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()

    return str(rows)
