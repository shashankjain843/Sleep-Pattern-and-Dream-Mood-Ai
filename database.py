"""
database.py — SQLite session persistence layer.
Schema is intentionally simple so it can be migrated to Postgres later
by swapping sqlite3 for psycopg2 and adjusting the connection string.
"""
import sqlite3
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

DB_PATH = Path("sleep_ai.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    with get_connection() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT    NOT NULL,
                user_id         TEXT    NOT NULL DEFAULT 'anonymous',
                input_source    TEXT,
                sleep_metrics   TEXT,
                mood_metrics    TEXT,
                suggestions     TEXT
            )
        """)
        conn.commit()
    logger.info("Database initialised at %s", DB_PATH)


def save_session(
    user_id: str,
    input_source: str,
    sleep_metrics: dict,
    mood_metrics: dict,
    suggestions: list,
) -> int:
    """Persist one analysis session and return its row id."""
    ts = datetime.now(timezone.utc).isoformat()
    with get_connection() as conn:
        cur = conn.execute(
            """INSERT INTO sessions
               (timestamp, user_id, input_source, sleep_metrics, mood_metrics, suggestions)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                ts,
                user_id,
                input_source,
                json.dumps(sleep_metrics),
                json.dumps(mood_metrics),
                json.dumps(suggestions),
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
    logger.debug("Session %d saved for user '%s'", row_id, user_id)
    return row_id


def get_sessions(user_id: str = "anonymous", limit: int = 50) -> list[dict]:
    """Return the last *limit* sessions for a user, newest first."""
    with get_connection() as conn:
        rows = conn.execute(
            """SELECT id, timestamp, input_source, sleep_metrics, mood_metrics, suggestions
               FROM sessions
               WHERE user_id = ?
               ORDER BY id DESC
               LIMIT ?""",
            (user_id, limit),
        ).fetchall()

    results = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "timestamp": row["timestamp"],
                "input_source": row["input_source"],
                "sleep_metrics": json.loads(row["sleep_metrics"] or "{}"),
                "mood_metrics": json.loads(row["mood_metrics"] or "{}"),
                "suggestions": json.loads(row["suggestions"] or "[]"),
            }
        )
    return results
