"""Call log: outcome, transcript and optional recording paths in the plugin's SQLite DB."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS calls (
    id TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    number TEXT NOT NULL,
    callee TEXT,
    goal TEXT,
    status TEXT NOT NULL,
    state TEXT,
    dialed INTEGER NOT NULL DEFAULT 0,
    duration_s REAL,
    outcome_json TEXT,
    transcript_json TEXT,
    brief_json TEXT,
    recording TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS calls_created ON calls(created_at);
"""


class CallLog:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @classmethod
    def at_path(cls, path: Path) -> "CallLog":
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite3.connect(str(path), check_same_thread=False))

    def start(self, call_id: str, number: str, callee: str, goal: str, brief: dict[str, Any]) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO calls (id, created_at, number, callee, goal, status, state, brief_json) "
                "VALUES (?, ?, ?, ?, ?, 'in_progress', 'created', ?)",
                (call_id, time.time(), number, callee, goal, json.dumps(brief, ensure_ascii=False)))
            self._conn.commit()

    def mark_dialed(self, call_id: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE calls SET dialed = 1, state = 'dialing' WHERE id = ?", (call_id,))
            self._conn.commit()

    def finish(self, call_id: str, *, status: str, state: str, duration_s: float, outcome: dict[str, Any],
               transcript: list[dict[str, Any]], recording: Optional[str] = None,
               error: Optional[str] = None) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE calls SET status = ?, state = ?, duration_s = ?, outcome_json = ?, transcript_json = ?, "
                "recording = ?, error = ? WHERE id = ?",
                (status, state, duration_s, json.dumps(outcome, ensure_ascii=False),
                 json.dumps(transcript, ensure_ascii=False), recording, error, call_id))
            self._conn.commit()

    def calls_since(self, since_ts: float) -> int:
        """Dialed calls since *since_ts* (for the hourly rate limit)."""
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM calls WHERE dialed = 1 AND created_at >= ?",
                                     (since_ts,)).fetchone()
        return int(row[0]) if row else 0

    def recent(self, limit: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, created_at, number, callee, goal, status, duration_s, outcome_json FROM calls "
                "ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            outcome = json.loads(r[7]) if r[7] else {}
            out.append({"id": r[0], "created_at": r[1], "number": r[2], "callee": r[3], "goal": r[4],
                        "status": r[5], "duration_s": r[6], "summary": outcome.get("summary", "")})
        return out

    def get(self, call_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            r = self._conn.execute(
                "SELECT id, created_at, number, callee, goal, status, state, duration_s, outcome_json, "
                "transcript_json, recording, error FROM calls WHERE id = ? OR id LIKE ? "
                "ORDER BY created_at DESC LIMIT 1", (call_id, call_id + "%")).fetchone()
        if not r:
            return None
        return {"id": r[0], "created_at": r[1], "number": r[2], "callee": r[3], "goal": r[4], "status": r[5],
                "state": r[6], "duration_s": r[7], "outcome": json.loads(r[8]) if r[8] else {},
                "transcript": json.loads(r[9]) if r[9] else [], "recording": r[10], "error": r[11]}
