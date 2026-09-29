"""
database.py
------------
Milestone 2 - Task 5: Meeting Data Model & Database Persistence

Implements the flow:
    Transcript -> AI Processing -> Structured Output -> Validation -> Database

Uses SQLite via the standard library (no extra dependency, zero-config,
works identically on Windows/Mac/Linux) - a good fit for this stage of
the project. The schema is deliberately normalized (participants are a
separate table, not duplicated per action item) so the "duplicate
participant records are avoided" requirement from Task 4 holds at the
persistence layer too, across meetings, not just within one.
"""

import sqlite3
import json
import os
from contextlib import contextmanager
from datetime import datetime

DEFAULT_DB_PATH = os.path.join(os.path.dirname(__file__), "meetings.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS meetings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    filename TEXT NOT NULL,
    transcript TEXT NOT NULL,
    language TEXT,
    audio_duration_seconds REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS participants (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE
);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    summary_text TEXT,
    key_points TEXT,     -- JSON-encoded list
    decisions TEXT,       -- JSON-encoded list
    priorities TEXT,      -- JSON-encoded list
    created_at TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings (id)
);

CREATE TABLE IF NOT EXISTS action_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    meeting_id INTEGER NOT NULL,
    participant_id INTEGER,
    task TEXT NOT NULL,
    deadline TEXT,
    priority TEXT,
    status TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (meeting_id) REFERENCES meetings (id),
    FOREIGN KEY (participant_id) REFERENCES participants (id)
);

CREATE INDEX IF NOT EXISTS idx_action_items_meeting ON action_items (meeting_id);
CREATE INDEX IF NOT EXISTS idx_summaries_meeting ON summaries (meeting_id);
"""


class Database:
    """Thin, explicit persistence layer - no ORM, so the schema and every
    query are easy to read and audit."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._init_schema()

    @contextmanager
    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA foreign_keys = ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_schema(self):
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    # -- Meetings -------------------------------------------------------------

    def create_meeting(self, filename: str, transcript: str, language: str = None,
                        audio_duration_seconds: float = None) -> int:
        """Persist the transcript stage of the pipeline. Returns the new meeting_id."""
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO meetings (filename, transcript, language, audio_duration_seconds, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (filename, transcript, language, audio_duration_seconds, datetime.now().isoformat()),
            )
            return cur.lastrowid

    def get_meeting(self, meeting_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM meetings WHERE id = ?", (meeting_id,)).fetchone()
            return dict(row) if row else None

    def list_meetings(self) -> list:
        with self._connect() as conn:
            rows = conn.execute("SELECT id, filename, language, created_at FROM meetings ORDER BY id DESC").fetchall()
            return [dict(r) for r in rows]

    # -- Participants (deduplicated by UNIQUE name constraint) ----------------

    def get_or_create_participant(self, name: str) -> int:
        """
        Insert-or-fetch a participant by name. The UNIQUE constraint on
        `name` is the hard guarantee against duplicate participant records
        at the database level - even if two different meetings' processing
        both try to create "Ravi Kumar", only one row will ever exist.
        """
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM participants WHERE name = ?", (name,)).fetchone()
            if row:
                return row["id"]
            cur = conn.execute("INSERT INTO participants (name) VALUES (?)", (name,))
            return cur.lastrowid

    def list_participants(self) -> list:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM participants ORDER BY name").fetchall()
            return [dict(r) for r in rows]

    # -- Summaries --------------------------------------------------------------

    def save_summary(self, meeting_id: int, summary_text: str, key_points: list,
                      decisions: list, priorities: list) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO summaries (meeting_id, summary_text, key_points, decisions, priorities, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (meeting_id, summary_text, json.dumps(key_points), json.dumps(decisions),
                 json.dumps(priorities), datetime.now().isoformat()),
            )
            return cur.lastrowid

    def get_summary(self, meeting_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM summaries WHERE meeting_id = ? ORDER BY id DESC LIMIT 1", (meeting_id,)
            ).fetchone()
            if not row:
                return None
            result = dict(row)
            result["key_points"] = json.loads(result["key_points"] or "[]")
            result["decisions"] = json.loads(result["decisions"] or "[]")
            result["priorities"] = json.loads(result["priorities"] or "[]")
            return result

    # -- Action items (linked to meeting + participant) --------------------------

    def save_action_item(self, meeting_id: int, participant_id: int, task: str,
                          deadline: str, priority: str, status: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO action_items (meeting_id, participant_id, task, deadline, priority, status, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (meeting_id, participant_id, task, deadline, priority, status, datetime.now().isoformat()),
            )
            return cur.lastrowid

    def get_action_items(self, meeting_id: int) -> list:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT action_items.*, participants.name AS participant_name
                   FROM action_items
                   LEFT JOIN participants ON action_items.participant_id = participants.id
                   WHERE action_items.meeting_id = ?
                   ORDER BY action_items.id""",
                (meeting_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    # -- Milestone 3 read helpers (Task 1: granular per-field retrieval) --------
    # Thin wrappers over the existing tables/queries above - no schema change,
    # no duplicate storage. They exist so api.py can expose one focused
    # endpoint per data type (GET /meetings/{id}/transcript, /decisions, etc.)
    # without every caller having to know the join logic themselves.

    def get_transcript(self, meeting_id: int) -> str:
        meeting = self.get_meeting(meeting_id)
        return meeting["transcript"] if meeting else None

    def get_decisions(self, meeting_id: int) -> list:
        summary = self.get_summary(meeting_id)
        return summary["decisions"] if summary else []

    def get_participants_for_meeting(self, meeting_id: int) -> list:
        """Distinct participant names linked to this meeting via its action
        items - the existing schema has no separate meeting<->participant
        join table, so this reuses the same relationship Task 4 (Milestone 2)
        already established."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT participants.id, participants.name
                   FROM action_items
                   JOIN participants ON action_items.participant_id = participants.id
                   WHERE action_items.meeting_id = ?
                   ORDER BY participants.name""",
                (meeting_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_deadlines_for_meeting(self, meeting_id: int) -> list:
        """Distinct, non-empty deadlines linked to this meeting via its
        action items."""
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT DISTINCT deadline FROM action_items
                   WHERE meeting_id = ? AND deadline IS NOT NULL AND TRIM(deadline) != ''
                   ORDER BY deadline""",
                (meeting_id,),
            ).fetchall()
            return [r["deadline"] for r in rows]

    # -- Full meeting read (used by API / UI) --------------------------------

    def get_meeting_full(self, meeting_id: int) -> dict:
        """Returns the meeting plus its summary and action items in one call -
        exactly the shape the Streamlit UI and API responses need."""
        meeting = self.get_meeting(meeting_id)
        if not meeting:
            return None
        return {
            "meeting": meeting,
            "summary": self.get_summary(meeting_id),
            "action_items": self.get_action_items(meeting_id),
        }
