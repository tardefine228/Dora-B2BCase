"""Persist voluntary feedback and export it for training (BE-2 / ML-4).

PostgreSQL is used in Docker (shared persistent database); the SQLite fallback is
for local single-process development. No unauthenticated list endpoint exists:
procurement descriptions may be confidential, so reading the log requires the
admin key (``GET /admin/feedback/export``).

The export exists for one reason: ``ml.hard_negatives.load_feedback`` turns
specialist verdicts into training pairs, and until the log could be read out of
the database that mechanism had no fuel. The column set below is exactly what the
loader reads, so an export goes straight into training::

    python -m ml.train_reranker --output models/reranker.json --feedback export.csv
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from app.core.config import settings

CSV_DELIMITER = ";"
# `proposed_code` (not the API field name `predicted_code`) is what the training
# loader expects; `created_at` is extra and harmless — the loader ignores columns
# it does not know.
EXPORT_COLUMNS = (
    "query",
    "proposed_code",
    "verdict",
    "correct_code",
    "comment",
    "created_at",
)
# Excel on a Russian locale expects ";" separators and a BOM in a UTF-8 file.
FEEDBACK_BOM = "\ufeff"

# The PostgreSQL table is created by the Alembic migrations (api/migrations,
# revision 0001_baseline); the store must not issue DDL of its own, otherwise the
# schema has two descriptions that can drift. SQLite has no migrations, so it
# keeps its inline (idempotent) CREATE and an in-place ADD COLUMN for databases
# written by an older release.
_SQLITE_TABLE = """
CREATE TABLE IF NOT EXISTS classification_feedback (
  id TEXT PRIMARY KEY,
  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
  query TEXT NOT NULL, predicted_code TEXT,
  verdict TEXT NOT NULL, corrected_code TEXT, comment TEXT
)
"""


@dataclass(frozen=True)
class FeedbackRow:
    """One stored verdict, in the shape the training loader reads."""

    query: str
    proposed_code: str = ""
    verdict: str = ""
    correct_code: str = ""
    comment: str = ""
    created_at: str = ""

    def as_dict(self) -> dict[str, str]:
        return {
            "query": self.query,
            "proposed_code": self.proposed_code,
            "verdict": self.verdict,
            "correct_code": self.correct_code,
            "comment": self.comment,
            "created_at": self.created_at,
        }


def _sqlite_path() -> Path:
    return Path(settings.job_store_path).with_name("classification_feedback.sqlite")


def _is_postgres() -> bool:
    return settings.job_store == "postgres"


def save_feedback(
    query: str,
    predicted_code: str | None,
    verdict: str,
    corrected_code: str | None,
    comment: str | None = None,
) -> str:
    feedback_id = str(uuid4())
    if _is_postgres():
        import psycopg

        with psycopg.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """INSERT INTO classification_feedback
                      (id, query, predicted_code, verdict, corrected_code, comment)
                      VALUES (%s, %s, %s, %s, %s, %s)""",
                    (feedback_id, query, predicted_code, verdict, corrected_code, comment),
                )
    else:
        location = _sqlite_path()
        location.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(location, timeout=10) as conn:
            conn.execute(_SQLITE_TABLE)
            _sqlite_add_comment_column(conn)
            conn.execute(
                """INSERT INTO classification_feedback
                  (id, query, predicted_code, verdict, corrected_code, comment)
                  VALUES (?, ?, ?, ?, ?, ?)""",
                (feedback_id, query, predicted_code, verdict, corrected_code, comment),
            )
    return feedback_id


def _sqlite_add_comment_column(conn: sqlite3.Connection) -> None:
    """SQLite has no ``ADD COLUMN IF NOT EXISTS``: ask the schema first."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(classification_feedback)")}
    if "comment" not in columns:
        conn.execute("ALTER TABLE classification_feedback ADD COLUMN comment TEXT")


def list_feedback(
    *,
    since: datetime | None = None,
    until: datetime | None = None,
    verdict: str | None = None,
    limit: int | None = None,
) -> list[FeedbackRow]:
    """Read stored verdicts (oldest first) for export."""
    where: list[str] = []
    params: list[object] = []
    if since is not None:
        where.append("created_at >= " + ("%s" if _is_postgres() else "datetime(?)"))
        params.append(since)
    if until is not None:
        where.append("created_at <= " + ("%s" if _is_postgres() else "datetime(?)"))
        params.append(until)
    if verdict:
        where.append("verdict = " + ("%s" if _is_postgres() else "?"))
        params.append(verdict)
    clause = f" WHERE {' AND '.join(where)}" if where else ""
    order = " ORDER BY created_at"
    limit_clause = ""
    if limit is not None:
        limit_clause = " LIMIT %s" if _is_postgres() else " LIMIT ?"
        params.append(int(limit))
    sql = (
        "SELECT query, predicted_code, verdict, corrected_code, comment, created_at"
        f" FROM classification_feedback{clause}{order}{limit_clause}"
    )

    if _is_postgres():
        import psycopg

        with psycopg.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(sql, params)
                rows = cur.fetchall()
        return [_row(values) for values in rows]

    location = _sqlite_path()
    if not location.is_file():
        return []
    with sqlite3.connect(location, timeout=10) as conn:
        conn.execute(_SQLITE_TABLE)
        _sqlite_add_comment_column(conn)
        rows = conn.execute(sql, params).fetchall()
    return [_row(values) for values in rows]


def _row(values: tuple) -> FeedbackRow:
    query, predicted, verdict, corrected, comment, created_at = values
    return FeedbackRow(
        query=query or "",
        proposed_code=predicted or "",
        verdict=verdict or "",
        correct_code=corrected or "",
        comment=comment or "",
        created_at=str(created_at) if created_at is not None else "",
    )


def export_feedback(rows: list[FeedbackRow], fmt: str = "csv") -> str:
    """Serialise verdicts in a format ``ml.hard_negatives`` can read back."""
    if fmt == "jsonl":
        return "\n".join(
            json.dumps(row.as_dict(), ensure_ascii=False) for row in rows
        ) + ("\n" if rows else "")
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer, fieldnames=list(EXPORT_COLUMNS), delimiter=CSV_DELIMITER, lineterminator="\r\n"
    )
    writer.writeheader()
    for row in rows:
        writer.writerow(row.as_dict())
    return FEEDBACK_BOM + buffer.getvalue()
