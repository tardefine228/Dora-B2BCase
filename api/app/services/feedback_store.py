"""Persist voluntary feedback without exposing other users' queries.

PostgreSQL is used in Docker (shared persistent database); the SQLite fallback is
for local single-process development. No unauthenticated list endpoint exists:
procurement descriptions may be confidential.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

from app.core.config import settings


def save_feedback(
    query: str, predicted_code: str | None, verdict: str, corrected_code: str | None
) -> str:
    feedback_id = str(uuid4())
    if settings.job_store == "postgres":
        import psycopg

        with psycopg.connect(settings.database_url) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """CREATE TABLE IF NOT EXISTS classification_feedback (
                      id TEXT PRIMARY KEY,
                      created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                      query TEXT NOT NULL, predicted_code TEXT,
                      verdict TEXT NOT NULL, corrected_code TEXT
                    )"""
                )
                cur.execute(
                    """INSERT INTO classification_feedback
                      (id, query, predicted_code, verdict, corrected_code)
                      VALUES (%s, %s, %s, %s, %s)""",
                    (feedback_id, query, predicted_code, verdict, corrected_code),
                )
    else:
        location = Path(settings.job_store_path).with_name("classification_feedback.sqlite")
        location.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(location, timeout=10) as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS classification_feedback (
                  id TEXT PRIMARY KEY,
                  created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                  query TEXT NOT NULL, predicted_code TEXT,
                  verdict TEXT NOT NULL, corrected_code TEXT
                )"""
            )
            conn.execute(
                """INSERT INTO classification_feedback
                  (id, query, predicted_code, verdict, corrected_code)
                  VALUES (?, ?, ?, ?, ?)""",
                (feedback_id, query, predicted_code, verdict, corrected_code),
            )
    return feedback_id
