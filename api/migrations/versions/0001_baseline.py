"""Baseline: the PostgreSQL schema the API shipped before migrations existed.

Revision ID: 0001_baseline
Revises: —
Create Date: 2026-10-25

Everything here was previously issued as `CREATE TABLE IF NOT EXISTS` by
`PostgresJobStore.__init__` and `feedback_store.save_feedback`. The statements
stay idempotent on purpose: a database created by an older release has no
`alembic_version` row and no `comment` column, so `upgrade head` replays this
baseline over it, finds the tables already in place, and only applies what is
missing. No manual `alembic stamp` on the target machine.
"""

from __future__ import annotations

from alembic import op

revision = "0001_baseline"
down_revision = None
branch_labels = None
depends_on = None

_STATEMENTS = (
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        items JSONB NOT NULL DEFAULT '[]'::jsonb,
        completed INTEGER NOT NULL DEFAULT 0,
        results JSONB NOT NULL DEFAULT '[]'::jsonb,
        total INTEGER,
        indexed INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, updated_at)",
    """
    CREATE TABLE IF NOT EXISTS classification_log (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL DEFAULT now(),
        source TEXT NOT NULL,
        query TEXT NOT NULL,
        code TEXT,
        status TEXT NOT NULL,
        confidence REAL,
        latency_ms REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_classification_log_ts ON classification_log(ts)",
    """
    CREATE TABLE IF NOT EXISTS classification_feedback (
        id TEXT PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        query TEXT NOT NULL,
        predicted_code TEXT,
        verdict TEXT NOT NULL,
        corrected_code TEXT,
        comment TEXT
    )
    """,
    # The export filters by period and by verdict type (GET /admin/feedback/export).
    "CREATE INDEX IF NOT EXISTS idx_feedback_created_at "
    "ON classification_feedback(created_at)",
)


def upgrade() -> None:
    for statement in _STATEMENTS:
        op.execute(statement)


def downgrade() -> None:
    for table in ("classification_feedback", "classification_log", "jobs"):
        op.execute(f"DROP TABLE IF EXISTS {table}")
