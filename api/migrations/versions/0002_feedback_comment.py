"""Add the specialist's free-text comment to the feedback log.

Revision ID: 0002_feedback_comment
Revises: 0001_baseline
Create Date: 2026-10-25

`POST /feedback` accepts a comment and the training export carries it, but a
database created before this change has no column for it. The column is added in
place: the existing verdicts are training data, so the table is never rebuilt.
"""

from __future__ import annotations

from alembic import op

revision = "0002_feedback_comment"
down_revision = "0001_baseline"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        "ALTER TABLE classification_feedback ADD COLUMN IF NOT EXISTS comment TEXT"
    )


def downgrade() -> None:
    op.execute("ALTER TABLE classification_feedback DROP COLUMN IF EXISTS comment")
