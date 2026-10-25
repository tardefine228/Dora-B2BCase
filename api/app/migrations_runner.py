"""Apply the PostgreSQL schema migrations at startup.

`JOB_STORE=postgres` used to mean "the store creates its own tables with CREATE
TABLE IF NOT EXISTS". That worked until a column had to change: there was no
record of what the schema was, nothing owned the transitions, and an old
database silently kept the old shape. Migrations now own the schema and the
store runs them before its first query — so `docker compose up` on an existing
volume upgrades in place, and there is nothing to type by hand.

SQLite deployments (`JOB_STORE=sqlite`, the default for local runs) are
untouched: their schema is created inline by `SqliteJobStore` and needs no
migrations.

Reviewing the SQL without a database::

    DATABASE_URL=postgresql://any PYTHONPATH=api \\
        alembic -c api/alembic.ini upgrade head --sql
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("dora.migrations")

REVISION_HEAD = "0002_feedback_comment"


def _base_dir() -> Path:
    """``api/`` in the source tree, ``/app`` in the image (same relative layout)."""
    return Path(__file__).resolve().parents[1]


def alembic_config(database_url: str, *, base_dir: Path | None = None):
    """Alembic config pointing at this checkout's ``alembic.ini`` and scripts."""
    from alembic.config import Config

    base = base_dir or _base_dir()
    ini = base / "alembic.ini"
    if not ini.is_file():
        raise RuntimeError(
            f"alembic.ini не найден: {ini}. Миграции обязательны для JOB_STORE=postgres; "
            "в образе файл лежит в /app/alembic.ini (см. api/Dockerfile)."
        )
    config = Config(str(ini))
    config.set_main_option("script_location", str(base / "migrations"))
    config.set_main_option("sqlalchemy.url", database_url)
    return config


def upgrade_to_head(database_url: str) -> None:
    """Bring the database to the latest revision (idempotent)."""
    from alembic import command

    logger.info("Применяю миграции схемы (head=%s)", REVISION_HEAD)
    command.upgrade(alembic_config(database_url), "head")


def current_revision(database_url: str) -> str | None:
    """Revision recorded in the database, ``None`` on a database without alembic."""
    from alembic.runtime.migration import MigrationContext
    from sqlalchemy import create_engine

    engine = create_engine(database_url)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()
