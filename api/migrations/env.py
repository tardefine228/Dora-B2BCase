"""Alembic environment: URL from the application settings, offline mode supported.

The migrations are the *only* description of the PostgreSQL schema: the API no
longer issues `CREATE TABLE IF NOT EXISTS` at runtime, it runs `upgrade head`
(app/migrations_runner.py). Keeping one description is the point — the old
runtime DDL and the real schema could drift, and nothing detected it.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

config = context.config


def _database_url() -> str:
    """``sqlalchemy.url`` from the caller, else ``DATABASE_URL``."""
    url = (config.get_main_option("sqlalchemy.url") or "").strip()
    if url:
        return url
    url = os.getenv("DATABASE_URL", "").strip()
    if not url:
        raise RuntimeError(
            "DATABASE_URL не задан: миграции работают только с PostgreSQL "
            "(JOB_STORE=postgres). Для локального sqlite-режима они не нужны."
        )
    return url


def run_migrations_offline() -> None:
    """Render SQL without a database: `alembic upgrade head --sql`."""
    context.configure(
        url=_database_url(),
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    engine = create_engine(_database_url(), poolclass=pool.NullPool)
    with engine.connect() as connection:
        context.configure(connection=connection, target_metadata=None)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
