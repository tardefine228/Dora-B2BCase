"""Schema migrations own the PostgreSQL schema (BE-5).

Why these tests exist without a database in CI: the thing that broke before was
not a query, it was the *absence of an owner*. `CREATE TABLE IF NOT EXISTS` lived
in the store and in the feedback module, so a column change had nowhere to happen
and an old database kept the old shape silently.

Two guards that run everywhere:

* the migration chain is linear and ends at the revision the application declares
  (`REVISION_HEAD`) — adding a migration without wiring it is a red test;
* the rendered SQL parses as PostgreSQL and contains every table, column and index
  the code queries. A typo in DDL or a forgotten table shows up here instead of on
  the target machine.

The real end-to-end check (`alembic upgrade head` against a live PostgreSQL) runs
in `test_postgres_store.py` when `TEST_DATABASE_URL` is set.
"""

from __future__ import annotations

import importlib.util
import io
from contextlib import redirect_stdout
from pathlib import Path

import pytest
import sqlglot
from app.migrations_runner import REVISION_HEAD, alembic_config

API_DIR = Path(__file__).resolve().parents[1]
VERSIONS_DIR = API_DIR / "migrations" / "versions"


def _versions() -> dict[str, dict]:
    """Parse the version scripts by import (no Alembic environment needed)."""
    found: dict[str, dict] = {}
    for path in sorted(VERSIONS_DIR.glob("*.py")):
        spec = importlib.util.spec_from_file_location(f"_migration_{path.stem}", path)
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert module.revision not in found, f"дубль revision {module.revision}"
        found[module.revision] = {
            "down": module.down_revision,
            "path": path,
        }
    return found


def test_migration_chain_is_linear_and_ends_at_the_declared_head() -> None:
    versions = _versions()
    assert versions, "в api/migrations/versions нет ни одной миграции"

    heads = [revision for revision, meta in versions.items() if meta["down"] is None]
    assert len(heads) == 1, f"должна быть ровно одна базовая миграция, найдено: {heads}"
    assert heads[0] == "0001_baseline"

    # Walk forward from the base: every migration is reachable exactly once.
    children: dict[str, list[str]] = {}
    for revision, meta in versions.items():
        if meta["down"] is not None:
            children.setdefault(meta["down"], []).append(revision)
    for parent, kids in children.items():
        assert len(kids) == 1, f"разветвление после {parent}: {kids}"

    current = heads[0]
    seen = [current]
    while current in children:
        current = children[current][0]
        seen.append(current)
    assert len(seen) == len(versions), "часть миграций недостижима из базовой"
    assert current == REVISION_HEAD, (
        f"голова цепочки {current!r}, а приложение объявляет {REVISION_HEAD!r}: "
        "обновите REVISION_HEAD в app/migrations_runner.py"
    )


def _rendered_sql() -> str:
    """`alembic upgrade head --sql` — compiles the migrations without a database."""
    from alembic import command

    config = alembic_config("postgresql://user:pass@localhost:5432/dora")
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        command.upgrade(config, "head", sql=True)
    return buffer.getvalue()


def test_offline_render_is_valid_postgres_sql() -> None:
    sql = _rendered_sql()
    statements = [statement for statement in sqlglot.parse(sql, dialect="postgres") if statement]
    assert statements, "пустой рендер: alembic не увидел миграций"


def test_rendered_sql_creates_every_table_the_code_queries() -> None:
    sql = _rendered_sql()
    # Tables referenced by PostgresJobStore and feedback_store. A missing one is a
    # runtime "relation does not exist" on a fresh deployment.
    for table in ("jobs", "classification_log", "classification_feedback"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in sql, table
    # Columns the API writes and reads back.
    for column in ("predicted_code", "corrected_code", "comment", "latency_ms"):
        assert column in sql, column
    # Indexes: job status for the queue, ts for the dashboard, created_at for the
    # feedback export's period filter.
    for index in ("idx_jobs_status", "idx_classification_log_ts", "idx_feedback_created_at"):
        assert index in sql, index


def test_migrations_are_idempotent_so_an_old_database_upgrades_in_place() -> None:
    """No manual `alembic stamp`: every DDL statement tolerates an existing object.

    A database created by a release without migrations already has the tables and
    no `alembic_version` row. `upgrade head` must be able to replay the baseline
    over it, which only works while the statements are `IF NOT EXISTS`.
    """
    sql = _rendered_sql()
    # `alembic_version` is Alembic's own bookkeeping table, not ours.
    ours = "\n".join(
        line for line in sql.splitlines() if "alembic_version" not in line
    )
    for line in ours.splitlines():
        stripped = line.strip()
        if stripped.startswith("CREATE TABLE") or stripped.startswith("CREATE INDEX"):
            assert "IF NOT EXISTS" in stripped, stripped
        if stripped.startswith("ALTER TABLE") and "ADD COLUMN" in stripped:
            assert "IF NOT EXISTS" in stripped, stripped


def test_alembic_config_points_at_the_repository_scripts() -> None:
    config = alembic_config("postgresql://user:pass@localhost:5432/dora")
    script_location = Path(config.get_main_option("script_location"))
    assert script_location == API_DIR / "migrations"
    assert (script_location / "env.py").is_file()
    assert config.get_main_option("sqlalchemy.url").startswith("postgresql://")


def _ddl_literals(path: Path, keywords: tuple[str, ...]) -> list[str]:
    """String constants of a module that contain DDL, comments excluded."""
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            text = node.value.strip()
            if any(text.upper().startswith(keyword) for keyword in keywords):
                found.append(text)
    return found


def test_store_does_not_issue_its_own_ddl() -> None:
    """The schema has exactly one description: the migrations.

    This is the regression that started BE-5 — DDL living in the store meant an
    old database silently kept an old shape while the code assumed a new one.
    Comments are ignored: only real SQL string constants count.
    """
    store = API_DIR / "app" / "job_store_postgres.py"
    assert _ddl_literals(store, ("CREATE TABLE", "CREATE INDEX", "ALTER TABLE")) == []
    assert "upgrade_to_head" in store.read_text(encoding="utf-8")

    feedback = API_DIR / "app" / "services" / "feedback_store.py"
    create_statements = _ddl_literals(feedback, ("CREATE TABLE",))
    # SQLite keeps its inline schema (no migrations there); PostgreSQL must not —
    # exactly one CREATE, and it is the SQLite one.
    assert len(create_statements) == 1, create_statements
    assert "DEFAULT CURRENT_TIMESTAMP" in create_statements[0]
    alters = _ddl_literals(feedback, ("ALTER TABLE",))
    assert len(alters) == 1 and "PRAGMA" not in alters[0]


def test_offline_render_requires_a_database_url(monkeypatch: pytest.MonkeyPatch) -> None:
    from alembic import command

    monkeypatch.delenv("DATABASE_URL", raising=False)
    config = alembic_config("")
    with pytest.raises(RuntimeError, match="DATABASE_URL"):
        with redirect_stdout(io.StringIO()):
            command.upgrade(config, "head", sql=True)
