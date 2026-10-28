"""OPS-5: Postgres backup/restore operator scripts stay runnable and correct.

Functional verification runs in CI against a real PostgreSQL service container
(the ``postgres-restore`` job). Without Docker in the sandbox these tests pin
the scripts' contract instead: bash syntax, dump flags, scratch database,
revision and row-count comparison, and guaranteed cleanup.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BACKUP = ROOT / "scripts" / "pg_backup.sh"
RESTORE = ROOT / "scripts" / "pg_restore_check.sh"


def _bash_syntax(path: Path) -> None:
    result = subprocess.run(["bash", "-n", str(path)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


def test_scripts_exist_are_executable_and_syntax_valid() -> None:
    for path in (BACKUP, RESTORE):
        assert path.is_file()
        assert path.stat().st_mode & 0o111, "скрипт должен быть исполняемым"
        _bash_syntax(path)


def test_backup_uses_plain_sql_dump_without_roles() -> None:
    text = BACKUP.read_text(encoding="utf-8")
    assert "pg_dump" in text
    assert "--no-owner" in text
    assert "gzip" in text
    # Atomic publish so an interrupted run leaves no truncated dump behind.
    assert ".tmp" in text and 'mv "$out.tmp" "$out"' in text
    # The dump lands in a host directory, never inside the database volume.
    assert 'BACKUP_DIR="${BACKUP_DIR:-backups}"' in text


def test_restore_checks_revision_row_counts_and_cleans_up() -> None:
    text = RESTORE.read_text(encoding="utf-8")
    # Scratch database, never the live one.
    assert 'SCRATCH="${SCRATCH_DB:-dora_restore_check}"' in text
    assert "CREATE DATABASE $SCRATCH" in text
    # A restore error must abort instead of being swallowed.
    assert "ON_ERROR_STOP=1" in text
    # The migration identity and every application table are compared.
    assert "alembic_version" in text
    for table in ("jobs", "classification_log", "classification_feedback"):
        assert table in text
    # The scratch database is removed even when a check fails.
    assert "trap cleanup EXIT" in text
    assert "DROP DATABASE IF EXISTS $SCRATCH" in text


def test_backup_directory_is_ignored_by_git() -> None:
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "/backups/" in ignore
