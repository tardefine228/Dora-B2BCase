"""Retention for the audit and feedback logs (BE-5).

CI has no PostgreSQL, so the connection is faked: what is asserted here is the
*decision* — which table is trimmed, with what interval, what is skipped and what
an operator sees. The SQL itself is rendered from constants only (table and
column names are never taken from input), which is why the fake is safe to use
as a contract.
"""

from __future__ import annotations

import sys
import types

import pytest
from app import maintenance
from app.core.config import settings
from app.main import app
from app.schemas import MaintenanceReport
from fastapi.testclient import TestClient


class FakeCursor:
    def __init__(self, rowcounts: list[int] | None = None, counts: list[int] | None = None):
        self.executed: list[tuple[str, tuple]] = []
        self._rowcounts = list(rowcounts or [7, 3])
        self._counts = list(counts or [12, 4])
        self.rowcount = 0

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.executed.append((sql.strip(), tuple(params or ())))
        if sql.strip().startswith("DELETE"):
            self.rowcount = self._rowcounts.pop(0) if self._rowcounts else 0
        return self

    def fetchone(self):
        return (self._counts.pop(0) if self._counts else 0,)


class FakeConnection:
    def __init__(self, cursor: FakeCursor):
        self._cursor = cursor

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def cursor(self):
        return self._cursor


@pytest.fixture()
def fake_postgres(monkeypatch):
    """PostgreSQL mode with a recording connection instead of a real server."""
    cursor = FakeCursor()
    module = types.SimpleNamespace(
        connect=lambda *_args, **_kwargs: FakeConnection(cursor)
    )
    monkeypatch.setitem(sys.modules, "psycopg", module)
    monkeypatch.setattr(settings, "job_store", "postgres")
    monkeypatch.setattr(settings, "database_url", "postgresql://fake/db")
    monkeypatch.setattr(settings, "audit_retention_days", 90)
    monkeypatch.setattr(settings, "feedback_retention_days", 365)
    return cursor


def test_purge_skips_non_postgres_backends(monkeypatch) -> None:
    monkeypatch.setattr(settings, "job_store", "sqlite")
    report = maintenance.purge_expired()
    assert report.skipped == "not postgres"
    assert report.deleted == 0
    # `dry_run` must survive the early return, otherwise the API answers
    # `dry_run: false` to a request that asked for a dry run (found live).
    assert maintenance.purge_expired(dry_run=True).dry_run is True


def test_purge_is_disabled_when_both_retentions_are_zero(monkeypatch) -> None:
    monkeypatch.setattr(settings, "job_store", "postgres")
    monkeypatch.setattr(settings, "audit_retention_days", 0)
    monkeypatch.setattr(settings, "feedback_retention_days", 0)
    report = maintenance.purge_expired()
    assert report.skipped == "retention disabled"
    assert report.deleted == 0
    assert maintenance.purge_expired(dry_run=True).dry_run is True


def test_purge_deletes_from_both_logs_with_the_configured_intervals(fake_postgres) -> None:
    report = maintenance.purge_expired()

    statements = [sql for sql, _params in fake_postgres.executed]
    assert len(statements) == 2
    assert statements[0].startswith("DELETE FROM classification_log")
    assert statements[1].startswith("DELETE FROM classification_feedback")
    assert [params for _sql, params in fake_postgres.executed] == [(90,), (365,)]
    assert report.audit_deleted == 7
    assert report.feedback_deleted == 3
    assert report.deleted == 10


def test_dry_run_counts_without_deleting(fake_postgres) -> None:
    report = maintenance.purge_expired(dry_run=True)

    statements = [sql for sql, _params in fake_postgres.executed]
    assert statements == [
        "SELECT count(*) FROM classification_log WHERE ts < now() - make_interval(days => %s)",
        "SELECT count(*) FROM classification_feedback "
        "WHERE created_at < now() - make_interval(days => %s)",
    ]
    assert report.dry_run is True
    assert report.audit_deleted == 12
    assert report.feedback_deleted == 4


def test_feedback_is_kept_when_only_the_audit_log_has_a_retention(
    fake_postgres, monkeypatch
) -> None:
    monkeypatch.setattr(settings, "feedback_retention_days", 0)
    report = maintenance.purge_expired()
    assert [sql for sql, _params in fake_postgres.executed] == [
        "DELETE FROM classification_log WHERE ts < now() - make_interval(days => %s)"
    ]
    assert report.feedback_deleted == 0


def test_sql_only_accepts_the_two_known_tables_and_a_positive_depth() -> None:
    with pytest.raises(ValueError, match="Неизвестная таблица"):
        maintenance.delete_older_than_sql("users", "id", 30)
    with pytest.raises(ValueError, match="Неизвестная таблица"):
        maintenance.count_older_than_sql("classification_log", "created_at", 30)
    with pytest.raises(ValueError, match="не меньше одного дня"):
        maintenance.delete_older_than_sql("classification_log", "ts", 0)


def test_admin_endpoint_requires_the_key(monkeypatch) -> None:
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    client = TestClient(app)
    assert client.post("/admin/maintenance/purge").status_code == 403


def test_admin_endpoint_reports_what_it_did(monkeypatch) -> None:
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")
    called: dict = {}

    def fake_purge(*, dry_run: bool = False):
        called["dry_run"] = dry_run
        return maintenance.PurgeReport(
            audit_deleted=5, feedback_deleted=0, audit_days=90, feedback_days=0, dry_run=dry_run
        )

    monkeypatch.setattr("app.main.purge_expired", fake_purge)
    client = TestClient(app)

    response = client.post(
        "/admin/maintenance/purge?dry_run=true", headers={"X-Admin-API-Key": "test-secret"}
    )
    assert response.status_code == 200, response.text
    body = MaintenanceReport(**response.json())
    assert body.audit_deleted == 5
    assert body.audit_retention_days == 90
    assert body.dry_run is True
    assert called["dry_run"] is True


def test_admin_endpoint_reports_storage_failure_as_503(monkeypatch) -> None:
    monkeypatch.setattr(settings, "admin_api_key", "test-secret")

    def boom(*_args, **_kwargs):
        raise RuntimeError("connection refused")

    monkeypatch.setattr("app.main.purge_expired", boom)
    client = TestClient(app)
    response = client.post(
        "/admin/maintenance/purge", headers={"X-Admin-API-Key": "test-secret"}
    )
    assert response.status_code == 503


def test_maintenance_loop_runs_immediately_and_keeps_going(monkeypatch) -> None:
    """Housekeeping starts at boot, not after the first full interval."""
    import asyncio

    calls: list[int] = []

    def fake_purge(*, dry_run: bool = False):
        calls.append(1)
        if len(calls) >= 2:
            raise RuntimeError("второй запуск падает — цикл обязан продолжиться")
        return maintenance.PurgeReport()

    monkeypatch.setattr("app.main.purge_expired", fake_purge)

    async def scenario() -> None:
        from app.main import maintenance_loop

        task = asyncio.create_task(maintenance_loop(interval_seconds=0.01))
        for _ in range(200):
            if len(calls) >= 3:
                break
            await asyncio.sleep(0.01)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

    asyncio.run(scenario())
    # Three runs: the immediate one, then two more despite the failure in between.
    assert len(calls) >= 3


def test_maintenance_is_enabled_only_for_postgres_with_a_retention(monkeypatch) -> None:
    from app.main import _maintenance_enabled

    monkeypatch.setattr(settings, "job_store", "sqlite")
    monkeypatch.setattr(settings, "audit_retention_days", 90)
    assert _maintenance_enabled() is False

    monkeypatch.setattr(settings, "job_store", "postgres")
    monkeypatch.setattr(settings, "audit_retention_days", 0)
    monkeypatch.setattr(settings, "feedback_retention_days", 0)
    assert _maintenance_enabled() is False

    monkeypatch.setattr(settings, "audit_retention_days", 90)
    assert _maintenance_enabled() is True
