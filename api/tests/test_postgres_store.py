"""PostgreSQL job store. Needs a database: set TEST_DATABASE_URL (skipped otherwise)."""

import os

import pytest
from app.job_store import BatchJob, JobStoreCapacityError, ReindexAlreadyRunningError
from app.schemas import ClassifyResponse

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")


@pytest.fixture
def store(monkeypatch):
    from app.core.config import settings
    from app.job_store_postgres import PostgresJobStore

    # Feedback and retention use settings, not the store instance's URL.
    # Point every participant at the same isolated test database.
    monkeypatch.setattr(settings, "job_store", "postgres")
    monkeypatch.setattr(settings, "database_url", DATABASE_URL)

    instance = PostgresJobStore(DATABASE_URL, max_active_jobs=3, stale_seconds=1)
    with instance._pool.connection() as conn:
        conn.execute("TRUNCATE jobs, classification_log, classification_feedback")
    yield instance
    instance.close()


def _answer(query: str) -> ClassifyResponse:
    return ClassifyResponse(
        query=query, code="26.20.11", title="t", confidence=0.9, status="ok", latency_ms=1.0
    )


def test_batch_lifecycle_and_progress(store):
    job = store.create_batch(["a", "b"])
    assert store.get(job.job_id).status == "queued"
    assert store.start(job.job_id) and not store.start(job.job_id)
    assert store.add_batch_result(job.job_id, _answer("a"))
    assert store.add_batch_result(job.job_id, _answer("b"))
    assert store.complete_batch(job.job_id)
    done = store.get(job.job_id)
    assert isinstance(done, BatchJob) and done.status == "completed"
    assert [r.query for r in done.results] == ["a", "b"] and done.completed == 2


def test_results_are_rejected_for_a_job_that_is_not_running(store):
    job = store.create_batch(["a"])
    assert not store.add_batch_result(job.job_id, _answer("a"))


def test_active_job_capacity_and_single_reindex(store):
    store.create_reindex()
    with pytest.raises(ReindexAlreadyRunningError):
        store.create_reindex()
    store.create_batch(["x"])
    store.create_batch(["y"])
    with pytest.raises(JobStoreCapacityError):
        store.create_batch(["z"])


def test_stale_jobs_are_failed_but_fresh_ones_are_kept(store):
    import time

    job = store.create_batch(["a"])
    store.start(job.job_id)
    assert store.mark_interrupted() == 0  # touched just now: a live worker may own it
    time.sleep(1.2)
    assert store.mark_interrupted() == 1
    assert store.get(job.job_id).status == "failed"


def test_audit_log_is_written(store):
    store.log_classifications("single", [_answer("ноутбук")])
    with store._pool.connection() as conn:
        count = conn.execute("SELECT COUNT(*) FROM classification_log").fetchone()[0]
    assert count == 1


# --- schema migrations and retention on a real PostgreSQL -------------------
# All tests here require TEST_DATABASE_URL; these two also exercise migrations
# and are the end-to-end check of what `test_migrations.py` verifies statically.


def test_store_brings_the_schema_to_the_declared_head(store) -> None:
    from app.migrations_runner import REVISION_HEAD, current_revision

    assert current_revision(DATABASE_URL) == REVISION_HEAD
    with store._pool.connection() as conn:
        columns = {
            row[0]
            for row in conn.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'classification_feedback'"
            )
        }
    # 0002_feedback_comment: the column appears on an existing database too.
    assert "comment" in columns


def test_retention_purges_old_rows_and_keeps_fresh_ones(store) -> None:
    from app import maintenance
    from app.core.config import settings
    from app.services.feedback_store import save_feedback

    save_feedback("Старый вердикт", "01.1", "correct", None, "старое")
    save_feedback("Свежий вердикт", "01.1", "correct", None, "свежее")
    store.log_classifications("single", [_answer("свежий запрос")])
    with store._pool.connection() as conn:
        conn.execute(
            "UPDATE classification_feedback SET created_at = now() - interval '400 days' "
            "WHERE query = 'Старый вердикт'"
        )
        conn.execute(
            "INSERT INTO classification_log (source, query, status, confidence) "
            "VALUES ('single', 'старый запрос', 'ok', 0.9)"
        )
        conn.execute(
            "UPDATE classification_log SET ts = now() - interval '400 days' "
            "WHERE query = 'старый запрос'"
        )

    original = settings.audit_retention_days, settings.feedback_retention_days
    try:
        settings.audit_retention_days = 90
        settings.feedback_retention_days = 365
        dry = maintenance.purge_expired(dry_run=True)
        assert dry.audit_deleted >= 1
        assert dry.feedback_deleted >= 1
        # Dry run changed nothing.
        with store._pool.connection() as conn:
            assert conn.execute(
                "SELECT count(*) FROM classification_log WHERE query = 'старый запрос'"
            ).fetchone()[0] == 1

        report = maintenance.purge_expired()
        assert report.audit_deleted >= 1
        assert report.feedback_deleted >= 1
        with store._pool.connection() as conn:
            assert conn.execute(
                "SELECT count(*) FROM classification_log WHERE query = 'старый запрос'"
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT count(*) FROM classification_feedback WHERE query = 'Старый вердикт'"
            ).fetchone()[0] == 0
            assert conn.execute(
                "SELECT count(*) FROM classification_log WHERE query = 'свежий запрос'"
            ).fetchone()[0] == 1
            assert conn.execute(
                "SELECT count(*) FROM classification_feedback WHERE query = 'Свежий вердикт'"
            ).fetchone()[0] == 1
    finally:
        settings.audit_retention_days, settings.feedback_retention_days = original
