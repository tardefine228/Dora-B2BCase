"""PostgreSQL job store. Needs a database: set TEST_DATABASE_URL (skipped otherwise)."""

import os

import pytest
from app.job_store import BatchJob, JobStoreCapacityError, ReindexAlreadyRunningError
from app.schemas import ClassifyResponse

DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="TEST_DATABASE_URL is not set")


@pytest.fixture
def store():
    from app.job_store_postgres import PostgresJobStore

    instance = PostgresJobStore(DATABASE_URL, max_active_jobs=3, stale_seconds=1)
    with instance._pool.connection() as conn:
        conn.execute("TRUNCATE jobs, classification_log")
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
