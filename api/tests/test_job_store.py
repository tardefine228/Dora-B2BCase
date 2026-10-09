"""SQLite-backed job store: persistence, restart honesty, TTL and concurrency."""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest
from app.core.config import settings
from app.job_store import (
    BatchJob,
    MemoryJobStore,
    ReindexAlreadyRunningError,
    ReindexJob,
    SqliteJobStore,
    add_batch_result,
    complete_batch,
    create_job,
    fail_job,
    get_job,
    get_store,
    init_job_store,
    reset_store,
    start_job,
)
from app.main import app
from app.schemas import ClassifyResponse
from fastapi.testclient import TestClient


def _response(index: int) -> ClassifyResponse:
    return ClassifyResponse(
        query=f"Товар {index}",
        code="26.20.11",
        title="Компьютеры портативные",
        confidence=0.5,
        status="review",
        latency_ms=1.0,
    )


@pytest.fixture
def restartable_store(tmp_path, monkeypatch):
    """Switch the API to a SQLite store in a temporary file and restart it on demand."""
    path = tmp_path / "jobs.sqlite"
    monkeypatch.setattr(settings, "job_store", "sqlite")
    monkeypatch.setattr(settings, "job_store_path", str(path))
    reset_store()

    def restart() -> SqliteJobStore:
        """Re-create the store on the same file, as a new API process would."""
        reset_store()
        store = get_store()
        assert isinstance(store, SqliteJobStore)
        return store

    yield path, restart
    reset_store()


def test_jobs_survive_store_recreation(restartable_store) -> None:
    path, restart = restartable_store
    job = create_job(["Ноутбук Lenovo", "Мышь компьютерная"])
    assert start_job(job.job_id)
    assert add_batch_result(job.job_id, _response(1))
    assert complete_batch(job.job_id)
    assert path.is_file()

    restart()
    restored = get_job(job.job_id)
    assert isinstance(restored, BatchJob)
    assert restored.status == "completed"
    assert restored.items == ["Ноутбук Lenovo", "Мышь компьютерная"]
    assert restored.completed == 1
    assert restored.results[0].query == "Товар 1"


def test_running_jobs_become_failed_after_restart(restartable_store) -> None:
    _path, restart = restartable_store
    batch = create_job(["Товар A"])
    queued = create_job(["Товар B"])
    assert start_job(batch.job_id)

    restart()
    interrupted = init_job_store()
    assert interrupted == 2  # running batch + queued batch

    for job_id in (batch.job_id, queued.job_id):
        restored = get_job(job_id)
        assert restored is not None
        assert restored.status == "failed"
        assert restored.error == "interrupted by restart"


def test_sqlite_store_is_wired_into_the_api(restartable_store) -> None:
    """The existing batch flow of test_api.py works against the sqlite store."""
    path, _restart = restartable_store
    assert isinstance(get_store(), SqliteJobStore)

    with TestClient(app) as client:
        accepted = client.post(
            "/classify/batch",
            json={"items": ["Ноутбук Lenovo", "Мышь компьютерная"]},
        )
        assert accepted.status_code == 202
        status = client.get(accepted.json()["status_url"]).json()
        assert status["status"] == "completed"
        assert status["completed"] == 2
        assert [row["query"] for row in status["results"]] == [
            "Ноутбук Lenovo",
            "Мышь компьютерная",
        ]

    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT status, completed FROM jobs").fetchall()
    assert rows == [("completed", 2)]


def test_concurrent_progress_updates_are_not_lost(restartable_store) -> None:
    _path, _restart = restartable_store
    store = get_store()
    job = store.create_batch(["query"] * 200)
    assert store.start(job.job_id)

    def worker(offset: int) -> None:
        for index in range(offset, 200, 8):
            assert store.add_batch_result(job.job_id, _response(index))

    threads = [threading.Thread(target=worker, args=(offset,)) for offset in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stored = store.get(job.job_id)
    assert isinstance(stored, BatchJob)
    assert stored.completed == 200
    assert len(stored.results) == 200
    assert sorted(row.query for row in stored.results) == sorted(
        f"Товар {index}" for index in range(200)
    )


def test_expired_finished_jobs_are_purged(restartable_store) -> None:
    path, _restart = restartable_store
    store = get_store()
    job = store.create_batch(["Товар"])
    assert store.start(job.job_id)
    assert store.complete_batch(job.job_id)

    stale = time.time() - (settings.job_ttl_hours * 3600 + 60)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE jobs SET updated_at = ?", (stale,))
        connection.commit()

    assert store.purge_expired() == 1
    assert store.get(job.job_id) is None


def test_purge_does_not_leave_a_write_transaction_open(restartable_store) -> None:
    """A stray uncommitted DELETE locks the sqlite file for other connections."""
    _path, _restart = restartable_store
    store = get_store()
    assert isinstance(store, SqliteJobStore)

    assert store.purge_expired() == 0
    assert store._connection.in_transaction is False

    job = create_job(["Товар"])
    assert store._connection.in_transaction is False
    assert start_job(job.job_id)
    assert store._connection.in_transaction is False


def test_reindex_job_is_exclusive_and_persisted(restartable_store) -> None:
    _path, restart = restartable_store
    store = get_store()
    job = store.create_reindex()
    assert isinstance(job, ReindexJob)
    with pytest.raises(ReindexAlreadyRunningError):
        store.create_reindex()

    assert store.start(job.job_id)
    assert store.complete_reindex(job.job_id, 1126)
    restart()
    restored = get_job(job.job_id)
    assert isinstance(restored, ReindexJob)
    assert (restored.status, restored.total, restored.indexed) == ("completed", 1126, 1126)


def test_failed_job_keeps_error(restartable_store) -> None:
    _path, _restart = restartable_store
    job = create_job(["Товар"])
    assert fail_job(job.job_id, "Не удалось обработать список товаров")
    restored = get_job(job.job_id)
    assert restored is not None
    assert restored.status == "failed"
    assert restored.error == "Не удалось обработать список товаров"
    # A finished job cannot be started or failed twice.
    assert not start_job(job.job_id)
    assert not fail_job(job.job_id, "снова")


def test_memory_store_parity() -> None:
    store = MemoryJobStore(retention_seconds=3600)
    job = store.create_batch(["a", "b"])
    assert store.get(job.job_id) is not None
    assert store.start(job.job_id)
    assert store.add_batch_result(job.job_id, _response(1))
    assert store.complete_batch(job.job_id)
    assert store.mark_interrupted() == 0  # a fresh process starts from an empty dict
    assert store.purge_expired() == 0
    store.close()
