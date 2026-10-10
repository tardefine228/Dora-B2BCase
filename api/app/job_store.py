"""Background job storage: in-memory or SQLite-backed (``JOB_STORE``).

Two interchangeable implementations share one interface:

* :class:`MemoryJobStore` — the historical process-local store; the fastest and
  the right choice for tests and single-shot demos;
* :class:`SqliteJobStore` — persists batches to ``JOB_STORE_PATH`` (WAL, stdlib
  ``sqlite3`` only), so a restarted API still answers ``GET /classify/batch/{id}``
  and a finished batch is not lost.

Honesty about restarts: the worker lives in the API process, so jobs that were
``queued``/``running`` when it stopped can never finish. :func:`init_job_store`
marks them ``failed`` with ``error="interrupted by restart"`` at startup instead
of leaving them hanging.

Both stores support TTL cleanup: finished jobs older than ``JOB_TTL_HOURS`` are
removed at startup and periodically (every ``_PURGE_EVERY_CALLS`` store calls).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypeAlias
from uuid import uuid4

from app.core.config import settings
from app.schemas import ClassifyResponse

logger = logging.getLogger("dora.jobs")

JobStatus = Literal["queued", "running", "completed", "failed"]
_FINISHED_STATUSES = ("completed", "failed")
_ACTIVE_STATUSES = ("queued", "running")
_PURGE_EVERY_CALLS = 100
INTERRUPTED_ERROR = "interrupted by restart"


@dataclass
class BatchJob:
    job_id: str
    items: list[str]
    status: JobStatus = "queued"
    completed: int = 0
    results: list[ClassifyResponse] = field(default_factory=list)
    error: str | None = None
    updated_at: float = field(default_factory=time.time, repr=False)


@dataclass
class ReindexJob:
    job_id: str
    status: JobStatus = "queued"
    total: int | None = None
    indexed: int = 0
    error: str | None = None
    updated_at: float = field(default_factory=time.time, repr=False)


Job: TypeAlias = BatchJob | ReindexJob


class JobStoreCapacityError(RuntimeError):
    """Raised when the configured job limits have been reached."""


class ReindexAlreadyRunningError(RuntimeError):
    """Only one index rebuild may run at a time in this process."""


class JobStore(Protocol):
    """The storage interface shared by the memory and SQLite implementations."""

    def create_batch(self, items: list[str]) -> BatchJob: ...

    def create_reindex(self) -> ReindexJob: ...

    def get(self, job_id: str) -> Job | None: ...

    def start(self, job_id: str) -> bool: ...

    def add_batch_result(self, job_id: str, result: ClassifyResponse) -> bool: ...

    def complete_batch(self, job_id: str) -> bool: ...

    def complete_reindex(self, job_id: str, total: int) -> bool: ...

    def fail(self, job_id: str, error: str) -> bool: ...

    def mark_interrupted(self, reason: str = INTERRUPTED_ERROR) -> int: ...

    def purge_expired(self) -> int: ...


class MemoryJobStore:
    """Bounded, thread-safe in-memory store for short-lived background jobs."""

    def __init__(
        self,
        *,
        max_active_jobs: int = 16,
        max_stored_jobs: int = 512,
        retention_seconds: int = 24 * 3600,
    ) -> None:
        self._max_active_jobs = max_active_jobs
        self._max_stored_jobs = max_stored_jobs
        self._retention_seconds = retention_seconds
        self._lock = threading.RLock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()

    def _purge_locked(self) -> int:
        now = time.time()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in _FINISHED_STATUSES and now - job.updated_at >= self._retention_seconds
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)
        return len(expired)

    def purge_expired(self) -> int:
        with self._lock:
            return self._purge_locked()

    def _reserve_locked(self) -> None:
        self._purge_locked()
        active_count = sum(job.status in _ACTIVE_STATUSES for job in self._jobs.values())
        if active_count >= self._max_active_jobs:
            raise JobStoreCapacityError("Слишком много активных фоновых задач")

        while len(self._jobs) >= self._max_stored_jobs:
            finished_job_id = next(
                (job_id for job_id, job in self._jobs.items() if job.status in _FINISHED_STATUSES),
                None,
            )
            if finished_job_id is None:
                raise JobStoreCapacityError("Хранилище фоновых задач заполнено")
            self._jobs.pop(finished_job_id, None)

    def create_batch(self, items: list[str]) -> BatchJob:
        with self._lock:
            self._reserve_locked()
            job = BatchJob(job_id=str(uuid4()), items=list(items))
            self._jobs[job.job_id] = job
            return deepcopy(job)

    def create_reindex(self) -> ReindexJob:
        with self._lock:
            self._purge_locked()
            if any(
                isinstance(job, ReindexJob) and job.status in _ACTIVE_STATUSES
                for job in self._jobs.values()
            ):
                raise ReindexAlreadyRunningError("Переиндексация уже выполняется")
            self._reserve_locked()
            job = ReindexJob(job_id=str(uuid4()))
            self._jobs[job.job_id] = job
            return deepcopy(job)

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._purge_locked()
            job = self._jobs.get(job_id)
            return deepcopy(job) if job is not None else None

    def start(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status != "queued":
                return False
            job.status = "running"
            job.updated_at = time.time()
            return True

    def add_batch_result(self, job_id: str, result: ClassifyResponse) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not isinstance(job, BatchJob) or job.status != "running":
                return False
            job.results.append(result)
            job.completed += 1
            job.updated_at = time.time()
            return True

    def complete_batch(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not isinstance(job, BatchJob) or job.status != "running":
                return False
            job.status = "completed"
            job.updated_at = time.time()
            return True

    def complete_reindex(self, job_id: str, total: int) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not isinstance(job, ReindexJob) or job.status != "running":
                return False
            job.total = total
            job.indexed = total
            job.status = "completed"
            job.updated_at = time.time()
            return True

    def fail(self, job_id: str, error: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in _ACTIVE_STATUSES:
                return False
            job.status = "failed"
            job.error = error
            job.updated_at = time.time()
            return True

    def mark_interrupted(self, reason: str = INTERRUPTED_ERROR) -> int:
        """No-op for the memory store: a new process starts with an empty dict."""
        return 0

    def close(self) -> None:
        """Nothing to close for the in-memory store."""


class SqliteJobStore:
    """Persistent job store on stdlib ``sqlite3`` (WAL, one write lock).

    Results are stored as JSON (``ClassifyResponse`` payloads) in a single row
    per job, which keeps the schema small and makes progress updates atomic:
    ``add_batch_result`` reads the row, appends and writes it back under the
    store lock, so concurrent workers cannot lose updates.
    """

    _SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            status TEXT NOT NULL,
            items TEXT NOT NULL DEFAULT '[]',
            completed INTEGER NOT NULL DEFAULT 0,
            results TEXT NOT NULL DEFAULT '[]',
            total INTEGER,
            indexed INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, updated_at)",
    )

    def __init__(
        self,
        path: str | Path,
        *,
        max_active_jobs: int = 16,
        max_stored_jobs: int = 512,
        ttl_seconds: int = 24 * 3600,
    ) -> None:
        self.path = Path(path)
        self._max_active_jobs = max_active_jobs
        self._max_stored_jobs = max_stored_jobs
        self._ttl_seconds = ttl_seconds
        self._lock = threading.RLock()
        self._calls_since_purge = 0
        if self.path.parent and str(self.path.parent) not in ("", "."):
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self._connection.row_factory = sqlite3.Row
        with self._lock:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA synchronous=NORMAL")
            for statement in self._SCHEMA:
                self._connection.execute(statement)
            self._connection.commit()

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> Job:
        common = {
            "job_id": row["job_id"],
            "status": row["status"],
            "error": row["error"],
            "updated_at": row["updated_at"],
        }
        if row["kind"] == "batch":
            results = json.loads(row["results"] or "[]")
            return BatchJob(
                items=json.loads(row["items"] or "[]"),
                completed=row["completed"],
                results=[ClassifyResponse.model_validate(item) for item in results],
                **common,
            )
        return ReindexJob(
            total=row["total"],
            indexed=row["indexed"],
            **common,
        )

    def _fetch_locked(self, job_id: str) -> sqlite3.Row | None:
        cursor = self._connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        return cursor.fetchone()

    def _touch(self) -> None:
        """Purge expired jobs every ``_PURGE_EVERY_CALLS`` store calls."""
        self._calls_since_purge += 1
        if self._calls_since_purge >= _PURGE_EVERY_CALLS:
            self._calls_since_purge = 0
            self._purge_locked()

    def _purge_locked(self) -> int:
        threshold = time.time() - self._ttl_seconds
        cursor = self._connection.execute(
            "DELETE FROM jobs WHERE status IN ('completed', 'failed') AND updated_at < ?",
            (threshold,),
        )
        removed = cursor.rowcount or 0
        # Enforce the stored-jobs bound by dropping the oldest finished jobs.
        cursor = self._connection.execute(
            "SELECT COUNT(*) AS count FROM jobs",
        )
        stored = cursor.fetchone()["count"]
        if stored >= self._max_stored_jobs:
            cursor = self._connection.execute(
                """
                DELETE FROM jobs WHERE job_id IN (
                    SELECT job_id FROM jobs
                    WHERE status IN ('completed', 'failed')
                    ORDER BY updated_at ASC
                    LIMIT ?
                )
                """,
                (stored - self._max_stored_jobs + 1,),
            )
            removed += cursor.rowcount or 0
        # Always commit: an uncommitted DELETE would keep a write transaction
        # open on the connection and lock the file for every other connection.
        self._connection.commit()
        if removed:
            logger.info("Очистка фоновых задач: удалено %d записей", removed)
        return removed

    def _reserve_locked(self) -> None:
        if self._purge_locked():
            self._calls_since_purge = 0
        active = self._connection.execute(
            "SELECT COUNT(*) AS count FROM jobs WHERE status IN ('queued', 'running')"
        ).fetchone()["count"]
        if active >= self._max_active_jobs:
            raise JobStoreCapacityError("Слишком много активных фоновых задач")

    # --- interface --------------------------------------------------------
    def create_batch(self, items: list[str]) -> BatchJob:
        job = BatchJob(job_id=str(uuid4()), items=list(items))
        now = time.time()
        with self._lock:
            self._reserve_locked()
            self._connection.execute(
                """
                INSERT INTO jobs (job_id, kind, status, items, completed, results,
                                  created_at, updated_at)
                VALUES (?, 'batch', 'queued', ?, 0, '[]', ?, ?)
                """,
                (job.job_id, json.dumps(list(items), ensure_ascii=False), now, now),
            )
            self._connection.commit()
            job.updated_at = now
        return job

    def create_reindex(self) -> ReindexJob:
        job = ReindexJob(job_id=str(uuid4()))
        now = time.time()
        with self._lock:
            running = self._connection.execute(
                "SELECT COUNT(*) AS count FROM jobs WHERE kind = 'reindex' "
                "AND status IN ('queued', 'running')"
            ).fetchone()["count"]
            if running:
                raise ReindexAlreadyRunningError("Переиндексация уже выполняется")
            self._reserve_locked()
            self._connection.execute(
                """
                INSERT INTO jobs (job_id, kind, status, items, completed, results,
                                  total, indexed, created_at, updated_at)
                VALUES (?, 'reindex', 'queued', '[]', 0, '[]', NULL, 0, ?, ?)
                """,
                (job.job_id, now, now),
            )
            self._connection.commit()
            job.updated_at = now
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            self._touch()
            row = self._fetch_locked(job_id)
            return self._row_to_job(row) if row is not None else None

    def start(self, job_id: str) -> bool:
        with self._lock:
            row = self._fetch_locked(job_id)
            if row is None or row["status"] != "queued":
                return False
            self._connection.execute(
                "UPDATE jobs SET status = 'running', updated_at = ? WHERE job_id = ?",
                (time.time(), job_id),
            )
            self._connection.commit()
            return True

    def add_batch_result(self, job_id: str, result: ClassifyResponse) -> bool:
        with self._lock:
            row = self._fetch_locked(job_id)
            if row is None or row["kind"] != "batch" or row["status"] != "running":
                return False
            results = json.loads(row["results"] or "[]")
            results.append(result.model_dump(mode="json"))
            self._connection.execute(
                """
                UPDATE jobs SET results = ?, completed = completed + 1, updated_at = ?
                WHERE job_id = ?
                """,
                (json.dumps(results, ensure_ascii=False), time.time(), job_id),
            )
            self._connection.commit()
            return True

    def complete_batch(self, job_id: str) -> bool:
        with self._lock:
            row = self._fetch_locked(job_id)
            if row is None or row["kind"] != "batch" or row["status"] != "running":
                return False
            self._connection.execute(
                "UPDATE jobs SET status = 'completed', updated_at = ? WHERE job_id = ?",
                (time.time(), job_id),
            )
            self._connection.commit()
            return True

    def complete_reindex(self, job_id: str, total: int) -> bool:
        with self._lock:
            row = self._fetch_locked(job_id)
            if row is None or row["kind"] != "reindex" or row["status"] != "running":
                return False
            self._connection.execute(
                "UPDATE jobs SET status = 'completed', total = ?, indexed = ?, "
                "updated_at = ? WHERE job_id = ?",
                (total, total, time.time(), job_id),
            )
            self._connection.commit()
            return True

    def fail(self, job_id: str, error: str) -> bool:
        with self._lock:
            row = self._fetch_locked(job_id)
            if row is None or row["status"] not in _ACTIVE_STATUSES:
                return False
            self._connection.execute(
                "UPDATE jobs SET status = 'failed', error = ?, updated_at = ? WHERE job_id = ?",
                (error, time.time(), job_id),
            )
            self._connection.commit()
            return True

    def mark_interrupted(self, reason: str = INTERRUPTED_ERROR) -> int:
        """A restarted process cannot finish queued/running jobs: fail them honestly."""
        with self._lock:
            cursor = self._connection.execute(
                """
                UPDATE jobs SET status = 'failed', error = ?, updated_at = ?
                WHERE status IN ('queued', 'running')
                """,
                (reason, time.time()),
            )
            self._connection.commit()
            if cursor.rowcount:
                logger.warning("Помечено failed после рестарта: %d задач", cursor.rowcount)
            return cursor.rowcount or 0

    def purge_expired(self) -> int:
        with self._lock:
            self._calls_since_purge = 0
            return self._purge_locked()

    def close(self) -> None:
        with self._lock:
            self._connection.close()


# ``JobStore`` used to be the only (memory) implementation; keep the alias so
# external imports keep working.
InMemoryJobStore = MemoryJobStore

_store: JobStore | None = None
_store_lock = threading.RLock()


def build_job_store() -> JobStore:
    """Create the store selected by ``JOB_STORE`` (``sqlite`` by default)."""
    backend = (settings.job_store or "sqlite").strip().lower()
    if backend == "memory":
        logger.info("Фоновые задачи: in-memory store")
        return MemoryJobStore(
            max_active_jobs=settings.job_max_active,
            max_stored_jobs=settings.job_max_stored,
            retention_seconds=settings.job_ttl_hours * 3600,
        )
    if backend == "sqlite":
        path = _resolve_job_store_path(settings.job_store_path)
        logger.info("Фоновые задачи: sqlite store %s", path)
        return SqliteJobStore(
            path,
            max_active_jobs=settings.job_max_active,
            max_stored_jobs=settings.job_max_stored,
            ttl_seconds=settings.job_ttl_hours * 3600,
        )
    raise ValueError(f"Неизвестный JOB_STORE: {backend!r} (ожидается memory или sqlite)")


def _resolve_job_store_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path
    project_root = Path(__file__).resolve().parents[2]
    return project_root / path


def get_store() -> JobStore:
    """Process-wide store instance, created on first use from the settings."""
    global _store
    with _store_lock:
        if _store is None:
            _store = build_job_store()
        return _store


def reset_store() -> None:
    """Drop the singleton (tests, and after changing ``JOB_STORE`` settings)."""
    global _store
    with _store_lock:
        if _store is not None:
            try:
                _store.close()
            except Exception:  # pragma: no cover - closing must never break a reset
                logger.exception("Не удалось закрыть store типа %s", type(_store).__name__)
            _store = None


def init_job_store() -> int:
    """Startup hook: fail interrupted jobs, then clean expired ones."""
    store = get_store()
    interrupted = store.mark_interrupted()
    store.purge_expired()
    return interrupted


# --- module-level helpers (used by the API handlers) ----------------------


def create_job(items: list[str]) -> BatchJob:
    """Backwards-compatible name for creating a classification batch."""
    return get_store().create_batch(items)


def create_reindex_job() -> ReindexJob:
    return get_store().create_reindex()


def get_job(job_id: str) -> Job | None:
    return get_store().get(job_id)


def start_job(job_id: str) -> bool:
    return get_store().start(job_id)


def add_batch_result(job_id: str, result: ClassifyResponse) -> bool:
    return get_store().add_batch_result(job_id, result)


def complete_batch(job_id: str) -> bool:
    return get_store().complete_batch(job_id)


def complete_reindex(job_id: str, total: int) -> bool:
    return get_store().complete_reindex(job_id, total)


def fail_job(job_id: str, error: str) -> bool:
    return get_store().fail(job_id, error)
