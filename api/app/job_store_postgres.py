"""PostgreSQL-backed job store (``JOB_STORE=postgres``) and the request audit log.

Why PostgreSQL: the API and the Celery worker are separate processes (separate
containers), so batch state has to live in a database both can reach. A job
survives restarts of either side; ``GET /classify/batch/{id}`` keeps answering.

Same interface as :class:`app.job_store.SqliteJobStore`. Differences that matter:

* progress is appended with a single ``results = results || ...`` statement, so a
  worker never rewrites the whole result array for every product;
* ``mark_interrupted`` only fails *stale* jobs (not touched for
  ``JOB_STALE_SECONDS``): an API restart must not kill batches that a worker is
  still processing, while a worker that really died is detected by the timeout.
"""

from __future__ import annotations

import json
import logging
import time
from uuid import uuid4

from app.job_store import (
    _ACTIVE_STATUSES,
    INTERRUPTED_ERROR,
    BatchJob,
    Job,
    JobStoreCapacityError,
    ReindexAlreadyRunningError,
    ReindexJob,
)
from app.schemas import ClassifyResponse

logger = logging.getLogger("dora.jobs.postgres")

_SCHEMA = (
    """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        kind TEXT NOT NULL,
        status TEXT NOT NULL,
        items JSONB NOT NULL DEFAULT '[]'::jsonb,
        completed INTEGER NOT NULL DEFAULT 0,
        results JSONB NOT NULL DEFAULT '[]'::jsonb,
        total INTEGER,
        indexed INTEGER NOT NULL DEFAULT 0,
        error TEXT,
        created_at DOUBLE PRECISION NOT NULL,
        updated_at DOUBLE PRECISION NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status, updated_at)",
    """
    CREATE TABLE IF NOT EXISTS classification_log (
        id BIGSERIAL PRIMARY KEY,
        ts TIMESTAMPTZ NOT NULL DEFAULT now(),
        source TEXT NOT NULL,
        query TEXT NOT NULL,
        code TEXT,
        status TEXT NOT NULL,
        confidence REAL,
        latency_ms REAL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_classification_log_ts ON classification_log(ts)",
)


def _connect_pool(database_url: str, *, retries: int = 30, delay: float = 1.0):
    """Open a connection pool, waiting for PostgreSQL to accept connections.

    ``docker compose`` starts the database and the API together; the first seconds
    the database refuses connections, so the first connection is retried.
    """
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(database_url, min_size=1, max_size=8, open=False)
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            pool.open(wait=True, timeout=delay + 2)
            return pool
        except Exception as exc:  # noqa: BLE001 - connection errors vary
            last_error = exc
            logger.info("PostgreSQL ещё не готов (попытка %d/%d): %s", attempt + 1, retries, exc)
            try:
                pool.close()
            except Exception:  # noqa: BLE001
                pass
            pool = ConnectionPool(database_url, min_size=1, max_size=8, open=False)
            time.sleep(delay)
    raise RuntimeError(f"PostgreSQL недоступен: {last_error}")


class PostgresJobStore:
    def __init__(
        self,
        database_url: str,
        *,
        max_active_jobs: int = 16,
        max_stored_jobs: int = 512,
        ttl_seconds: int = 24 * 3600,
        stale_seconds: int = 900,
    ) -> None:
        self._max_active_jobs = max_active_jobs
        self._max_stored_jobs = max_stored_jobs
        self._ttl_seconds = ttl_seconds
        self._stale_seconds = stale_seconds
        self._pool = _connect_pool(database_url)
        with self._pool.connection() as conn:
            # Concurrent API/worker starts must not race on CREATE TABLE.
            conn.execute("SELECT pg_advisory_xact_lock(1875)")
            for statement in _SCHEMA:
                conn.execute(statement)

    # --- helpers ----------------------------------------------------------
    @staticmethod
    def _row_to_job(row: tuple, columns: list[str]) -> Job:
        data = dict(zip(columns, row, strict=True))
        common = {
            "job_id": data["job_id"],
            "status": data["status"],
            "error": data["error"],
            "updated_at": data["updated_at"],
        }
        if data["kind"] == "batch":
            return BatchJob(
                items=list(data["items"] or []),
                completed=data["completed"],
                results=[ClassifyResponse.model_validate(item) for item in data["results"] or []],
                **common,
            )
        return ReindexJob(total=data["total"], indexed=data["indexed"], **common)

    def _purge(self, conn) -> int:
        threshold = time.time() - self._ttl_seconds
        removed = conn.execute(
            "DELETE FROM jobs WHERE status IN ('completed', 'failed') AND updated_at < %s",
            (threshold,),
        ).rowcount
        stored = conn.execute("SELECT COUNT(*) FROM jobs").fetchone()[0]
        if stored >= self._max_stored_jobs:
            removed += conn.execute(
                """
                DELETE FROM jobs WHERE job_id IN (
                    SELECT job_id FROM jobs WHERE status IN ('completed', 'failed')
                    ORDER BY updated_at ASC LIMIT %s
                )
                """,
                (stored - self._max_stored_jobs + 1,),
            ).rowcount
        return removed or 0

    def _reserve(self, conn) -> None:
        conn.execute("SELECT pg_advisory_xact_lock(1876)")  # serialize capacity checks
        self._purge(conn)
        active = conn.execute(
            "SELECT COUNT(*) FROM jobs WHERE status IN ('queued', 'running')"
        ).fetchone()[0]
        if active >= self._max_active_jobs:
            raise JobStoreCapacityError("Слишком много активных фоновых задач")

    # --- interface --------------------------------------------------------
    def create_batch(self, items: list[str]) -> BatchJob:
        job = BatchJob(job_id=str(uuid4()), items=list(items))
        now = time.time()
        with self._pool.connection() as conn:
            self._reserve(conn)
            conn.execute(
                """
                INSERT INTO jobs (job_id, kind, status, items, created_at, updated_at)
                VALUES (%s, 'batch', 'queued', %s::jsonb, %s, %s)
                """,
                (job.job_id, json.dumps(list(items), ensure_ascii=False), now, now),
            )
        job.updated_at = now
        return job

    def create_reindex(self) -> ReindexJob:
        job = ReindexJob(job_id=str(uuid4()))
        now = time.time()
        with self._pool.connection() as conn:
            self._reserve(conn)
            running = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE kind = 'reindex' "
                "AND status IN ('queued', 'running')"
            ).fetchone()[0]
            if running:
                raise ReindexAlreadyRunningError("Переиндексация уже выполняется")
            conn.execute(
                "INSERT INTO jobs (job_id, kind, status, created_at, updated_at) "
                "VALUES (%s, 'reindex', 'queued', %s, %s)",
                (job.job_id, now, now),
            )
        job.updated_at = now
        return job

    def get(self, job_id: str) -> Job | None:
        with self._pool.connection() as conn:
            cursor = conn.execute("SELECT * FROM jobs WHERE job_id = %s", (job_id,))
            row = cursor.fetchone()
            if row is None:
                return None
            columns = [column.name for column in cursor.description]
        return self._row_to_job(row, columns)

    def start(self, job_id: str) -> bool:
        with self._pool.connection() as conn:
            return bool(
                conn.execute(
                    "UPDATE jobs SET status = 'running', updated_at = %s "
                    "WHERE job_id = %s AND status = 'queued'",
                    (time.time(), job_id),
                ).rowcount
            )

    def add_batch_result(self, job_id: str, result: ClassifyResponse) -> bool:
        payload = json.dumps([result.model_dump(mode="json")], ensure_ascii=False)
        with self._pool.connection() as conn:
            return bool(
                conn.execute(
                    "UPDATE jobs SET results = results || %s::jsonb, "
                    "completed = completed + 1, updated_at = %s "
                    "WHERE job_id = %s AND kind = 'batch' AND status = 'running'",
                    (payload, time.time(), job_id),
                ).rowcount
            )

    def complete_batch(self, job_id: str) -> bool:
        with self._pool.connection() as conn:
            return bool(
                conn.execute(
                    "UPDATE jobs SET status = 'completed', updated_at = %s "
                    "WHERE job_id = %s AND kind = 'batch' AND status = 'running'",
                    (time.time(), job_id),
                ).rowcount
            )

    def complete_reindex(self, job_id: str, total: int) -> bool:
        with self._pool.connection() as conn:
            return bool(
                conn.execute(
                    "UPDATE jobs SET status = 'completed', total = %s, indexed = %s, "
                    "updated_at = %s WHERE job_id = %s AND kind = 'reindex' "
                    "AND status = 'running'",
                    (total, total, time.time(), job_id),
                ).rowcount
            )

    def fail(self, job_id: str, error: str) -> bool:
        with self._pool.connection() as conn:
            return bool(
                conn.execute(
                    "UPDATE jobs SET status = 'failed', error = %s, updated_at = %s "
                    "WHERE job_id = %s AND status = ANY(%s)",
                    (error, time.time(), job_id, list(_ACTIVE_STATUSES)),
                ).rowcount
            )

    def mark_interrupted(self, reason: str = INTERRUPTED_ERROR) -> int:
        """Fail jobs nobody has touched for ``stale_seconds`` (their worker is gone)."""
        threshold = time.time() - self._stale_seconds
        with self._pool.connection() as conn:
            count = conn.execute(
                "UPDATE jobs SET status = 'failed', error = %s, updated_at = %s "
                "WHERE status IN ('queued', 'running') AND updated_at < %s",
                (reason, time.time(), threshold),
            ).rowcount
        if count:
            logger.warning("Помечено failed (зависшие задачи): %d", count)
        return count or 0

    def purge_expired(self) -> int:
        with self._pool.connection() as conn:
            return self._purge(conn)

    # --- audit log ----------------------------------------------------------
    def log_classifications(self, source: str, rows: list[ClassifyResponse]) -> None:
        """Best effort: an audit failure must never break an answer."""
        if not rows:
            return
        try:
            with self._pool.connection() as conn, conn.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO classification_log "
                    "(source, query, code, status, confidence, latency_ms) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (source, r.query[:500], r.code, r.status, r.confidence, r.latency_ms)
                        for r in rows
                    ],
                )
        except Exception:  # noqa: BLE001
            logger.exception("Не удалось записать журнал классификаций")

    def close(self) -> None:
        self._pool.close()
