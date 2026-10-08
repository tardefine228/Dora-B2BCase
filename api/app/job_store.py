"""Bounded, thread-safe in-memory store for short-lived background jobs.

This store is intentionally process-local for the demo deployment. Use Redis or a
persistent job queue before running multiple API workers or requiring restart safety.
"""

from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
from time import monotonic
from typing import Literal, TypeAlias
from uuid import uuid4

from app.schemas import ClassifyResponse

JobStatus = Literal["queued", "running", "completed", "failed"]


@dataclass
class BatchJob:
    job_id: str
    items: list[str]
    status: JobStatus = "queued"
    completed: int = 0
    results: list[ClassifyResponse] = field(default_factory=list)
    error: str | None = None
    updated_at: float = field(default_factory=monotonic, repr=False)


@dataclass
class ReindexJob:
    job_id: str
    status: JobStatus = "queued"
    total: int | None = None
    indexed: int = 0
    error: str | None = None
    updated_at: float = field(default_factory=monotonic, repr=False)


Job: TypeAlias = BatchJob | ReindexJob


class JobStoreCapacityError(RuntimeError):
    """Raised when the configured in-memory job limits have been reached."""


class ReindexAlreadyRunningError(RuntimeError):
    """Only one index rebuild may run at a time in this process."""


class JobStore:
    def __init__(
        self,
        *,
        max_active_jobs: int = 16,
        max_stored_jobs: int = 512,
        retention_seconds: int = 3600,
    ) -> None:
        self._max_active_jobs = max_active_jobs
        self._max_stored_jobs = max_stored_jobs
        self._retention_seconds = retention_seconds
        self._lock = RLock()
        self._jobs: OrderedDict[str, Job] = OrderedDict()

    def _purge_locked(self) -> None:
        now = monotonic()
        expired = [
            job_id
            for job_id, job in self._jobs.items()
            if job.status in {"completed", "failed"}
            and now - job.updated_at >= self._retention_seconds
        ]
        for job_id in expired:
            self._jobs.pop(job_id, None)

    def _reserve_locked(self) -> None:
        self._purge_locked()
        active_count = sum(job.status in {"queued", "running"} for job in self._jobs.values())
        if active_count >= self._max_active_jobs:
            raise JobStoreCapacityError("Слишком много активных фоновых задач")

        while len(self._jobs) >= self._max_stored_jobs:
            finished_job_id = next(
                (
                    job_id
                    for job_id, job in self._jobs.items()
                    if job.status in {"completed", "failed"}
                ),
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
                isinstance(job, ReindexJob) and job.status in {"queued", "running"}
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
            job.updated_at = monotonic()
            return True

    def add_batch_result(self, job_id: str, result: ClassifyResponse) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not isinstance(job, BatchJob) or job.status != "running":
                return False
            job.results.append(result)
            job.completed += 1
            job.updated_at = monotonic()
            return True

    def complete_batch(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not isinstance(job, BatchJob) or job.status != "running":
                return False
            job.status = "completed"
            job.updated_at = monotonic()
            return True

    def complete_reindex(self, job_id: str, total: int) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not isinstance(job, ReindexJob) or job.status != "running":
                return False
            job.total = total
            job.indexed = total
            job.status = "completed"
            job.updated_at = monotonic()
            return True

    def fail(self, job_id: str, error: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None or job.status not in {"queued", "running"}:
                return False
            job.status = "failed"
            job.error = error
            job.updated_at = monotonic()
            return True


store = JobStore()


def create_job(items: list[str]) -> BatchJob:
    """Backwards-compatible name for creating a classification batch."""
    return store.create_batch(items)


def get_job(job_id: str) -> Job | None:
    return store.get(job_id)
