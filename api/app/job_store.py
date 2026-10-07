from dataclasses import dataclass, field
from typing import Literal
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


_jobs: dict[str, BatchJob] = {}


def create_job(items: list[str]) -> BatchJob:
    job = BatchJob(job_id=str(uuid4()), items=items)
    _jobs[job.job_id] = job
    return job


def get_job(job_id: str) -> BatchJob | None:
    return _jobs.get(job_id)
