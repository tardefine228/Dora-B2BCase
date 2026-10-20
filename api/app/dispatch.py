"""Hand a batch job to its executor: FastAPI background task or Celery worker."""

from __future__ import annotations

import logging
from collections.abc import Callable

from fastapi import BackgroundTasks, HTTPException

from app.core.config import settings
from app.job_store import fail_job

logger = logging.getLogger("dora.dispatch")

BATCH_TASK_NAME = "dora.classify_batch"


def enqueue_batch(
    job_id: str,
    background_tasks: BackgroundTasks,
    inprocess_runner: Callable[[str], None],
) -> str:
    """Start ``job_id`` and return the executor name (``celery`` or ``inprocess``).

    With ``QUEUE_BACKEND=celery`` the job goes to Redis and a worker picks it up, so a
    heavy batch never occupies the API process. If the broker is unreachable the job
    is failed at once and the client gets a 503 instead of a job that never starts.
    """
    if settings.queue_backend == "celery":
        try:
            from app.worker import celery_app

            celery_app.send_task(BATCH_TASK_NAME, args=[job_id])
            return "celery"
        except Exception as exc:
            logger.exception("Очередь Celery/Redis недоступна, job_id=%s", job_id)
            fail_job(job_id, "Очередь задач недоступна")
            raise HTTPException(
                status_code=503,
                detail="Очередь задач временно недоступна; повторите запрос позже.",
            ) from exc
    background_tasks.add_task(inprocess_runner, job_id)
    return "inprocess"
