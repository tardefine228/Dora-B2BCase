"""Celery worker for heavy batches (``QUEUE_BACKEND=celery``, broker = Redis).

Run:  celery -A app.worker.celery_app worker --loglevel=INFO --concurrency=1

Job state is kept in PostgreSQL (``JOB_STORE=postgres``), not in Celery's result
backend: the API reads progress from the same table the worker writes to.
"""

from __future__ import annotations

import logging

from celery import Celery
from celery.signals import worker_process_init

from app.core.config import settings
from app.core.log_config import configure_logging
from app.dispatch import BATCH_TASK_NAME
from app.services.classifier import get_classifier
from app.tasks import run_batch_job

configure_logging(settings.log_level, settings.log_format)
logger = logging.getLogger("dora.worker")

celery_app = Celery("dora", broker=settings.redis_url)
celery_app.conf.update(
    task_acks_late=True,  # a crashed worker returns the job to the queue
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,  # one heavy batch at a time per worker process
    task_time_limit=3600,
    broker_connection_retry_on_startup=True,
    task_ignore_result=True,
)


@worker_process_init.connect
def _warm_classifier(**_kwargs) -> None:
    """Load the classifier (embedder + vector index) before the first task arrives."""
    try:
        get_classifier()
        logger.info("Классификатор в воркере готов")
    except Exception:
        logger.exception("Классификатор в воркере не загрузился; повтор при первой задаче")


@celery_app.task(name=BATCH_TASK_NAME)
def classify_batch(job_id: str) -> None:
    # get_classifier() rebuilds itself when the catalog on the shared volume changed.
    run_batch_job(job_id, get_classifier)
