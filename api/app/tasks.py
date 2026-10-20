"""Background job bodies shared by the in-process runner and the Celery worker.

The functions take a *classifier provider* (a callable returning the classifier) so
the same code serves FastAPI background tasks (provider raises HTTP 503 when the
classifier is unavailable) and the worker process (provider reloads the classifier
when the catalog on the shared volume changed).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from app.job_store import (
    BatchJob,
    add_batch_result,
    audit_classifications,
    complete_batch,
    fail_job,
    get_job,
    start_job,
)

logger = logging.getLogger("dora.tasks")

BATCH_CHUNK_SIZE = 16


def run_batch_job(job_id: str, classifier_provider: Callable[[], object]) -> None:
    """Classify a batch in bounded chunks and publish progress after each product."""
    snapshot = get_job(job_id)
    if not isinstance(snapshot, BatchJob) or not start_job(job_id):
        return

    try:
        classifier = classifier_provider()
        for offset in range(0, len(snapshot.items), BATCH_CHUNK_SIZE):
            names = snapshot.items[offset : offset + BATCH_CHUNK_SIZE]
            started = time.perf_counter()
            results = classifier.classify_many(names)
            per_item_latency = (time.perf_counter() - started) * 1000 / max(len(names), 1)
            published = []
            for name, result in zip(names, results, strict=True):
                item = result.model_copy(
                    update={"query": name, "latency_ms": round(per_item_latency, 3)}
                )
                add_batch_result(job_id, item)
                published.append(item)
            audit_classifications("batch", published)
        complete_batch(job_id)
    except Exception:
        logger.exception("Batch job failed: %s", job_id)
        fail_job(job_id, "Не удалось обработать список товаров; проверьте логи сервиса")
