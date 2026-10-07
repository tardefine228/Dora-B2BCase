"""Dora B2BCase API — классификация товаров по ПП № 1875."""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, FastAPI, HTTPException

from app.core.config import settings
from app.core.log_config import configure_logging
from app.core.metrics import metrics as metrics_store
from app.job_store import BatchJob, create_job, get_job
from app.middleware.observe import ObserveMiddleware
from app.schemas import (
    BatchClassifyRequest,
    BatchJobAccepted,
    BatchJobStatusResponse,
    ClassifyRequest,
    ClassifyResponse,
)
from app.services.classifier import get_classifier

configure_logging(settings.log_level)
logger = logging.getLogger("dora.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        get_classifier()
        logger.info("Классификатор готов (backend=%s)", settings.classifier_backend)
    except Exception:
        logger.exception("Не удалось инициализировать классификатор")
    yield


app = FastAPI(
    title="Dora B2BCase API",
    description="API классификатора товаров по ПП № 1875",
    version="0.2.0",
    lifespan=lifespan,
)
app.add_middleware(ObserveMiddleware)


def _job_status_response(job: BatchJob) -> BatchJobStatusResponse:
    total = len(job.items)
    progress_percent = round(job.completed * 100 / total, 1) if total else 0.0
    return BatchJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        total=total,
        completed=job.completed,
        progress_percent=progress_percent,
        results=job.results,
        error=job.error,
    )


@app.get("/health", tags=["system"])
def health() -> dict:
    return {
        "status": "ok",
        "service": "dora-api",
        "classifier_backend": settings.classifier_backend,
        "version": app.version,
    }


@app.get("/metrics", tags=["system"])
def metrics_endpoint() -> dict:
    return metrics_store.snapshot()


@app.post("/classify", response_model=ClassifyResponse, tags=["classification"])
def classify(request: ClassifyRequest) -> ClassifyResponse:
    started = time.perf_counter()
    result = get_classifier().classify(request.name)
    latency_ms = (time.perf_counter() - started) * 1000
    return result.model_copy(update={"query": request.name, "latency_ms": round(latency_ms, 3)})


async def _run_batch_job(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        return
    job.status = "running"
    try:
        classifier = get_classifier()
        for name in job.items:
            started = time.perf_counter()
            result = classifier.classify(name)
            latency_ms = (time.perf_counter() - started) * 1000
            job.results.append(
                result.model_copy(update={"query": name, "latency_ms": round(latency_ms, 3)})
            )
            job.completed += 1
        job.status = "completed"
    except Exception:
        logger.exception("Batch job failed: %s", job_id)
        job.status = "failed"
        job.error = "Не удалось обработать список товаров"


@app.post(
    "/classify/batch",
    response_model=BatchJobAccepted,
    status_code=202,
    tags=["classification"],
)
async def classify_batch(
    request: BatchClassifyRequest,
    background_tasks: BackgroundTasks,
) -> BatchJobAccepted:
    job = create_job(request.items)
    background_tasks.add_task(_run_batch_job, job.job_id)
    return BatchJobAccepted(
        job_id=job.job_id,
        status="queued",
        total=len(job.items),
        status_url=f"/classify/batch/{job.job_id}",
    )


@app.get(
    "/classify/batch/{job_id}",
    response_model=BatchJobStatusResponse,
    tags=["classification"],
)
async def get_batch_status(job_id: str) -> BatchJobStatusResponse:
    job = get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Задача не найдена")
    return _job_status_response(job)
