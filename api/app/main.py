import asyncio
import logging
import time

from fastapi import BackgroundTasks, FastAPI, HTTPException

from app.job_store import BatchJob, create_job, get_job
from app.schemas import (
    BatchClassifyRequest,
    BatchJobAccepted,
    BatchJobStatusResponse,
    ClassifyRequest,
    ClassifyResponse,
)

logger = logging.getLogger("dora.api")

app = FastAPI(
    title="Dora B2BCase API",
    description="API классификатора товаров по ПП № 1875",
    version="0.1.0",
)


def _mock_result(query: str, latency_ms: float = 0) -> ClassifyResponse:
    """Temporary response so the API and frontend can be developed before ML is ready."""
    return ClassifyResponse(
        query=query,
        code="MOCK-001",
        title="Демонстрационный ответ — ML-модуль ещё не подключён",
        confidence=0,
        status="review",
        clause_ref=None,
        clause_url=None,
        alternatives=[],
        latency_ms=latency_ms,
    )


async def _run_batch_job(job_id: str) -> None:
    job = get_job(job_id)
    if job is None:
        return

    job.status = "running"
    try:
        for name in job.items:
            started_at = time.perf_counter()
            result = await asyncio.to_thread(_mock_result, name)
            latency_ms = (time.perf_counter() - started_at) * 1000
            job.results.append(result.model_copy(update={"latency_ms": latency_ms}))
            job.completed += 1
        job.status = "completed"
    except Exception:
        logger.exception("Batch job failed: %s", job_id)
        job.status = "failed"
        job.error = "Не удалось обработать список товаров"


def _job_status_response(job: BatchJob) -> BatchJobStatusResponse:
    progress_percent = round(job.completed * 100 / len(job.items), 1)
    return BatchJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        total=len(job.items),
        completed=job.completed,
        progress_percent=progress_percent,
        results=job.results,
        error=job.error,
    )


@app.middleware("http")
async def log_request_time(request, call_next):
    started_at = time.perf_counter()
    response = await call_next(request)
    latency_ms = (time.perf_counter() - started_at) * 1000
    response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
    logger.info(
        "%s %s status=%s latency_ms=%.2f",
        request.method,
        request.url.path,
        response.status_code,
        latency_ms,
    )
    return response


@app.get("/health", tags=["system"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/classify", response_model=ClassifyResponse, tags=["classification"])
def classify(request: ClassifyRequest) -> ClassifyResponse:
    started_at = time.perf_counter()
    result = _mock_result(request.name)
    latency_ms = (time.perf_counter() - started_at) * 1000
    return result.model_copy(update={"latency_ms": latency_ms})


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
