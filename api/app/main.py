"""Dora B2BCase API — product classification for Russian Decree No. 1875."""

import importlib.util
import logging
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException
from fastapi.responses import RedirectResponse

from app.core.config import settings
from app.core.log_config import configure_logging
from app.core.metrics import metrics as metrics_store
from app.job_store import (
    BatchJob,
    JobStoreCapacityError,
    ReindexAlreadyRunningError,
    ReindexJob,
    create_job,
    get_job,
)
from app.job_store import (
    store as job_store,
)
from app.middleware.observe import ObserveMiddleware
from app.middleware.size_limit import RequestSizeLimitMiddleware
from app.schemas import (
    BatchClassifyRequest,
    BatchJobAccepted,
    BatchJobStatusResponse,
    ClassifyRequest,
    ClassifyResponse,
    ReindexJobAccepted,
    ReindexJobStatusResponse,
)
from app.services.classifier import CachedClassifier, get_classifier

configure_logging(settings.log_level)
logger = logging.getLogger("dora.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the classifier once; keep liveness available if an optional ML backend fails."""
    app.state.classifier_initialization_attempted = True
    app.state.classifier_error = None
    try:
        get_classifier()
        logger.info("Классификатор готов (backend=%s)", settings.classifier_backend)
    except Exception as exc:
        app.state.classifier_error = str(exc)
        logger.exception("Не удалось инициализировать классификатор")
    yield


app = FastAPI(
    title="Dora B2BCase API",
    description=(
        "API классификатора товаров по ПП № 1875. "
        "Автоматическая классификация является подсказкой и не заменяет проверку специалиста."
    ),
    version="0.3.0",
    lifespan=lifespan,
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_bytes=settings.max_request_body_bytes,
)
# Add the observer last so it also measures and logs requests rejected by the size limiter.
app.add_middleware(ObserveMiddleware)


def _classifier_or_503() -> CachedClassifier:
    if getattr(app.state, "classifier_initialization_attempted", False) and getattr(
        app.state, "classifier_error", None
    ):
        raise HTTPException(
            status_code=503,
            detail="Классификатор временно недоступен; проверьте готовность сервиса и логи.",
        )
    try:
        classifier = get_classifier()
    except Exception as exc:
        app.state.classifier_error = str(exc)
        logger.exception("Классификатор недоступен")
        raise HTTPException(
            status_code=503,
            detail="Классификатор временно недоступен; проверьте готовность сервиса и логи.",
        ) from exc
    app.state.classifier_initialization_attempted = True
    app.state.classifier_error = None
    return classifier


def _batch_status(job: BatchJob) -> BatchJobStatusResponse:
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


def _run_batch_job(job_id: str) -> None:
    """Classify a batch off the event loop and publish progress in bounded chunks."""
    snapshot = get_job(job_id)
    if not isinstance(snapshot, BatchJob) or not job_store.start(job_id):
        return

    try:
        classifier = _classifier_or_503()
        chunk_size = 16
        for offset in range(0, len(snapshot.items), chunk_size):
            names = snapshot.items[offset : offset + chunk_size]
            started = time.perf_counter()
            results = classifier.classify_many(names)
            per_item_latency = (time.perf_counter() - started) * 1000 / max(len(names), 1)
            for name, result in zip(names, results, strict=True):
                job_store.add_batch_result(
                    job_id,
                    result.model_copy(
                        update={"query": name, "latency_ms": round(per_item_latency, 3)}
                    ),
                )
        job_store.complete_batch(job_id)
    except Exception:
        logger.exception("Batch job failed: %s", job_id)
        job_store.fail(job_id, "Не удалось обработать список товаров; проверьте логи сервиса")


def _run_reindex_job(job_id: str) -> None:
    snapshot = get_job(job_id)
    if not isinstance(snapshot, ReindexJob) or not job_store.start(job_id):
        return

    try:
        from app.services.classifier import exclusive_inference

        with exclusive_inference():
            total = _do_reindex()
            try:
                get_classifier().clear_cache()
                app.state.classifier_initialization_attempted = True
                app.state.classifier_error = None
            except Exception:
                # The Qdrant index was published; a cache reset must not mark it failed.
                logger.exception("Index is ready, but the classifier cache could not be cleared")
        job_store.complete_reindex(job_id, total)
    except Exception:
        logger.exception("Reindex job failed: %s", job_id)
        job_store.fail(job_id, "Переиндексация не выполнена; проверьте конфигурацию и логи")


def _do_reindex() -> int:
    """Import heavyweight optional ML dependencies only inside the background worker."""
    from ml.indexer import index_catalog

    return index_catalog()


def require_admin_key(
    x_admin_api_key: str | None = Header(default=None, alias="X-Admin-API-Key"),
) -> None:
    expected_key = settings.admin_api_key
    if not expected_key:
        raise HTTPException(
            status_code=503,
            detail="Административные эндпоинты отключены: задайте ADMIN_API_KEY.",
        )
    if not x_admin_api_key or not secrets.compare_digest(x_admin_api_key, expected_key):
        raise HTTPException(
            status_code=403,
            detail="Неверный или отсутствующий API-ключ администратора",
        )


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Make the live API preview open its interactive documentation by default."""
    return RedirectResponse(url="/docs", status_code=307)


@app.get("/health", tags=["system"], summary="Liveness probe")
def health() -> dict:
    """Cheap liveness check; it does not load models or require Qdrant."""
    return {
        "status": "ok",
        "service": "dora-api",
        "classifier_backend": settings.classifier_backend,
        "version": app.version,
    }


@app.get("/health/ready", tags=["system"], summary="Readiness probe")
def readiness() -> dict:
    """Ready only when the configured classifier and its required resources are usable."""
    try:
        get_classifier()
    except Exception as exc:
        app.state.classifier_initialization_attempted = True
        app.state.classifier_error = str(exc)
        logger.exception("Readiness check failed")
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "classifier_backend": settings.classifier_backend,
                "message": "Классификатор не инициализирован; проверьте конфигурацию и логи.",
            },
        ) from exc

    app.state.classifier_initialization_attempted = True
    app.state.classifier_error = None
    return {
        "status": "ready",
        "service": "dora-api",
        "classifier_backend": settings.classifier_backend,
        "version": app.version,
    }


@app.get("/metrics", tags=["system"])
def metrics_endpoint() -> dict:
    return metrics_store.snapshot()


@app.post(
    "/classify",
    response_model=ClassifyResponse,
    tags=["classification"],
    summary="Классифицировать один товар",
)
def classify(request: ClassifyRequest) -> ClassifyResponse:
    started = time.perf_counter()
    result = _classifier_or_503().classify(request.name)
    latency_ms = (time.perf_counter() - started) * 1000
    return result.model_copy(update={"query": request.name, "latency_ms": round(latency_ms, 3)})


@app.post(
    "/classify/batch",
    response_model=BatchJobAccepted,
    status_code=202,
    tags=["classification"],
    summary="Запустить фоновую классификацию списка товаров",
)
def classify_batch(
    request: BatchClassifyRequest,
    background_tasks: BackgroundTasks,
) -> BatchJobAccepted:
    _classifier_or_503()
    try:
        job = create_job(request.items)
    except JobStoreCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc

    logger.info("Batch job accepted job_id=%s batch_size=%d", job.job_id, len(job.items))
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
    summary="Получить статус фоновой классификации",
)
def get_batch_status(job_id: str) -> BatchJobStatusResponse:
    job = get_job(job_id)
    if not isinstance(job, BatchJob):
        raise HTTPException(status_code=404, detail="Задача классификации не найдена")
    return _batch_status(job)


@app.post(
    "/admin/reindex",
    response_model=ReindexJobAccepted,
    status_code=202,
    tags=["admin"],
    summary="Переиндексировать каталог в Qdrant",
    dependencies=[Depends(require_admin_key)],
)
def start_reindex(background_tasks: BackgroundTasks) -> ReindexJobAccepted:
    missing_ml_dependency = any(
        importlib.util.find_spec(module_name) is None
        for module_name in ("sentence_transformers", "qdrant_client")
    )
    if not settings.ml_enabled or missing_ml_dependency:
        raise HTTPException(
            status_code=503,
            detail=(
                "Переиндексация недоступна: установите ML-зависимости "
                "(INSTALL_ML=true) и включите ML_ENABLED=true."
            ),
        )

    try:
        job = job_store.create_reindex()
    except ReindexAlreadyRunningError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except JobStoreCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc

    logger.info("Reindex job accepted job_id=%s", job.job_id)
    background_tasks.add_task(_run_reindex_job, job.job_id)
    return ReindexJobAccepted(
        job_id=job.job_id,
        status="queued",
        status_url=f"/admin/reindex/{job.job_id}",
    )


@app.get(
    "/admin/reindex/{job_id}",
    response_model=ReindexJobStatusResponse,
    tags=["admin"],
    summary="Получить статус переиндексации",
    dependencies=[Depends(require_admin_key)],
)
def get_reindex_status(job_id: str) -> ReindexJobStatusResponse:
    job = get_job(job_id)
    if not isinstance(job, ReindexJob):
        raise HTTPException(status_code=404, detail="Задача переиндексации не найдена")
    return ReindexJobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        total=job.total,
        indexed=job.indexed,
        error=job.error,
    )
