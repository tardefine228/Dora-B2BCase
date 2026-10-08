"""Dora B2BCase API — product classification for Russian Government Decree No. 1875."""

import importlib.util
import logging
import secrets
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

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
    CatalogInfo,
    CatalogUploadResponse,
    ClassifyRequest,
    ClassifyResponse,
    ReindexJobAccepted,
    ReindexJobStatusResponse,
)
from app.services.catalog_io import (
    catalog_facts,
    normalize_rows,
    parse_catalog_bytes,
    resolve_data_dir,
    validate_rows,
    write_catalog,
)
from app.services.classifier import CachedClassifier, get_classifier, reset_classifier

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
    version="0.4.0",
    lifespan=lifespan,
)
app.add_middleware(
    RequestSizeLimitMiddleware,
    max_body_bytes=settings.max_request_body_bytes,
)
if settings.cors_allow_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allow_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
# Add the observer last so it also measures and logs requests rejected by the size limiter.
app.add_middleware(ObserveMiddleware)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIR = PROJECT_ROOT / "frontend"
if FRONTEND_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")


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


def _ml_dependencies_missing() -> list[str]:
    """Only require the heavy dependencies of the currently configured embedder."""
    missing: list[str] = []
    if importlib.util.find_spec("qdrant_client") is None:
        missing.append("qdrant-client")
    try:
        from ml.embedder import resolve_backend

        if resolve_backend() == "sentence-transformers" and (
            importlib.util.find_spec("sentence_transformers") is None
        ):
            missing.append("sentence-transformers")
    except Exception:  # ml package problems surface as an unmet dependency
        missing.append("ml")
    return missing


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Open the demo UI by default; API docs stay available at /docs."""
    return RedirectResponse(url="/ui/", status_code=307)


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
    missing_ml_dependency = _ml_dependencies_missing()
    if not settings.ml_enabled or missing_ml_dependency:
        raise HTTPException(
            status_code=503,
            detail=(
                "Переиндексация недоступна: установите ML-зависимости "
                f"({', '.join(missing_ml_dependency) or 'INSTALL_ML=true'}) "
                "и включите ML_ENABLED=true."
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


@app.get(
    "/admin/catalog",
    response_model=CatalogInfo,
    tags=["admin"],
    summary="Информация об установленном каталоге",
    dependencies=[Depends(require_admin_key)],
)
def get_catalog_info() -> CatalogInfo:
    return catalog_facts(resolve_data_dir())


@app.post(
    "/admin/catalog",
    response_model=CatalogUploadResponse,
    tags=["admin"],
    summary="Загрузить и проверить новый каталог (CSV или JSON)",
    dependencies=[Depends(require_admin_key)],
)
async def upload_catalog(
    background_tasks: BackgroundTasks,
    file: Annotated[
        UploadFile,
        File(description="data.csv или data.json с перечнем ПП № 1875"),
    ],
    dry_run: bool = Query(
        default=False,
        description="Только проверить файл, не заменяя установленный каталог.",
    ),
    auto_reindex: bool = Query(
        default=False,
        description="Сразу запустить переиндексацию Qdrant (только при ML-режиме).",
    ),
) -> CatalogUploadResponse:
    content = await file.read()
    try:
        parsed = parse_catalog_bytes(file.filename or "", content)
        report = validate_rows(parsed)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    applied = False
    catalog_info = None
    reindex_required = False

    if report.ok and not dry_run:
        rows = normalize_rows(parsed)
        catalog_info = write_catalog(rows, resolve_data_dir())
        applied = True
        # Drop caches so the next request sees the new catalog immediately.
        reset_classifier()
        try:
            from ml.embedder import reset_encoder_cache

            reset_encoder_cache()
        except Exception:  # pragma: no cover - ML stack is optional
            logger.exception("Не удалось сбросить кэш эмбеддера после обновления каталога")
        reindex_required = bool(settings.ml_enabled and settings.classifier_backend == "qdrant")
        if auto_reindex and reindex_required:
            try:
                job = job_store.create_reindex()
                background_tasks.add_task(_run_reindex_job, job.job_id)
            except (ReindexAlreadyRunningError, JobStoreCapacityError):
                # The catalog is already installed; the operator can reindex later.
                logger.exception("Переиндексация после загрузки каталога не запущена")

    logger.info(
        "Catalog upload finished filename=%s dry_run=%s applied=%s rows=%d errors=%d",
        file.filename,
        dry_run,
        applied,
        report.rows_total,
        len(report.errors),
    )
    return CatalogUploadResponse(
        dry_run=dry_run,
        applied=applied,
        rows_total=report.rows_total,
        rows_valid=report.rows_valid,
        errors=report.errors,
        warnings=report.warnings,
        catalog=catalog_info,
        reindex_required=reindex_required,
    )
