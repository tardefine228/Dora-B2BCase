"""Dora B2BCase API — product classification for Russian Government Decree No. 1875."""

import asyncio
import csv
import importlib.util
import io
import json
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
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.log_config import configure_logging
from app.core.metrics import metrics as metrics_store
from app.job_store import (
    BatchJob,
    JobStoreCapacityError,
    ReindexAlreadyRunningError,
    ReindexJob,
    add_batch_result,
    complete_batch,
    complete_reindex,
    create_job,
    create_reindex_job,
    fail_job,
    get_job,
    init_job_store,
    start_job,
)
from app.middleware.observe import ObserveMiddleware
from app.middleware.ratelimit import RateLimitMiddleware
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
    SourceFragment,
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
from app.services.source_lookup import (
    SourceUnavailableError,
    get_source_lookup,
    reset_source_lookup,
)

configure_logging(settings.log_level)
logger = logging.getLogger("dora.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm the classifier once; keep liveness available if an optional ML backend fails."""
    interrupted = init_job_store()
    if interrupted:
        logger.warning("После рестарта помечено failed задач: %d", interrupted)
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
# The rate limiter sits inside CORS (so browsers can read a 429) and inside the
# observer (so rejections still get X-Request-ID, logs and metrics).
app.add_middleware(RateLimitMiddleware)
# Add the observer last so it also measures and logs requests rejected by the
# rate limiter and the size limiter.
app.add_middleware(ObserveMiddleware)

def _resolve_frontend_dir(module_file: str | Path = __file__) -> Path | None:
    """Find static UI files in both the source tree and the Docker image.

    In the source tree, ``main.py`` lives at ``<repo>/api/app/main.py`` and
    the UI is at ``<repo>/frontend``. The image copies the app to
    ``/app/app/main.py`` and the UI to ``/app/frontend``.
    """
    module_path = Path(module_file).resolve()
    candidates = (
        module_path.parents[1] / "frontend",  # /app/frontend in the image
        module_path.parents[2] / "frontend",  # <repo>/frontend in source
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


FRONTEND_DIR = _resolve_frontend_dir()
if FRONTEND_DIR is not None:
    app.mount("/ui", StaticFiles(directory=FRONTEND_DIR, html=True), name="ui")
else:
    logger.warning("Frontend directory not found; /ui will not be mounted")


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


# CSV export of a finished batch (matches the client-side exporter in frontend/app.js).
CSV_COLUMNS: tuple[str, ...] = (
    "query",
    "code",
    "title",
    "status",
    "confidence",
    "clause_ref",
    "clause_url",
    "alt1_code",
    "alt1_title",
    "alt2_code",
    "alt2_title",
    "alt3_code",
    "alt3_title",
)
_CSV_ALTERNATIVES = 3
# Excel on a Russian locale expects ";" separators and a BOM in an UTF-8 file.
BATCH_CSV_BOM = "\ufeff"

# Server-sent events for batch progress (see frontend/app.js for the client).
SSE_POLL_SECONDS = 0.25
SSE_HEARTBEAT_SECONDS = 15.0


def _build_batch_csv(results: list[ClassifyResponse]) -> str:
    buffer = io.StringIO()
    writer = csv.writer(buffer, delimiter=";", lineterminator="\r\n")
    writer.writerow(CSV_COLUMNS)
    for result in results:
        row: list[str] = [
            result.query,
            result.code or "",
            result.title or "",
            result.status,
            f"{result.confidence:.4f}",
            result.clause_ref or "",
            result.clause_url or "",
        ]
        alternatives = result.alternatives[:_CSV_ALTERNATIVES]
        for alternative in alternatives:
            row.extend([alternative.code or "", alternative.title or ""])
        row.extend([""] * (2 * (_CSV_ALTERNATIVES - len(alternatives))))
        writer.writerow(row)
    return BATCH_CSV_BOM + buffer.getvalue()


def _sse_event(event: str, payload: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _run_batch_job(job_id: str) -> None:
    """Classify a batch off the event loop and publish progress in bounded chunks."""
    snapshot = get_job(job_id)
    if not isinstance(snapshot, BatchJob) or not start_job(job_id):
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
                add_batch_result(
                    job_id,
                    result.model_copy(
                        update={"query": name, "latency_ms": round(per_item_latency, 3)}
                    ),
                )
        complete_batch(job_id)
    except Exception:
        logger.exception("Batch job failed: %s", job_id)
        fail_job(job_id, "Не удалось обработать список товаров; проверьте логи сервиса")


def _run_reindex_job(job_id: str) -> None:
    snapshot = get_job(job_id)
    if not isinstance(snapshot, ReindexJob) or not start_job(job_id):
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
        complete_reindex(job_id, total)
    except Exception:
        logger.exception("Reindex job failed: %s", job_id)
        fail_job(job_id, "Переиндексация не выполнена; проверьте конфигурацию и логи")


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
    "/classify/batch/{job_id}.csv",
    tags=["classification"],
    summary="Скачать результат батча в CSV",
    response_class=Response,
)
def get_batch_csv(job_id: str) -> Response:
    """CSV (``;``, UTF-8 BOM) for a finished batch; useful for large batches.

    Declared before ``/classify/batch/{job_id}`` because the plain path
    parameter would otherwise swallow the ``.csv`` suffix.
    """
    job = get_job(job_id)
    if not isinstance(job, BatchJob):
        raise HTTPException(status_code=404, detail="Задача классификации не найдена")
    if job.status in ("queued", "running"):
        raise HTTPException(
            status_code=409,
            detail=f"Задача ещё выполняется ({job.completed} из {len(job.items)})",
        )
    return Response(
        content=_build_batch_csv(job.results).encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="batch-{job.job_id}.csv"',
        },
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


@app.get(
    "/classify/batch/{job_id}/events",
    tags=["classification"],
    summary="Прогресс батча Server-Sent Events",
    response_class=StreamingResponse,
)
def stream_batch_events(job_id: str) -> StreamingResponse:
    """``text/event-stream`` with ``progress`` events and a final ``done`` event.

    The store — not process memory — is the source of truth (a job may have been
    created by another worker of the same store), so the stream polls it and
    closes as soon as the job is finished.
    """
    job = get_job(job_id)
    if not isinstance(job, BatchJob):
        raise HTTPException(status_code=404, detail="Задача классификации не найдена")

    async def event_stream():
        last_completed = -1
        idle_seconds = 0.0
        while True:
            current = get_job(job_id)
            if not isinstance(current, BatchJob):
                yield _sse_event("error", {"detail": "Задача классификации не найдена"})
                return
            total = len(current.items)
            if current.completed != last_completed:
                last_completed = current.completed
                idle_seconds = 0.0
                yield _sse_event(
                    "progress",
                    {
                        "job_id": job_id,
                        "status": current.status,
                        "completed": current.completed,
                        "total": total,
                        "progress_percent": round(current.completed * 100 / total, 1)
                        if total
                        else 0.0,
                    },
                )
            if current.status in ("completed", "failed"):
                yield _sse_event(
                    "done",
                    _batch_status(current).model_dump(mode="json"),
                )
                return
            await asyncio.sleep(SSE_POLL_SECONDS)
            idle_seconds += SSE_POLL_SECONDS
            if idle_seconds >= SSE_HEARTBEAT_SECONDS:
                idle_seconds = 0.0
                yield ": keep-alive\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tell nginx and friends not to buffer the stream.
            "X-Accel-Buffering": "no",
        },
    )


@app.get(
    "/source/{record_id}",
    response_model=SourceFragment,
    tags=["sources"],
    summary="Фрагмент текста ПП № 1875 по позиции каталога",
)
def get_source_fragment(record_id: str, response: Response) -> SourceFragment:
    """Human-readable source of record behind ``clause_url`` of a classification.

    ``pp_link`` in the catalog is a citation, not a URL, so the service serves the
    pages of ``data/source_text.txt`` recorded in ``pdf_page_start``/``pdf_page_end``
    for the row ``record_id``.
    """
    try:
        fragment = get_source_lookup().fragment(record_id)
    except SourceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if fragment is None:
        raise HTTPException(
            status_code=404,
            detail="Позиция каталога с таким record_id не найдена",
        )
    response.headers["Cache-Control"] = "public, max-age=3600"
    return fragment


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
        job = create_reindex_job()
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
        reset_source_lookup()
        try:
            from ml.embedder import reset_encoder_cache

            reset_encoder_cache()
        except Exception:  # pragma: no cover - ML stack is optional
            logger.exception("Не удалось сбросить кэш эмбеддера после обновления каталога")
        reindex_required = bool(settings.ml_enabled and settings.classifier_backend == "qdrant")
        if auto_reindex and reindex_required:
            try:
                job = create_reindex_job()
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
