"""Dora B2BCase API — product classification for Russian Government Decree No. 1875."""

import asyncio
import csv
import importlib.util
import io
import json
import logging
import secrets
import time
from contextlib import asynccontextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    Header,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.log_config import configure_logging
from app.core.metrics import (
    metrics as metrics_store,
)
from app.core.metrics import (
    prometheus_latest,
    refresh_health_gauges,
    set_build_info,
)
from app.dispatch import enqueue_batch
from app.job_store import (
    BatchJob,
    JobStoreCapacityError,
    ReindexAlreadyRunningError,
    ReindexJob,
    audit_classifications,
    complete_reindex,
    create_job,
    create_reindex_job,
    fail_job,
    get_job,
    init_job_store,
    start_job,
)
from app.maintenance import purge_expired
from app.middleware.observe import ObserveMiddleware
from app.middleware.ratelimit import RateLimitMiddleware
from app.middleware.size_limit import RequestSizeLimitMiddleware
from app.schemas import (
    BatchClassifyRequest,
    BatchJobAccepted,
    BatchJobStatusResponse,
    CatalogInfo,
    CatalogRollbackResponse,
    CatalogUploadResponse,
    CatalogVersion,
    CatalogVersionsResponse,
    ClassifyRequest,
    ClassifyResponse,
    FeedbackRequest,
    FeedbackSaved,
    FileBatchJobAccepted,
    MaintenanceReport,
    ReindexJobAccepted,
    ReindexJobStatusResponse,
    SourceFragment,
)
from app.services.catalog_io import (
    CatalogVersion as CatalogVersionRecord,
)
from app.services.catalog_io import (
    catalog_facts,
    export_catalog_csv,
    list_versions,
    normalize_rows,
    parse_catalog_bytes,
    resolve_data_dir,
    restore_version,
    validate_rows,
    write_catalog,
)
from app.services.classifier import (
    CachedClassifier,
    get_classifier,
    reset_classifier,
    vector_layer_health,
)
from app.services.feedback_store import export_feedback, list_feedback, save_feedback
from app.services.file_items import (
    SUPPORTED_EXTENSIONS,
    FileParseError,
    UnsupportedFileError,
    extract_items,
)
from app.services.source_lookup import (
    SourceUnavailableError,
    get_source_lookup,
    reset_source_lookup,
)
from app.tasks import run_batch_job

configure_logging(settings.log_level)
logger = logging.getLogger("dora.api")

# Where the specialist can verify the decree text served from data/source_text.txt:
# (title, url, note) triplets rendered on the /source/{record_id} HTML page.
DECREE_OFFICIAL_SOURCES: tuple[tuple[str, str, str], ...] = (
    (
        "Официальный интернет-портал правовой информации",
        "http://publication.pravo.gov.ru/document/0001202412250018",
        "Официальное опубликование 25.12.2024, № 0001202412250018",
    ),
    (
        "Портал Правительства России",
        "http://government.ru/docs/all/157130/",
        "Подписан 23.12.2024; там же PDF-оригинал (32 МБ)",
    ),
    (
        "Действующая редакция (Контур.Норматив)",
        "https://normativ.kontur.ru/document?moduleId=1&documentId=507717",
        "Текст с изменениями — сверяйте дату редакции перед применением",
    ),
)


async def maintenance_loop(interval_seconds: float | None = None) -> None:
    """Trim the audit and feedback logs, then once per interval (BE-5).

    Runs immediately: a deployment that has been up for months gets its old rows
    removed on restart, not after another day. Failures are logged and retried on
    the next tick — retention is housekeeping, it must never take the API down.
    """
    interval = (
        interval_seconds
        if interval_seconds is not None
        else settings.maintenance_interval_hours * 3600
    )
    while True:
        try:
            await asyncio.to_thread(purge_expired)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - housekeeping must not kill the loop
            logger.exception("Очистка по сроку хранения не удалась, повтор через интервал")
        await asyncio.sleep(interval)


def _maintenance_enabled() -> bool:
    """Only PostgreSQL keeps the two logs, and only with a configured retention."""
    if settings.job_store != "postgres":
        return False
    return settings.audit_retention_days > 0 or settings.feedback_retention_days > 0


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
    housekeeping = asyncio.create_task(maintenance_loop()) if _maintenance_enabled() else None
    try:
        yield
    finally:
        if housekeeping is not None:
            housekeeping.cancel()
            with suppress(asyncio.CancelledError):
                await housekeeping


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
    path_limits={
        "/classify/file": settings.max_upload_bytes,
        "/admin/catalog": max(settings.max_request_body_bytes, settings.max_upload_bytes),
    },
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

# Pin the running build/configuration as an info gauge at import time so it is
# present in every scrape, even before the lifespan has run (BE-6).
set_build_info(app.version)


def _resolve_frontend_dir(module_file: str | Path = __file__) -> Path | None:
    """Find static UI files in both the source tree and the Docker image.

    In the source tree, ``main.py`` lives at ``<repo>/api/app/main.py`` and
    the UI is at ``<repo>/frontend`` (built output: ``<repo>/frontend/dist``).
    The image copies the app to ``/app/app/main.py`` and the built UI
    (``dist``) to ``/app/frontend``.
    """
    module_path = Path(module_file).resolve()
    candidates = (
        module_path.parents[1] / "frontend",  # /app/frontend in the image (built dist)
        module_path.parents[2] / "frontend" / "dist",  # <repo>/frontend/dist after npm run build
        module_path.parents[2] / "frontend",  # <repo>/frontend (unbuilt sources, dev fallback)
    )
    return next((candidate for candidate in candidates if candidate.is_dir()), None)


FRONTEND_DIR = _resolve_frontend_dir()
if FRONTEND_DIR is not None:
    if (FRONTEND_DIR / "src" / "main.tsx").is_file():
        logger.warning(
            "/ui serves UNBUILT frontend sources from %s; run `npm run build` in frontend/",
            FRONTEND_DIR,
        )
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


# CSV export of a finished batch (matches the client-side exporter in frontend/src).
# ``code_role``/``catalog_review_required`` (BE-9) are catalog facts the specialist
# must see in the export as well; empty cells mean "catalog does not say".
CSV_COLUMNS: tuple[str, ...] = (
    "query",
    "code",
    "title",
    "status",
    "confidence",
    "confidence_raw",
    "clause_ref",
    "clause_url",
    "code_role",
    "catalog_review_required",
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


def _csv_flag(value: bool | None) -> str:
    return "" if value is None else ("True" if value else "False")

# Server-sent events for batch progress (see frontend/src for the client).
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
            f"{result.confidence_raw:.4f}" if result.confidence_raw is not None else "",
            result.clause_ref or "",
            result.clause_url or "",
            result.code_role or "",
            _csv_flag(result.catalog_review_required),
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
    """In-process runner (``QUEUE_BACKEND=inprocess``); Celery runs the same body."""
    run_batch_job(job_id, _classifier_or_503)


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


def _vector_info() -> dict:
    """What the vector layer is actually doing (embedder, dimension, scoring mode)."""
    inner = getattr(get_classifier(), "_classifier", None)
    info = getattr(inner, "vector_info", None)
    return info if isinstance(info, dict) else {"enabled": False}


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
        "vector_search": _vector_info(),
        # Live health of the embedder and the vector store (BE-6): TTL-cached,
        # informational — a degraded store does not make the service unready
        # (search falls back to memory), but it must be visible to operators.
        "vector_health": vector_layer_health(),
        "job_store": settings.job_store,
        "queue_backend": settings.queue_backend,
        # Third stage (эксп. 32): visible to operators, because "why is this code
        # marked verified" must be answerable without reading the logs. `off` is
        # the normal state of a deployment without an LLM key.
        "llm_verifier": (
            "off"
            if not settings.llm_verifier_enabled or not settings.llm_verifier_model
            else f"on:{settings.llm_verifier_model}"
        ),
    }


@app.get("/metrics", tags=["system"], summary="Метрики: Prometheus (по умолчанию) или JSON")
def metrics_endpoint(request: Request) -> Response:
    """Operational metrics for scraping (BE-6).

    Default exposition is the Prometheus text format (``text/plain;
    version=0.0.4``): counters by path/method/status, a latency histogram,
    rate-limit rejections, classification answers by status, batch job events
    and health gauges of the classifier, embedder and vector store. Clients
    sending ``Accept: application/json`` get the legacy in-process snapshot
    plus the same health block under ``vector_health``.

    The health probe is TTL-cached (10 s), so scraping never turns into a
    network ping per scrape even when Qdrant is down.
    """
    health = vector_layer_health()
    refresh_health_gauges(health)
    if "application/json" in request.headers.get("accept", ""):
        payload = metrics_store.snapshot()
        payload["vector_health"] = health
        return JSONResponse(payload)
    body, content_type = prometheus_latest()
    return Response(content=body, media_type=content_type)


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
    answer = result.model_copy(update={"query": request.name, "latency_ms": round(latency_ms, 3)})
    metrics_store.record_classification("single", answer.status)
    audit_classifications("single", [answer])
    return answer


@app.post(
    "/feedback",
    response_model=FeedbackSaved,
    tags=["classification"],
    summary="Сохранить проверку кода специалистом",
    status_code=201,
)
def submit_feedback(payload: FeedbackRequest) -> FeedbackSaved:
    if payload.verdict == "incorrect" and not (payload.corrected_code or "").strip():
        raise HTTPException(status_code=422, detail="Введите правильный код ОКПД2")
    try:
        record_id = save_feedback(
            payload.query,
            payload.predicted_code,
            payload.verdict,
            payload.corrected_code.strip()
            if payload.verdict == "incorrect" and payload.corrected_code
            else None,
            (payload.comment or "").strip() or None,
        )
    except Exception as exc:
        logger.exception("Не удалось сохранить feedback")
        raise HTTPException(
            status_code=503, detail="Не удалось сохранить оценку. Повторите позже."
        ) from exc
    return FeedbackSaved(id=record_id)


@app.post(
    "/admin/maintenance/purge",
    response_model=MaintenanceReport,
    tags=["admin"],
    summary="Применить срок хранения журнала запросов и вердиктов (BE-5)",
    dependencies=[Depends(require_admin_key)],
)
def purge_logs(
    dry_run: Annotated[
        bool,
        Query(description="Только посчитать, сколько строк попадёт под очистку."),
    ] = False,
) -> MaintenanceReport:
    """Удалить строки старше `AUDIT_RETENTION_DAYS` / `FEEDBACK_RETENTION_DAYS`.

    Планировщик делает это сам при старте и раз в `MAINTENANCE_INTERVAL_HOURS`;
    эндпоинт нужен, чтобы проверить настройку (`dry_run=true`) и не ждать сутки.
    """
    try:
        report = purge_expired(dry_run=dry_run)
    except Exception as exc:  # noqa: BLE001 - psycopg raises different errors
        logger.exception("Очистка по сроку хранения не удалась")
        raise HTTPException(
            status_code=503, detail="Не удалось применить срок хранения. Повторите позже."
        ) from exc
    return MaintenanceReport(**report.as_dict())


@app.get(
    "/admin/feedback/export",
    tags=["admin"],
    summary="Выгрузить вердикты специалистов для обучения",
    response_class=Response,
    responses={
        200: {
            "content": {"text/csv": {}, "application/x-ndjson": {}},
            "description": "Файл выгрузки (вложение).",
        },
        403: {"description": "Нет или неверный X-Admin-API-Key."},
        503: {"description": "Административные эндпоинты отключены (нет ADMIN_API_KEY)."},
    },
    dependencies=[Depends(require_admin_key)],
)
def export_feedback_log(
    since: Annotated[
        datetime | None,
        Query(description="Начало периода (ISO 8601), включительно."),
    ] = None,
    until: Annotated[
        datetime | None,
        Query(description="Конец периода (ISO 8601), включительно."),
    ] = None,
    verdict: Annotated[
        Literal["correct", "incorrect", "absent"] | None,
        Query(description="Только один тип вердикта."),
    ] = None,
    fmt: Annotated[
        Literal["csv", "jsonl"],
        Query(alias="format", description="csv — для Excel и обучения, jsonl — построчно."),
    ] = "csv",
    limit: Annotated[
        int | None,
        Query(ge=1, le=100_000, description="Не больше N последних записей."),
    ] = None,
) -> Response:
    """Отдать вердикты в формате, который читает `ml.hard_negatives.load_feedback`.

    До появления этого эндпоинта вердикты писались в БД и прочитать их было
    нечем, поэтому hard-negative майнинг (ML-4) не мог начать работать: механизм
    есть, данных нет. Колонки совпадают с ожиданиями загрузчика, так что файл
    уходит в обучение без правок:

        GET /admin/feedback/export?format=csv > feedback.csv
        python -m ml.train_reranker --feedback feedback.csv
    """
    try:
        rows = list_feedback(since=since, until=until, verdict=verdict, limit=limit)
    except Exception as exc:  # noqa: BLE001 - storage backends raise different errors
        logger.exception("Не удалось выгрузить вердикты")
        raise HTTPException(
            status_code=503, detail="Хранилище вердиктов недоступно. Повторите позже."
        ) from exc
    body = export_feedback(rows, fmt)
    extension = "jsonl" if fmt == "jsonl" else "csv"
    media_type = "application/x-ndjson" if fmt == "jsonl" else "text/csv"
    filename = f"feedback-{datetime.now(UTC).strftime('%Y%m%d-%H%M')}.{extension}"
    headers = {
        "Content-Disposition": f'attachment; filename="{filename}"',
        # A browser download of a possibly large log should not be cached by proxies.
        "Cache-Control": "no-store",
        "X-Feedback-Rows": str(len(rows)),
    }
    return Response(
        content=body.encode("utf-8"), media_type=f"{media_type}; charset=utf-8", headers=headers
    )


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
    metrics_store.record_batch_job("accepted")
    enqueue_batch(job.job_id, background_tasks, _run_batch_job)
    return BatchJobAccepted(
        job_id=job.job_id,
        status="queued",
        total=len(job.items),
        status_url=f"/classify/batch/{job.job_id}",
    )


@app.post(
    "/classify/file",
    response_model=FileBatchJobAccepted,
    status_code=202,
    tags=["classification"],
    summary="Загрузить файл со списком товаров и классифицировать каждую позицию",
)
async def classify_file(
    background_tasks: BackgroundTasks,
    file: Annotated[
        UploadFile,
        File(description="Список товаров: " + ", ".join(sorted(SUPPORTED_EXTENSIONS))),
    ],
) -> FileBatchJobAccepted:
    """Parse the file, extract product names and start a regular batch job.

    Progress/results are served by the usual ``/classify/batch/{job_id}`` endpoints.
    """
    _classifier_or_503()
    filename = file.filename or ""
    content = await file.read()
    try:
        extracted = extract_items(filename, content, max_items=100)
    except UnsupportedFileError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc
    except FileParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        job = create_job(extracted.items)
    except JobStoreCapacityError as exc:
        raise HTTPException(status_code=429, detail=str(exc), headers={"Retry-After": "5"}) from exc

    logger.info(
        "File batch accepted job_id=%s filename=%s format=%s items=%d found=%d",
        job.job_id,
        filename,
        extracted.format,
        len(job.items),
        extracted.total_found,
    )
    metrics_store.record_batch_job("accepted")
    enqueue_batch(job.job_id, background_tasks, _run_batch_job)
    return FileBatchJobAccepted(
        job_id=job.job_id,
        status="queued",
        total=len(job.items),
        status_url=f"/classify/batch/{job.job_id}",
        filename=filename,
        format=extracted.format,
        items=extracted.items,
        total_found=extracted.total_found,
        duplicates=extracted.duplicates,
        skipped=extracted.skipped,
        truncated=extracted.truncated,
        source_column=extracted.source_column,
        warnings=extracted.warnings,
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
def get_source_fragment(
    record_id: str, request: Request, response: Response
) -> SourceFragment | HTMLResponse:
    """Human-readable source of record behind ``clause_url`` of a classification.

    ``pp_link`` in the catalog is a citation, not a URL, so the service serves the
    pages of ``data/source_text.txt`` recorded in ``pdf_page_start``/``pdf_page_end``
    for the row ``record_id``. Browsers (``Accept: text/html``) get a readable page
    because the classification table links here for people; API clients keep JSON.
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
    if "text/html" in request.headers.get("accept", ""):
        return HTMLResponse(
            _source_page(fragment), headers={"Cache-Control": "no-store"}
        )
    return fragment


def _render_decree(text: str, code: str) -> str:
    """Structure the decree pages: numbered positions, codes as badges."""
    import re
    from html import escape

    entry_re = re.compile(r"^(\d{1,5})\.\s+(.*)$")
    code_re = re.compile(r"\s(\d+(?:\.\d+){1,4})$")
    # Remove repeated OCR table headings from the presentation only.
    text = re.sub(
        r"(?m)^(?:[ \t]*\d+[ \t]*\n)?[ \t]*Код товара[^\n]*\n"
        r"(?:(?!\s*\d+\.\s|=== СТРАНИЦА)[^\n]*\n){0,12}?"
        r"[ \t]*КПЕС[ \t]+2008[^\n]*",
        "=== TABLE HEADER ===", text,
    )
    out: list[str] = []
    # Join PDF line wraps before rendering each numbered position.
    lines: list[str] = []
    for raw in text.splitlines():
        value = raw.strip()
        if not value:
            continue
        previous_is_position = bool(lines) and bool(entry_re.match(lines[-1]))
        if (
            previous_is_position
            and not entry_re.match(value)
            and not value.startswith("===")
            and lines[-1] != "=== TABLE HEADER ==="
        ):
            # Keep the code at the end while appending wrapped name text.
            match = code_re.search(lines[-1])
            if match:
                lines[-1] = lines[-1][:match.start()] + " " + value + " " + match.group(1)
            else:
                lines[-1] += " " + value
        else:
            lines.append(value)
    for raw in lines:
        line = raw.rstrip()
        if not line.strip():
            continue
        if line == "=== TABLE HEADER ===":
            continue
        page = re.match(r"^=== СТРАНИЦА (\d+) ===$", line.strip())
        if page:
            out.append(f'<div class="page-break">страница {escape(page.group(1))}</div>')
            continue
        entry = entry_re.match(line.strip())
        if entry:
            body = entry.group(2)
            hit_code = ""
            coded = code_re.search(body)
            if coded:
                hit_code = coded.group(1)
                body = body[: coded.start()]
            hit = " hit" if hit_code and hit_code == code else ""
            badge = f'<span class="code">{escape(hit_code)}</span>' if hit_code else ""
            out.append(
                f'<div class="pos{hit}"><span class="num">{escape(entry.group(1))}.</span>'
                f'<span class="name">{escape(body.strip())}</span>{badge}</div>'
            )
        else:
            out.append(f'<div class="cont">{escape(line.strip())}</div>')
    return "\n".join(out)


def _source_page(fragment: SourceFragment) -> str:
    from html import escape

    meta_items = "".join(
        f"<li>{part}</li>"
        for part in (
            escape(fragment.okpd2) if fragment.okpd2 else "",
            escape(fragment.clause_ref) if fragment.clause_ref else "",
            f"стр. {fragment.pages.start}–{fragment.pages.end}" if fragment.pages.end else "",
        )
        if part
    )
    sources = "\n".join(
        f'<li><a href="{url}">{escape(title)}</a>'
        f"<span>{escape(note)}</span></li>"
        for title, url, note in DECREE_OFFICIAL_SOURCES
    )
    decree = _render_decree(fragment.text, fragment.okpd2 or "")
    return f"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="icon" type="image/svg+xml" href="/ui/russia-flag.svg">
<title>{escape(fragment.name or fragment.record_id)} — ПП № 1875</title>
<style>
body{{margin:0;background:#f4f6fc;color:#141c3b;font:16px/1.6 Inter,Arial,sans-serif}}
main{{max-width:1020px;margin:0 auto;padding:28px 20px 60px}}
.topline{{display:flex;justify-content:space-between;align-items:center;gap:12px;
flex-wrap:wrap;margin-bottom:22px}}
.back{{color:#454cff;text-decoration:none;font-weight:600}}
.decree-badge{{background:#eef0ff;color:#3a3fd6;border-radius:30px;padding:8px 18px;
font-size:13px;font-weight:600}}
h1{{font-size:27px;line-height:1.35;margin:0 0 10px;letter-spacing:-.3px}}
.meta{{display:flex;gap:8px;flex-wrap:wrap;margin:0 0 22px;padding:0;list-style:none}}
.meta li{{background:#fff;border:1px solid #dbe1ef;border-radius:10px;padding:7px 14px;
font-size:14px;color:#4c5f94;font-weight:600}}
.verify{{background:#fff;border:1px solid #dbe1ef;border-radius:16px;padding:20px 24px;
margin:0 0 26px}}
.verify h2{{font-size:16px;margin:0 0 6px}}
.verify p{{margin:0 0 12px;color:#657293;font-size:14px}}
.verify ul{{margin:0;padding:0;list-style:none;display:grid;gap:10px}}
.verify a{{color:#454cff;font-weight:600;text-decoration:none}}
.verify a:hover{{text-decoration:underline}}
.verify li span{{display:block;color:#7584a8;font-size:13px;margin-top:2px}}
.doc{{background:#fff;border:1px solid #dbe1ef;border-radius:16px;padding:26px 30px}}
.doc h3{{margin:0 0 18px;font-size:15px;color:#657293;font-weight:600;
text-transform:uppercase;letter-spacing:.06em}}
.page-break{{margin:26px 0 12px;color:#9aa3bd;font-size:12px;letter-spacing:.12em;
text-transform:uppercase;border-bottom:1px dashed #e1e6f1;padding-bottom:6px}}
.page-break:first-child{{margin-top:0}}
.table-heading{{display:grid;grid-template-columns:2.6em minmax(0,1fr) auto;gap:10px;
padding:12px;margin-bottom:8px;border-bottom:1px solid #dbe1ef;color:#657293;
font-size:13px;font-weight:600}}
.pos{{display:flex;align-items:baseline;gap:10px;padding:7px 12px;border-radius:10px;
border:1px solid transparent}}
.pos .num{{color:#9aa3bd;min-width:2.6em;font-variant-numeric:tabular-nums}}
.pos .name{{flex:1;min-width:0;overflow-wrap:anywhere}}
.pos .code{{font-family:ui-monospace,Consolas,monospace;font-size:13px;background:#eef0ff;
color:#3a3fd6;border-radius:7px;padding:2px 9px;white-space:nowrap}}
.pos.hit{{background:#fff3bf;border-color:#f0d264}}
.pos.hit .name{{font-weight:700}}
.cont{{color:#55618a;padding:0 12px;font-size:14px;line-height:1.5}}
@media(max-width:600px){{.doc{{padding:18px 12px}}.pos{{flex-wrap:wrap}}
.pos .code{{margin-left:3.6em}}h1{{font-size:22px}}}}
.hint{{color:#7584a8;font-size:13px;margin:16px 2px 0}}
@media print{{.topline,.verify{{display:none}} body{{background:#fff}}
.doc{{border:0;padding:0}}}}
</style></head>
<body><main>
<div class="topline">
<a class="back" href="javascript:history.back()">← назад к результатам</a>
<span class="decree-badge">ПП РФ № 1875 от 23.12.2024</span>
</div>
<h1>{escape(fragment.name or fragment.record_id)}</h1>
<ul class="meta">{meta_items}</ul>
<section class="verify">
<h2>Удостовериться в подлинности</h2>
<p>Ниже — рабочий файл постановления; первоисточники и действующая редакция:</p>
<ul>
{sources}
</ul>
</section>
<section class="doc">
<h3>Текст перечня (фрагмент страниц {fragment.pages.start}–{fragment.pages.end})</h3>
{decree}
</section>
<p class="hint">Позиция, к которой привела классификация, подсвечена жёлтым.</p>
</main></body></html>"""


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


def _reset_after_catalog_change() -> bool:
    """Drop the caches so the next request sees the newly installed catalog.

    Returns whether a reindex is required (ML/qdrant mode). Shared by upload and
    rollback: both replace `data.csv`/`data.json` and both must invalidate the
    same things, or the running process keeps answering from the old catalog.
    """
    reset_classifier()
    reset_source_lookup()
    try:
        from ml.embedder import reset_encoder_cache

        reset_encoder_cache()
    except Exception:  # pragma: no cover - ML stack is optional
        logger.exception("Не удалось сбросить кэш эмбеддера после смены каталога")
    return bool(settings.ml_enabled and settings.classifier_backend == "qdrant")


def _version_model(version: CatalogVersionRecord) -> CatalogVersion:
    return CatalogVersion(
        version_id=version.version_id,
        rows=version.rows,
        created_at=version.created_at,
        sha256=version.sha256,
        files=version.files,
    )


@app.get(
    "/admin/catalog/export",
    tags=["admin"],
    summary="Скачать установленный каталог (data.csv) для резервной копии",
    response_class=Response,
    responses={
        200: {"content": {"text/csv": {}}, "description": "Файл каталога (вложение)."},
        404: {"description": "Каталог ещё не установлен."},
    },
    dependencies=[Depends(require_admin_key)],
)
def download_catalog() -> Response:
    """Отдать текущий `data.csv` ровно в том виде, в котором он работает.

    Нужно перед загрузкой новой редакции: если новая окажется неверной, вернуть
    прежнюю можно через `POST /admin/catalog/rollback` (снимки делаются
    автоматически), а этот файл — независимая копия вне контейнера.
    """
    try:
        content = export_catalog_csv(resolve_data_dir())
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M")
    return Response(
        content=content,
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="catalog-{stamp}.csv"',
            "Cache-Control": "no-store",
        },
    )


@app.get(
    "/admin/catalog/versions",
    response_model=CatalogVersionsResponse,
    tags=["admin"],
    summary="Снимки каталога, сделанные перед загрузками",
    dependencies=[Depends(require_admin_key)],
)
def catalog_versions() -> CatalogVersionsResponse:
    data_dir = resolve_data_dir()
    return CatalogVersionsResponse(
        versions=[_version_model(version) for version in list_versions(data_dir)],
        keep=settings.catalog_snapshot_keep,
        current=catalog_facts(data_dir),
    )


@app.post(
    "/admin/catalog/rollback",
    response_model=CatalogRollbackResponse,
    tags=["admin"],
    summary="Откатить каталог к предыдущей загруженной редакции",
    dependencies=[Depends(require_admin_key)],
)
def rollback_catalog(
    version: Annotated[
        str | None,
        Query(description="Идентификатор снимка; по умолчанию — самый свежий."),
    ] = None,
) -> CatalogRollbackResponse:
    """Вернуть каталог, который работал до последней загрузки.

    Состояние, которое заменяется, тоже снимается — поэтому повторный откат
    возвращает предыдущее состояние (откат обратим). Кэши сбрасываются, так что
    ответы меняются сразу, без перезапуска контейнера.
    """
    data_dir = resolve_data_dir()
    try:
        catalog_info, restored = restore_version(
            data_dir, version, keep=settings.catalog_snapshot_keep
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    reindex_required = _reset_after_catalog_change()
    logger.warning(
        "Каталог откачен к снимку %s: строк=%d, reindex_required=%s",
        restored.version_id,
        catalog_info.rows,
        reindex_required,
    )
    return CatalogRollbackResponse(
        restored=_version_model(restored),
        catalog=catalog_info,
        reindex_required=reindex_required,
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
    summary="Загрузить каталог (CSV/JSON) или пакет постановления (ZIP)",
    dependencies=[Depends(require_admin_key)],
)
async def upload_catalog(
    background_tasks: BackgroundTasks,
    file: Annotated[
        UploadFile,
        File(description="data.csv, data.json или ZIP: data.csv + source_text.txt + report.json"),
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
        # The catalog being replaced is snapshotted inside write_catalog, so
        # `POST /admin/catalog/rollback` can undo this upload.
        catalog_info = write_catalog(
            rows,
            resolve_data_dir(),
            snapshot=True,
            source_files=parsed.source_files or None,
            keep=settings.catalog_snapshot_keep,
        )
        applied = True
        # Drop caches so the next request sees the new catalog immediately.
        reindex_required = _reset_after_catalog_change()
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
