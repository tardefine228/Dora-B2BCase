"""Request statistics: Prometheus exposition (default) + the legacy JSON snapshot.

``GET /metrics`` serves the Prometheus text format (``text/plain; version=0.0.4``)
so a scraper picks the service up without configuration; clients that send
``Accept: application/json`` still get the original in-process JSON snapshot
(BE-6). Both views are fed by the same ``Metrics`` recorder used by the observer
and rate-limit middlewares.

Health gauges for the embedder and the vector store are refreshed at scrape time
from ``app.services.classifier.vector_layer_health`` (TTL-cached there, so a
scrape never turns into a per-scrape network probe).

Note: counters live in the process that serves requests. With
``QUEUE_BACKEND=celery`` the batch classifications happen in the worker process
and are counted there; the worker exposes no HTTP endpoint, so its counters are
visible in its own logs, not in the API's ``/metrics``.
"""

import threading
from collections import defaultdict

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    REGISTRY,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from app.core.config import settings

# Fast local API: sub-10 ms answers are the norm, seconds mean trouble.
LATENCY_BUCKETS_SECONDS = (
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
    5.0,
    10.0,
    30.0,
)

HTTP_REQUESTS_TOTAL = Counter(
    "dora_http_requests_total",
    "HTTP responses by route template, method and status code.",
    ["path", "method", "status"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "dora_http_request_duration_seconds",
    "HTTP request latency in seconds, by route template.",
    ["path"],
    buckets=LATENCY_BUCKETS_SECONDS,
)
HTTP_RATE_LIMITED_TOTAL = Counter(
    "dora_http_rate_limited_total",
    "Requests rejected by the rate limiter (HTTP 429), by path.",
    ["path"],
)
CLASSIFICATIONS_TOTAL = Counter(
    "dora_classifications_total",
    "Classification answers by call mode (single|batch), backend and status.",
    ["mode", "backend", "status"],
)
BATCH_JOBS_TOTAL = Counter(
    "dora_batch_jobs_total",
    "Batch jobs by lifecycle event: accepted, completed, failed.",
    ["event"],
)

CLASSIFIER_UP = Gauge(
    "dora_classifier_up",
    "1 when the classifier singleton loads and can answer requests.",
)
EMBEDDER_UP = Gauge(
    "dora_embedder_up",
    "1 when the embedder required by the configuration is serving "
    "(0 on silent degradation, e.g. VECTOR_SEARCH=true but no embedder loaded).",
)
EMBEDDER_INFO = Gauge(
    "dora_embedder_info",
    "Active embedder tag (informational; value is always 1).",
    ["tag"],
)
EMBEDDER_DIM = Gauge(
    "dora_embedder_dim",
    "Dimension of the active embedder (0 when the vector layer is off).",
)
VECTOR_SEARCH_ENABLED = Gauge(
    "dora_vector_search_enabled",
    "1 when the vector layer actually participates in answers.",
)
VECTOR_STORE_UP = Gauge(
    "dora_vector_store_up",
    "1 when the configured vector store answers probes "
    "(the in-memory store is up by definition; Qdrant is pinged).",
)
VECTOR_STORE_INFO = Gauge(
    "dora_vector_store_info",
    "Vector store actually serving searches (informational; value is always 1).",
    ["store"],
)
VECTOR_POSITIONS = Gauge(
    "dora_vector_positions",
    "Catalog positions held by the active vector index.",
)
BUILD_INFO = Gauge(
    "dora_build_info",
    "Service build and configuration (informational; value is always 1).",
    ["version", "classifier_backend", "job_store", "queue_backend"],
)


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = 0
        self.by_status: dict[int, int] = defaultdict(int)
        self.latency_sum = 0.0
        self.by_path: dict[str, int] = defaultdict(int)
        self.rate_limited = 0
        self.rate_limited_by_path: dict[str, int] = defaultdict(int)
        self.classifications: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.batch_jobs: dict[str, int] = defaultdict(int)

    def record(self, path: str, status: int, latency_ms: float, method: str = "") -> None:
        with self._lock:
            self.total += 1
            self.by_status[status] += 1
            self.latency_sum += latency_ms
            self.by_path[path] += 1
        HTTP_REQUESTS_TOTAL.labels(path=path, method=method or "GET", status=status).inc()
        HTTP_REQUEST_DURATION_SECONDS.labels(path=path).observe(latency_ms / 1000.0)

    def record_rate_limited(self, path: str) -> None:
        """Count a request rejected by the rate limiter (HTTP 429)."""
        with self._lock:
            self.rate_limited += 1
            self.rate_limited_by_path[path] += 1
        HTTP_RATE_LIMITED_TOTAL.labels(path=path).inc()

    def record_classification(self, mode: str, status: str) -> None:
        """Count one classification answer (``mode``: single | batch)."""
        with self._lock:
            self.classifications[mode][status] += 1
        CLASSIFICATIONS_TOTAL.labels(
            mode=mode, backend=settings.classifier_backend, status=status
        ).inc()

    def record_batch_job(self, event: str) -> None:
        """Count a batch job lifecycle event (accepted | completed | failed)."""
        with self._lock:
            self.batch_jobs[event] += 1
        BATCH_JOBS_TOTAL.labels(event=event).inc()

    def snapshot(self) -> dict:
        with self._lock:
            avg = (self.latency_sum / self.total) if self.total else 0.0
            return {
                "total_requests": self.total,
                "requests_by_status": dict(self.by_status),
                "average_latency_ms": round(avg, 3),
                "requests_by_path": dict(self.by_path),
                "rate_limited_total": self.rate_limited,
                "rate_limited_by_path": dict(self.rate_limited_by_path),
                "classifications": {
                    mode: dict(statuses) for mode, statuses in self.classifications.items()
                },
                "batch_jobs": dict(self.batch_jobs),
            }


metrics = Metrics()


def set_build_info(version: str) -> None:
    """Pin the running build/configuration as an info gauge (called at startup)."""
    BUILD_INFO.clear()
    BUILD_INFO.labels(
        version=version,
        classifier_backend=settings.classifier_backend,
        job_store=settings.job_store,
        queue_backend=settings.queue_backend,
    ).set(1)


def refresh_health_gauges(health: dict) -> None:
    """Map a ``vector_layer_health()`` snapshot onto the Prometheus gauges."""
    CLASSIFIER_UP.set(1 if health.get("classifier_up") else 0)
    EMBEDDER_UP.set(1 if health.get("embedder_up") else 0)
    VECTOR_SEARCH_ENABLED.set(1 if health.get("vector_search_enabled") else 0)
    VECTOR_STORE_UP.set(1 if health.get("store_up") else 0)
    EMBEDDER_DIM.set(int(health.get("dim") or 0))
    VECTOR_POSITIONS.set(int(health.get("positions") or 0))
    # Info gauges: drop stale label sets so only the *active* embedder/store show.
    EMBEDDER_INFO.clear()
    EMBEDDER_INFO.labels(tag=str(health.get("embedder") or "none")).set(1)
    VECTOR_STORE_INFO.clear()
    VECTOR_STORE_INFO.labels(store=str(health.get("store") or "none")).set(1)


def prometheus_latest() -> tuple[bytes, str]:
    """Render the registry in the Prometheus text exposition format."""
    return generate_latest(REGISTRY), CONTENT_TYPE_LATEST
