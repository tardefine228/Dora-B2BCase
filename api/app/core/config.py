"""Configuration loaded from environment variables (and an optional local .env)."""

from __future__ import annotations

import os

from dotenv import load_dotenv

# Load a local .env for `uvicorn` runs. Docker Compose injects the same values itself.
load_dotenv()


def _positive_int(name: str, default: int) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a positive integer") from exc
    if value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_float(name: str, default: float | None = None) -> float | None:
    """Float setting with an explicit "unset" state (empty string means unset)."""
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return float(value)


class Settings:
    """Small dependency-free settings object; values are read once at process start."""

    def __init__(self) -> None:
        self.app_name = os.getenv("APP_NAME", "Dora B2BCase API")
        self.log_level = os.getenv("LOG_LEVEL", "INFO").upper()

        # catalog | mock | qdrant
        self.classifier_backend = os.getenv("CLASSIFIER_BACKEND", "catalog").strip().lower()
        self.ml_enabled = _bool("ML_ENABLED", default=False)

        self.qdrant_url = os.getenv("QDRANT_URL", "http://qdrant:6333").strip()
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", "")
        self.qdrant_collection = os.getenv("QDRANT_COLLECTION", "pp1875_items")

        # Vector search (embeddings) joins the lexical matcher in the catalog backend.
        # Falls back to lexical-only (and says so in /health) if no embedder loads.
        self.vector_search = _bool("VECTOR_SEARCH", default=True)
        # memory (exact numpy search, default for local runs and tests) | qdrant
        # (vectors published to the Qdrant service; falls back to memory if it is down).
        self.vector_store = os.getenv("VECTOR_STORE", "memory").strip().lower()
        if self.vector_store not in {"memory", "qdrant"}:
            raise ValueError("VECTOR_STORE must be 'memory' or 'qdrant'")

        self.catalog_path = os.getenv("CATALOG_PATH", "").strip()
        self.data_dir = os.getenv("DATA_DIR", "data").strip() or "data"

        # Comma-separated origins for browser clients; empty value disables CORS
        # headers (same-origin deployments such as /ui do not need them).
        self.cors_allow_origins = [
            origin.strip()
            for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
            if origin.strip()
        ]

        self.model_name = os.getenv("MODEL_NAME", "BAAI/bge-m3").strip()
        self.model_device = os.getenv("MODEL_DEVICE", "cpu").strip() or "cpu"
        self.model_local_only = _bool("MODEL_LOCAL_ONLY", default=False)

        # Decision floors for the semantic backend. The artifact ships the point
        # measured on the lexical backend, but the semantic raw scale is compressed,
        # so its `review` floor is measured separately: 0.25 lifts the precision of
        # an honest `not_found` from 89.8% to 97.2% (eval/experiments.md, эксп. 16).
        # `ok` is aligned with the lexical floor (0.77) — it does not move any
        # measured metric, only how many answers are labelled confidently.
        # Empty string means "keep the artifact value".
        self.semantic_ok_floor = _optional_float("QDRANT_OK_FLOOR", 0.77)
        self.semantic_review_floor = _optional_float("QDRANT_REVIEW_FLOOR", 0.25)

        # Reindexing is disabled unless dependencies are installed; protect it with a key.
        self.admin_api_key = os.getenv("ADMIN_API_KEY", "")
        self.max_concurrent_inferences = _positive_int("MAX_CONCURRENT_INFERENCES", 1)
        self.classification_cache_size = _positive_int("CLASSIFICATION_CACHE_SIZE", 512)
        self.max_request_body_bytes = _positive_int("MAX_REQUEST_BODY_BYTES", 1_000_000)
        # Upload of a goods list file on the main page (POST /classify/file).
        self.max_upload_bytes = _positive_int("MAX_UPLOAD_BYTES", 10_000_000)

        # Per-client rate limiting (token bucket); probes, metrics and /ui are exempt.
        self.rate_limit_enabled = _bool("RATE_LIMIT_ENABLED", default=True)
        self.rate_limit_rps = _positive_int("RATE_LIMIT_RPS", 10)
        self.rate_limit_burst = _positive_int("RATE_LIMIT_BURST", 20)
        self.rate_limit_admin_rps = _positive_int("RATE_LIMIT_ADMIN_RPS", 1)
        self.rate_limit_admin_burst = _positive_int("RATE_LIMIT_ADMIN_BURST", 2)

        # Background jobs: sqlite survives an API restart, memory is for tests/demos.
        # memory | sqlite | postgres (postgres is the production store; DATABASE_URL).
        self.job_store = os.getenv("JOB_STORE", "sqlite").strip().lower()
        if self.job_store not in {"memory", "sqlite", "postgres"}:
            raise ValueError("JOB_STORE must be 'memory', 'sqlite' or 'postgres'")
        self.database_url = os.getenv("DATABASE_URL", "").strip()
        if self.job_store == "postgres" and not self.database_url:
            raise ValueError("JOB_STORE=postgres requires DATABASE_URL")
        # Request audit log in PostgreSQL (best effort, never blocks an answer).
        self.audit_log = _bool("AUDIT_LOG", default=self.job_store == "postgres")

        # Where heavy batches run: `inprocess` (FastAPI background task, tests and
        # local runs) or `celery` (a separate worker fed through Redis).
        self.queue_backend = os.getenv("QUEUE_BACKEND", "inprocess").strip().lower()
        if self.queue_backend not in {"inprocess", "celery"}:
            raise ValueError("QUEUE_BACKEND must be 'inprocess' or 'celery'")
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0").strip()
        # A job that has not been touched for this long while `running` is considered
        # lost (its worker died) and is failed instead of hanging forever.
        self.job_stale_seconds = _positive_int("JOB_STALE_SECONDS", 900)
        self.job_store_path = os.getenv("JOB_STORE_PATH", "data/jobs.sqlite").strip()
        self.job_ttl_hours = _positive_int("JOB_TTL_HOURS", 24)
        self.job_max_active = _positive_int("JOB_MAX_ACTIVE", 16)
        self.job_max_stored = _positive_int("JOB_MAX_STORED", 512)


settings = Settings()
