"""Configuration loaded from environment variables (and an optional local .env)."""

from __future__ import annotations

import math
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


def _confidence_floor(name: str) -> float | None:
    """Read an optional decision threshold in the inclusive probability range."""
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return None
    try:
        value = float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a finite number from 0 to 1") from exc
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a finite number from 0 to 1")
    return value


def _trusted_proxies(name: str) -> tuple[str, ...]:
    """Parse ``TRUSTED_PROXIES`` into ``("10.0.0.0/8", "127.0.0.1")``.

    Entries are IP addresses or CIDR networks. A typo must fail at startup: a
    silently ignored entry would either lock every client into one bucket or
    reopen the spoofing hole. Wildcards are deliberately rejected; trusting
    every peer would recreate the header-spoofing vulnerability.
    """
    import ipaddress

    raw = os.getenv(name, "")
    entries: list[str] = []
    for chunk in raw.split(","):
        entry = chunk.strip()
        if not entry:
            continue
        try:
            ipaddress.ip_network(entry, strict=False)
        except ValueError as exc:
            raise ValueError(
                f"{name}: '{entry}' is not an IP address or CIDR network"
            ) from exc
        entries.append(entry)
    return tuple(entries)


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

        # Catalog decision floors override the values in the reranker artifact.
        # Empty/unset keeps the artifact's calibrated operating point; the classifier
        # validates the ok > review relationship against the loaded artifact.
        self.catalog_ok_floor = _confidence_floor("CATALOG_OK_FLOOR")
        self.catalog_review_floor = _confidence_floor("CATALOG_REVIEW_FLOOR")

        # Comma-separated origins for browser clients; empty value disables CORS
        # headers (same-origin deployments such as /ui do not need them).
        self.cors_allow_origins = [
            origin.strip()
            for origin in os.getenv("CORS_ALLOW_ORIGINS", "*").split(",")
            if origin.strip()
        ]

        self.model_name = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small").strip()
        self.model_device = os.getenv("MODEL_DEVICE", "cpu").strip() or "cpu"
        self.model_local_only = _bool("MODEL_LOCAL_ONLY", default=False)

        # Decision floors for the semantic backend. The semantic raw scale is
        # compressed, so its `review` floor is measured separately from the catalog
        # one. 0.45 is the measured optimum of the shipped vector artifact on the
        # 274-row set: 152 correct decisions against 146 at the old 0.25, false
        # codes on out-of-list goods 14 -> 4 (eval/experiments.md, эксп. 28), and
        # 0.45 is also the median threshold the cross-validation picks (0.452), so
        # it is a plateau, not a cell read off one table. The value was measured
        # with the `label_kind` doctrine in place (docs/labeling-doctrine.md).
        # `ok` was aligned with the lexical floor of the time (0.77). The catalog
        # floor moved to 0.80 in эксп. 34 (rebuild on the wave-5 pairs put six wrong
        # confident answers through at 0.77 against four at 0.80, ml/train_reranker.py),
        # but the semantic scale is a different one, and its 0.77 is the value the
        # semantic measurements were taken at — raising it here would change a
        # published number (eval/backend_comparison.md) without a measurement.
        # Empty string means "keep the artifact value".
        self.semantic_ok_floor = _optional_float("QDRANT_OK_FLOOR", 0.77)
        self.semantic_review_floor = _optional_float("QDRANT_REVIEW_FLOOR", 0.45)

        # Which artifact scores the semantic backend: `auto` prefers the reranker
        # trained for the active embedder (reranker.<tag>.json) and falls back to the
        # lexical one, `lexical` always uses reranker.json. An escape hatch: switching
        # back must not require editing files inside the image.
        self.semantic_reranker = os.getenv("SEMANTIC_RERANKER", "auto").strip().lower()
        if self.semantic_reranker not in {"auto", "lexical"}:
            raise ValueError("SEMANTIC_RERANKER must be 'auto' or 'lexical'")

        # Reindexing is disabled unless dependencies are installed; protect it with a key.
        self.admin_api_key = os.getenv("ADMIN_API_KEY", "")
        self.max_concurrent_inferences = _positive_int("MAX_CONCURRENT_INFERENCES", 1)
        self.classification_cache_size = _positive_int("CLASSIFICATION_CACHE_SIZE", 512)

        # Third stage (off by default): a constrained LLM check of the top-3
        # hypotheses for `review`/`not_found` answers only. Any OpenAI-compatible
        # endpoint works; without a model the stage is simply absent
        # (see api/app/services/llm_verifier.py, eval/experiments.md эксп. 30).
        self.llm_verifier_enabled = _bool("LLM_VERIFIER_ENABLED", default=False)
        self.llm_verifier_base_url = os.getenv(
            "LLM_VERIFIER_BASE_URL", "https://api.openai.com/v1"
        ).strip()
        self.llm_verifier_api_key = os.getenv("LLM_VERIFIER_API_KEY", "").strip()
        self.llm_verifier_model = os.getenv("LLM_VERIFIER_MODEL", "").strip()
        self.llm_verifier_timeout_seconds = _optional_float("LLM_VERIFIER_TIMEOUT", 8.0) or 8.0
        # A ceiling, not a target: one batch of 100 purchase lines must not turn
        # into an unbounded bill. Calls beyond the budget are skipped and the
        # answer stays the model's own.
        self.llm_verifier_calls_per_minute = _positive_int("LLM_VERIFIER_CALLS_PER_MINUTE", 60)
        self.max_request_body_bytes = _positive_int("MAX_REQUEST_BODY_BYTES", 1_000_000)
        # Upload of a goods list file on the main page (POST /classify/file).
        self.max_upload_bytes = _positive_int("MAX_UPLOAD_BYTES", 10_000_000)

        # Per-client rate limiting (token bucket); probes, metrics and /ui are exempt.
        # ``X-Forwarded-For`` is honoured only when the request comes from one of
        # these addresses/networks (a reverse proxy). Empty = trust nobody, so a
        # direct client cannot pick its own bucket by forging the header.
        self.trusted_proxies = _trusted_proxies("TRUSTED_PROXIES")
        self.rate_limit_enabled = _bool("RATE_LIMIT_ENABLED", default=True)
        self.rate_limit_rps = _positive_int("RATE_LIMIT_RPS", 10)
        self.rate_limit_burst = _positive_int("RATE_LIMIT_BURST", 20)
        self.rate_limit_admin_rps = _positive_int("RATE_LIMIT_ADMIN_RPS", 1)
        self.rate_limit_admin_burst = _positive_int("RATE_LIMIT_ADMIN_BURST", 2)
        # ``POST /classify/file`` is far heavier than ``/classify``: one upload
        # becomes a batch of up to MAX_FILE_ITEMS classifications, so counting it
        # against the request bucket let a client queue hundreds of positions per
        # second. It gets its own bucket, counted in files.
        self.rate_limit_file_rps = _positive_int("RATE_LIMIT_FILE_RPS", 1)
        self.rate_limit_file_burst = _positive_int("RATE_LIMIT_FILE_BURST", 3)

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
