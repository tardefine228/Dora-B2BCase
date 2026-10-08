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

        # Reindexing is disabled unless dependencies are installed; protect it with a key.
        self.admin_api_key = os.getenv("ADMIN_API_KEY", "")
        self.max_concurrent_inferences = _positive_int("MAX_CONCURRENT_INFERENCES", 1)
        self.classification_cache_size = _positive_int("CLASSIFICATION_CACHE_SIZE", 512)
        self.max_request_body_bytes = _positive_int("MAX_REQUEST_BODY_BYTES", 1_000_000)


settings = Settings()
