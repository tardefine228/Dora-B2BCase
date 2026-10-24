"""Configuration for the optional ML and Qdrant modules."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")


def _project_path(value: str) -> Path:
    """Resolve relative filesystem settings against the repository/app root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT_DIR / path


# === Embedding backend ===
# auto | sentence-transformers | spacy | tfidf
# auto order: sentence-transformers (if a model loads) -> spacy vectors -> tfidf.
EMBEDDER_BACKEND: str = os.getenv("EMBEDDER_BACKEND", "auto").strip().lower()
TFIDF_DIM: int = int(os.getenv("TFIDF_DIM", "512"))
TFIDF_NGRAMS: tuple[int, ...] = tuple(
    int(size) for size in os.getenv("TFIDF_NGRAMS", "3,4").split(",") if size.strip()
)

# Static Russian word vectors (offline fallback when no transformer model is available).
SPACY_MODEL: str = os.getenv("SPACY_MODEL", "ru_core_news_md").strip()

# === Model ===
_model_value = os.getenv("MODEL_NAME", "intfloat/multilingual-e5-small").strip()
_model_path = Path(_model_value).expanduser()
if (
    _model_path.is_absolute()
    or _model_value.startswith((".", "~"))
    or (ROOT_DIR / _model_path).exists()
):
    MODEL_NAME: str = str(_project_path(_model_value))
else:
    # Keep hub IDs such as `intfloat/multilingual-e5-small` unchanged.
    MODEL_NAME = _model_value
MODEL_DEVICE: str = os.getenv("MODEL_DEVICE", "cpu")
MODEL_LOCAL_ONLY: bool = os.getenv("MODEL_LOCAL_ONLY", "false").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
MODEL_BATCH_SIZE: int = max(1, int(os.getenv("MODEL_BATCH_SIZE", "16")))
# Truncate inputs to this many tokens (0 = keep the model default). Product names are
# short (median 63 characters, the longest catalog row is 1164), while long-context
# embedders such as BAAI/bge-m3 default to 8192 tokens: the buffers for that are pure
# memory overhead and can alone exceed the container limit. 512 tokens fit every
# catalog name with room to spare.
MODEL_MAX_SEQ_LENGTH: int = max(0, int(os.getenv("MODEL_MAX_SEQ_LENGTH", "512")))

# === Qdrant ===
QDRANT_URL: str | None = os.getenv("QDRANT_URL") or None
QDRANT_API_KEY: str | None = os.getenv("QDRANT_API_KEY") or None
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "pp1875_items")
QDRANT_VECTOR_DIM: int = int(os.getenv("QDRANT_VECTOR_DIM", "1024"))
QDRANT_STORAGE_DIR: Path = _project_path(
    os.getenv("QDRANT_STORAGE_DIR", str(ROOT_DIR / "qdrant_storage"))
)
INDEX_BATCH_SIZE: int = max(1, int(os.getenv("INDEX_BATCH_SIZE", "64")))

# === Search ===
DEFAULT_TOP_K: int = max(1, int(os.getenv("DEFAULT_TOP_K", "5")))
RERANKER_THRESHOLD: float = float(os.getenv("RERANKER_THRESHOLD", "0.75"))

# === Mode ===
DEBUG: bool = os.getenv("DEBUG", "false").lower() in {"1", "true", "yes", "on"}
USE_ONNX: bool = os.getenv("USE_ONNX", "false").lower() in {"1", "true", "yes", "on"}
USE_INT8: bool = os.getenv("USE_INT8", "false").lower() in {"1", "true", "yes", "on"}

# === Paths ===
MODELS_DIR: Path = _project_path(os.getenv("MODELS_DIR", str(ROOT_DIR / "models")))
# Pre-built, version-controlled artifacts (reranker.json, tfidf_embedder.json). They ship
# inside the Docker image, so `docker compose up --build` needs no extra training step.
# MODELS_DIR (e.g. a mounted volume) still takes priority when it holds an artifact.
ARTIFACTS_DIR: Path = Path(__file__).resolve().parent / "artifacts"
DATA_DIR: Path = _project_path(os.getenv("DATA_DIR", str(ROOT_DIR / "data")))
# The checked-in JSON is stored directly in data/; allow RAW_DATA_DIR to override it.
RAW_DATA_DIR: Path = _project_path(os.getenv("RAW_DATA_DIR", str(DATA_DIR)))
PROCESSED_DATA_DIR: Path = _project_path(
    os.getenv("PROCESSED_DATA_DIR", str(DATA_DIR / "processed"))
)
