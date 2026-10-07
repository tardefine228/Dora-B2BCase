"""Конфигурация проекта. Читает переменные из .env в корне."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Корень проекта = на уровень выше от ml/
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

# === Model === 
MODEL_NAME: str = os.getenv("MODEL_NAME", "BAAI/bge-m3")

# === Qdrant ===
QDRANT_STORAGE_DIR: Path = Path(os.getenv("QDRANT_STORAGE_DIR", "./qdrant_storage"))
QDRANT_COLLECTION: str = os.getenv("QDRANT_COLLECTION", "pp1875_items")
QDRANT_VECTOR_DIM: int = int(os.getenv("QDRANT_VECTOR_DIM", "1024"))

# === API ===
API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
API_PORT: int = int(os.getenv("API_PORT", "8000"))

# === Search ===
DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
RERANKER_THRESHOLD: float = float(os.getenv("RERANKER_THRESHOLD", "0.75"))

# === Mode ===
DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"
USE_ONNX: bool = os.getenv("USE_ONNX", "false").lower() == "true"
USE_INT8: bool = os.getenv("USE_INT8", "false").lower() == "true"

# === Paths ===
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
QDRANT_STORAGE_DIR: Path = ROOT_DIR / "qdrant_storage" 