"""Embedding backends for the PP-1875 semantic index.

Two interchangeable backends implement the same ``encode()`` contract:

* ``sentence-transformers`` — a real transformer model (for example ``BAAI/bge-m3``)
  from a local SentenceTransformers directory. Heavy, optional, best quality.
* ``tfidf`` — a self-contained hashing TF-IDF embedder fitted on the catalog
  itself. No downloads, no torch; it keeps the ML/Qdrant pipeline testable
  offline and works as a calibrated fallback for demos.

``EMBEDDER_BACKEND`` selects the backend (``auto`` prefers the transformer when
its dependencies and weights are available, otherwise falls back to ``tfidf``).
Vectors are L2-normalized, so Qdrant cosine similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from functools import lru_cache
from math import log
from pathlib import Path
from typing import Protocol

from ml.config import (
    DEBUG,
    EMBEDDER_BACKEND,
    MODEL_BATCH_SIZE,
    MODEL_DEVICE,
    MODEL_LOCAL_ONLY,
    MODEL_NAME,
    MODELS_DIR,
    TFIDF_DIM,
    TFIDF_NGRAMS,
)

logger = logging.getLogger("dora.ml.embedder")

TFIDF_ARTIFACT_NAME = "tfidf_embedder.json"
ARTIFACT_VERSION = 1


class Encoder(Protocol):
    """Minimal embedding interface shared by all backends."""

    dim: int

    def encode(self, texts: list[str], normalize: bool = True) -> list[list[float]]: ...


def _bucket_and_sign(token: str, dim: int) -> tuple[int, float]:
    """Deterministic signed hashing: md5 keeps results stable across processes."""
    digest = hashlib.md5(token.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], "big") % dim
    sign = 1.0 if digest[8] & 1 else -1.0
    return bucket, sign


class TfidfEncoder:
    """Hashing TF-IDF encoder over word stems and character n-grams.

    Document frequencies live in ``dim`` hash buckets, which keeps the artifact
    tiny (a few kilobytes) while still downweighting ubiquitous fragments.
    """

    def __init__(
        self,
        dim: int = TFIDF_DIM,
        ngram_sizes: tuple[int, ...] = TFIDF_NGRAMS,
        doc_count: int = 0,
        doc_frequencies: list[int] | None = None,
    ) -> None:
        self.dim = max(16, int(dim))
        self.ngram_sizes = tuple(ngram_sizes)
        self.doc_count = max(0, int(doc_count))
        self.doc_frequencies = list(doc_frequencies or [0] * self.dim)
        if len(self.doc_frequencies) != self.dim:
            raise ValueError("doc_frequencies length must match dim")

    # --- fitting -----------------------------------------------------------
    def fit(self, texts: list[str]) -> None:
        self.doc_frequencies = [0] * self.dim
        self.doc_count = 0
        for text in texts:
            self.doc_count += 1
            for bucket in self._document_buckets(text):
                self.doc_frequencies[bucket] += 1

    def _idf(self, bucket: int) -> float:
        # Collisions only make idf estimates slightly flatter, never wrong-signed.
        return log((self.doc_count + 1) / (self.doc_frequencies[bucket] + 1)) + 1.0

    def _document_buckets(self, text: str) -> set[int]:
        buckets: set[int] = set()
        for token in self._features(text):
            bucket, _sign = _bucket_and_sign(token, self.dim)
            buckets.add(bucket)
        return buckets

    def _features(self, text: str) -> list[str]:
        from ml.features import char_ngrams, stems

        features = [f"w:{stem}" for stem in stems(text)]
        for size in self.ngram_sizes:
            features.extend(f"c:{gram}" for gram in char_ngrams(text, (size,)))
        return features

    # --- encoding ----------------------------------------------------------
    def encode(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * self.dim
            counts: dict[int, float] = {}
            signs: dict[int, float] = {}
            for token in self._features(text):
                bucket, sign = _bucket_and_sign(token, self.dim)
                counts[bucket] = counts.get(bucket, 0.0) + 1.0
                signs[bucket] = sign
            for bucket, count in counts.items():
                weight = (1.0 + log(count)) * self._idf(bucket)
                vector[bucket] += signs[bucket] * weight
            if normalize:
                norm = sum(value * value for value in vector) ** 0.5
                if norm > 0.0:
                    vector = [value / norm for value in vector]
            vectors.append(vector)
        return vectors

    # --- persistence -------------------------------------------------------
    def save(self, path: Path) -> None:
        payload = {
            "version": ARTIFACT_VERSION,
            "backend": "tfidf",
            "dim": self.dim,
            "ngram_sizes": list(self.ngram_sizes),
            "doc_count": self.doc_count,
            "doc_frequencies": self.doc_frequencies,
            "trained_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> TfidfEncoder:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("backend") != "tfidf":
            raise ValueError(f"Неизвестный формат артефакта эмбеддера: {path}")
        return cls(
            dim=int(payload["dim"]),
            ngram_sizes=tuple(payload["ngram_sizes"]),
            doc_count=int(payload["doc_count"]),
            doc_frequencies=[int(value) for value in payload["doc_frequencies"]],
        )


class SentenceTransformerEncoder:
    """Adapter around a SentenceTransformers model; imports stay lazy."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model %s on %s", MODEL_NAME, MODEL_DEVICE)
        started = time.perf_counter()
        self._model = SentenceTransformer(
            MODEL_NAME,
            device=MODEL_DEVICE,
            local_files_only=MODEL_LOCAL_ONLY,
        )
        self.dim = int(self._model.get_sentence_embedding_dimension())
        if DEBUG:
            logger.info("Embedding model loaded in %.1f seconds", time.perf_counter() - started)

    def encode(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            texts,
            batch_size=MODEL_BATCH_SIZE,
            normalize_embeddings=normalize,
            show_progress_bar=DEBUG,
            convert_to_numpy=True,
        )
        return vectors.tolist()


def sentence_transformers_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def tfidf_artifact_path() -> Path:
    return MODELS_DIR / TFIDF_ARTIFACT_NAME


def _build_tfidf_encoder() -> TfidfEncoder:
    artifact = tfidf_artifact_path()
    if artifact.is_file():
        logger.info("TF-IDF эмбеддер загружен из %s", artifact)
        return TfidfEncoder.load(artifact)

    # No artifact yet: fit on the checked-in catalog so quality is usable at once.
    from ml.catalog_loader import item_to_text, load_catalog

    catalog = load_catalog()
    encoder = TfidfEncoder()
    encoder.fit([item_to_text(item) for item in catalog])
    logger.warning(
        "TF-IDF эмбеддер обучен на каталоге (%d документов); для воспроизводимости "
        "сохраните артефакт: python scripts/build_model.py",
        encoder.doc_count,
    )
    return encoder


def resolve_backend() -> str:
    backend = EMBEDDER_BACKEND.strip().lower()
    if backend == "auto":
        return "sentence-transformers" if sentence_transformers_available() else "tfidf"
    return backend


@lru_cache(maxsize=1)
def get_encoder() -> Encoder:
    backend = resolve_backend()
    if backend == "sentence-transformers":
        return SentenceTransformerEncoder()
    if backend == "tfidf":
        return _build_tfidf_encoder()
    raise ValueError(f"Неизвестный EMBEDDER_BACKEND: {EMBEDDER_BACKEND}")


def encode_texts(texts: list[str], normalize: bool = True) -> list[list[float]]:
    """Encode texts in bounded mini-batches; JSON/Qdrant-friendly vectors."""
    if not texts:
        return []
    return get_encoder().encode(texts, normalize=normalize)


def encode_text(text: str, normalize: bool = True) -> list[float]:
    """Encode a single query."""
    return encode_texts([text], normalize=normalize)[0]


def reset_encoder_cache() -> None:
    """Drop the cached encoder (used after catalog updates and by tests)."""
    get_encoder.cache_clear()


if __name__ == "__main__":
    sample = encode_texts(["Ноутбук Lenovo IdeaPad", "Светильник настольный LED"])
    print(f"Encoded {len(sample)} texts; backend={resolve_backend()}; dim={len(sample[0])}")
