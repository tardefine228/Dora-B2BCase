"""Embedding model wrapper. The model is loaded once and shared by all requests."""

import logging
import time
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from ml.config import (
    DEBUG,
    MODEL_BATCH_SIZE,
    MODEL_DEVICE,
    MODEL_LOCAL_ONLY,
    MODEL_NAME,
)

logger = logging.getLogger("dora.ml.embedder")


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Load the configured model once per process.

    Set ``MODEL_NAME`` to a local SentenceTransformers model directory and
    ``MODEL_LOCAL_ONLY=true`` to guarantee there are no download attempts.
    """
    started = time.perf_counter()
    logger.info("Loading embedding model %s on %s", MODEL_NAME, MODEL_DEVICE)
    model = SentenceTransformer(
        MODEL_NAME,
        device=MODEL_DEVICE,
        local_files_only=MODEL_LOCAL_ONLY,
    )
    if DEBUG:
        logger.info("Embedding model loaded in %.1f seconds", time.perf_counter() - started)
    return model


def encode_texts(texts: list[str], normalize: bool = True) -> list[list[float]]:
    """Encode texts in bounded mini-batches and return JSON/Qdrant-friendly vectors."""
    if not texts:
        return []
    vectors = get_model().encode(
        texts,
        batch_size=MODEL_BATCH_SIZE,
        normalize_embeddings=normalize,
        show_progress_bar=DEBUG,
        convert_to_numpy=True,
    )
    return vectors.tolist()


def encode_text(text: str, normalize: bool = True) -> list[float]:
    """Encode a single query."""
    return encode_texts([text], normalize=normalize)[0]


if __name__ == "__main__":
    sample = encode_texts(["electric drill bosch", "kitchen faucet"])
    print(f"Encoded {len(sample)} texts; vector dimension={len(sample[0])}")
