"""Embedding model wrapper. Loaded once on first import."""

import time
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from ml.config import DEBUG, MODEL_NAME


@lru_cache(maxsize=1)
def get_model() -> SentenceTransformer:
    """Loads the model (cached on subsequent calls)."""
    if DEBUG:
        print(f"⏳ Loading model {MODEL_NAME}...")
        start = time.perf_counter()

    model = SentenceTransformer(MODEL_NAME)

    if DEBUG:
        elapsed = time.perf_counter() - start
        print(f"✅ Model loaded in {elapsed:.1f}s")

    return model


def encode_texts(texts: list[str], normalize: bool = True) -> list[list[float]]:
    """
    Converts a list of texts into a list of vectors.

    Args:
        texts: list of strings to encode.
        normalize: normalize vectors (required for cosine similarity).

    Returns:
        List of vectors, each is a list of 1024 floats.
    """
    model = get_model()
    vectors = model.encode(texts, normalize_embeddings=normalize)
    return vectors.tolist()


def encode_text(text: str, normalize: bool = True) -> list[float]:
    """Converts a single text into a vector."""
    return encode_texts([text], normalize=normalize)[0]


if __name__ == "__main__":
    # Quick smoke test
    test_texts = ["electric drill bosch", "kitchen faucet"]
    vectors = encode_texts(test_texts)
    print(f"Encoded {len(vectors)} texts")
    print(f"Vector dimension: {len(vectors[0])}")