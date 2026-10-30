"""Embedding backends for the PP-1875 semantic index.

Three interchangeable backends implement the same ``encode()`` contract:

* ``sentence-transformers`` — a real transformer model (for example
  ``intfloat/multilingual-e5-small``) from a local SentenceTransformers directory.
  Heavy, optional, best quality.
* ``spacy`` — static Russian word vectors (``ru_core_news_md``), IDF-weighted
  mean per phrase. Offline, no torch; semantic but coarse.
* ``tfidf`` — a self-contained hashing TF-IDF embedder fitted on the catalog
  itself. No downloads, no torch; it keeps the ML/Qdrant pipeline testable
  offline and works as a calibrated fallback for demos.

``EMBEDDER_BACKEND`` selects the backend (``auto`` prefers the transformer when
it loads, then spaCy vectors, then ``tfidf``). Every backend has a stable
``tag`` (see :func:`encoder_tag`) that keys cached vectors and trained rerankers:
cosine scales differ between embedders, so artifacts are never shared.
Vectors are L2-normalized, so Qdrant cosine similarity is a plain dot product.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from functools import lru_cache
from math import log
from pathlib import Path
from typing import Protocol

from ml.config import (
    ARTIFACTS_DIR,
    DEBUG,
    EMBEDDER_BACKEND,
    MODEL_BATCH_SIZE,
    MODEL_DEVICE,
    MODEL_LOCAL_ONLY,
    MODEL_MAX_SEQ_LENGTH,
    MODEL_NAME,
    MODELS_DIR,
    SPACY_MODEL,
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

    backend = "tfidf"
    tag = "tfidf"

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


def _e5_prefixes(model_name: str) -> tuple[str, str]:
    """E5 models are trained with ``query: `` / ``passage: `` prefixes; others use none."""
    if "e5" in model_name.lower():
        return "query: ", "passage: "
    return "", ""


def _model_identity(model_name: str) -> str:
    """Recover the hub ID saved by download_model.py, independently of its path."""
    metadata = Path(model_name) / "dora_model.json"
    if metadata.is_file():
        data = json.loads(metadata.read_text(encoding="utf-8"))
        model_id = data.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError(f"Missing model_id in {metadata}")
        return model_id.strip()
    return model_name


def _transformer_spec(model_name: str) -> tuple[str, str, str]:
    identity = _model_identity(model_name)
    query_prefix, doc_prefix = _e5_prefixes(identity)
    # New namespace: never reuse vectors/rerankers made with the old prefix bug.
    signature = json.dumps([identity, query_prefix, doc_prefix, "prefix-v2"])
    fingerprint = hashlib.sha256(signature.encode("utf-8")).hexdigest()[:12]
    label = Path(identity.rstrip("/\\")).name.lower()
    return f"st-{label}-v2-{fingerprint}", query_prefix, doc_prefix


class SentenceTransformerEncoder:
    """Adapter around a SentenceTransformers model; imports stay lazy."""

    backend = "sentence-transformers"

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.tag, self._query_prefix, self._doc_prefix = _transformer_spec(MODEL_NAME)
        logger.info("Loading embedding model %s on %s", MODEL_NAME, MODEL_DEVICE)
        started = time.perf_counter()
        self._model = SentenceTransformer(
            MODEL_NAME,
            device=MODEL_DEVICE,
            local_files_only=MODEL_LOCAL_ONLY,
        )
        # sentence-transformers 5 renamed this to `get_embedding_dimension`;
        # the old name still works but logs a FutureWarning on every start.
        dimension = getattr(self._model, "get_embedding_dimension", None)
        if callable(dimension):
            self.dim = int(dimension())
        else:  # pragma: no cover - sentence-transformers < 5
            self.dim = int(self._model.get_sentence_embedding_dimension())
        if MODEL_MAX_SEQ_LENGTH:
            # A long-context model (bge-m3: 8192 tokens) allocates buffers for the
            # configured length; catalog names never need more than a few hundred.
            self._model.max_seq_length = MODEL_MAX_SEQ_LENGTH
        if DEBUG:
            logger.info(
                "Embedding model loaded in %.1f seconds (dim=%d, max_seq_length=%d)",
                time.perf_counter() - started,
                self.dim,
                getattr(self._model, "max_seq_length", 0),
            )

    def encode(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        return self._encode(texts, self._doc_prefix, normalize)

    def encode_queries(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        return self._encode(texts, self._query_prefix, normalize)

    def encode_documents(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        return self._encode(texts, self._doc_prefix, normalize)

    def _encode(self, texts: list[str], prefix: str, normalize: bool) -> list[list[float]]:
        if not texts:
            return []
        vectors = self._model.encode(
            [prefix + text for text in texts],
            batch_size=MODEL_BATCH_SIZE,
            normalize_embeddings=normalize,
            show_progress_bar=DEBUG,
            convert_to_numpy=True,
        )
        return vectors.tolist()


_WORD_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)


class SpacyVectorEncoder:
    """IDF-weighted mean of static Russian word vectors (spaCy ``ru_core_news_md``).

    Only the vector table of the spaCy package is used (no pipeline), so loading
    takes a couple of seconds and ~400 MB of RAM. Words missing from the table
    fall back to their pymorphy3 lemma. IDF weights come from the catalog itself,
    so decisive words (\"чайник\") outweigh ubiquitous ones (\"прочие\").
    """

    backend = "spacy"

    def __init__(self, model_name: str = SPACY_MODEL, documents: list[str] | None = None) -> None:
        import spacy

        self._nlp = spacy.load(
            model_name,
            exclude=[
                "tagger",
                "parser",
                "senter",
                "attribute_ruler",
                "lemmatizer",
                "ner",
                "morphologizer",
            ],
        )
        if not self._nlp.vocab.vectors.shape[0]:
            raise RuntimeError(f"В модели spaCy {model_name} нет векторов слов")
        try:
            import pymorphy3

            self._morph = pymorphy3.MorphAnalyzer()
        except Exception:  # lemma fallback is optional
            self._morph = None
        self.tag = f"spacy-{model_name}"
        self.dim = int(self._nlp.vocab.vectors.shape[1])
        self._idf: dict[str, float] = {}
        self._default_idf = 1.0
        self._word_cache: dict[str, object] = {}
        if documents:
            self.fit(documents)

    def fit(self, documents: list[str]) -> None:
        frequencies: dict[str, int] = {}
        for text in documents:
            for word in {w for w in self._words(text)}:
                frequencies[word] = frequencies.get(word, 0) + 1
        count = len(documents)
        self._idf = {w: log((count + 1) / (df + 1)) + 1.0 for w, df in frequencies.items()}
        self._default_idf = log(count + 1) + 1.0  # unseen words are as rare as it gets

    @staticmethod
    def _words(text: str) -> list[str]:
        return [word for word in _WORD_RE.findall(text.lower()) if len(word) > 2]

    def _word_vector(self, word: str):
        import numpy as np

        if word in self._word_cache:
            return self._word_cache[word]
        vector = None
        candidates = [word]
        if self._morph is not None:
            candidates.append(self._morph.parse(word)[0].normal_form)
        for candidate in candidates:
            lexeme = self._nlp.vocab[candidate]
            if lexeme.has_vector:
                raw = lexeme.vector
                norm = float(np.linalg.norm(raw))
                if norm > 0.0:
                    vector = raw / norm
                    break
        self._word_cache[word] = vector
        return vector

    def encode(self, texts: list[str], normalize: bool = True) -> list[list[float]]:
        import numpy as np

        output: list[list[float]] = []
        for text in texts:
            vectors, weights = [], []
            for word in self._words(text):
                vector = self._word_vector(word)
                if vector is None:
                    continue
                vectors.append(vector)
                weights.append(self._idf.get(word, self._default_idf))
            if vectors:
                mean = np.average(np.stack(vectors), axis=0, weights=weights)
                if normalize:
                    mean = mean / (float(np.linalg.norm(mean)) + 1e-9)
            else:
                mean = np.zeros(self.dim, dtype=np.float32)
            output.append(mean.astype(float).tolist())
        return output


def sentence_transformers_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("sentence_transformers") is not None


def tfidf_artifact_path() -> Path:
    """MODELS_DIR first, then the artifact shipped in ml/artifacts."""
    custom = MODELS_DIR / TFIDF_ARTIFACT_NAME
    if custom.is_file():
        return custom
    return ARTIFACTS_DIR / TFIDF_ARTIFACT_NAME


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


def spacy_available() -> bool:
    import importlib.util

    return (
        importlib.util.find_spec("spacy") is not None
        and importlib.util.find_spec(SPACY_MODEL) is not None
    )


_fallback_warned = False


def _warn_about_silent_fallback(fallback: str) -> None:
    """Say it out loud when an installed transformer is skipped.

    ``auto`` degrading to spaCy/TF-IDF is the most expensive silent failure in this
    project: the service looks healthy, only ``/health/ready`` reveals that the
    embedder is not the one baked into the image. Warning once per process keeps
    that visible without spamming the log.
    """
    global _fallback_warned
    if _fallback_warned:
        return
    _fallback_warned = True
    logger.warning(
        "EMBEDDER_BACKEND=auto: sentence-transformers установлен, но MODEL_NAME=%r — "
        "не каталог с весами модели; используется %s. Скачайте модель "
        "(python scripts/download_model.py <repo_id> <dir>) и укажите этот каталог в "
        "MODEL_NAME; чтобы откат не был тихим, задайте EMBEDDER_BACKEND="
        "sentence-transformers.",
        MODEL_NAME,
        fallback,
    )


def resolve_backend() -> str:
    backend = EMBEDDER_BACKEND.strip().lower()
    if backend != "auto":
        return backend
    # `auto` never downloads a model at run time: the transformer is used only when
    # its weights are already on disk (baked into the image or mounted). Ask for a
    # hub download explicitly with EMBEDDER_BACKEND=sentence-transformers.
    if sentence_transformers_available() and Path(MODEL_NAME).is_dir():
        return "sentence-transformers"
    fallback = "spacy" if spacy_available() else "tfidf"
    if sentence_transformers_available():
        _warn_about_silent_fallback(fallback)
    return fallback


def _build_spacy_encoder() -> SpacyVectorEncoder:
    from ml.catalog_loader import load_catalog

    return SpacyVectorEncoder(documents=[item["name"] for item in load_catalog()])


@lru_cache(maxsize=1)
def get_encoder() -> Encoder:
    backend = resolve_backend()
    explicit = EMBEDDER_BACKEND.strip().lower() != "auto"
    chain = {
        "sentence-transformers": ["sentence-transformers", "spacy", "tfidf"],
        "spacy": ["spacy", "tfidf"],
        "tfidf": ["tfidf"],
    }.get(backend)
    if chain is None:
        raise ValueError(f"Неизвестный EMBEDDER_BACKEND: {EMBEDDER_BACKEND}")
    if explicit:
        chain = chain[:1]  # an explicit choice must not silently degrade
    last_error: Exception | None = None
    for name in chain:
        try:
            if name == "sentence-transformers":
                return SentenceTransformerEncoder()
            if name == "spacy":
                return _build_spacy_encoder()
            return _build_tfidf_encoder()
        except Exception as exc:  # noqa: BLE001 - degrade to the next backend
            last_error = exc
            logger.warning("Эмбеддер %s недоступен (%s) — пробуем следующий", name, exc)
    raise RuntimeError(f"Ни один эмбеддер не загрузился: {last_error}")


def encoder_tag() -> str:
    """Stable identifier of the active embedder (keys caches and rerankers)."""
    return getattr(get_encoder(), "tag", "unknown")


def encode_queries(texts: list[str], normalize: bool = True) -> list[list[float]]:
    """Encode search queries (applies the query prefix of asymmetric models like e5)."""
    encoder = get_encoder()
    method = getattr(encoder, "encode_queries", None)
    return method(texts, normalize=normalize) if method else encoder.encode(texts, normalize)


def encode_documents(texts: list[str], normalize: bool = True) -> list[list[float]]:
    """Encode catalog entries (applies the passage prefix of asymmetric models)."""
    encoder = get_encoder()
    method = getattr(encoder, "encode_documents", None)
    return method(texts, normalize=normalize) if method else encoder.encode(texts, normalize)


def encode_texts(texts: list[str], normalize: bool = True) -> list[list[float]]:
    """Encode texts in bounded mini-batches; JSON/Qdrant-friendly vectors."""
    if not texts:
        return []
    return get_encoder().encode(texts, normalize=normalize)


def reset_encoder_cache() -> None:
    """Drop the cached encoder (used after catalog updates and by tests)."""
    get_encoder.cache_clear()


if __name__ == "__main__":
    sample = encode_texts(["Ноутбук Lenovo IdeaPad", "Светильник настольный LED"])
    print(f"Encoded {len(sample)} texts; backend={resolve_backend()}; dim={len(sample[0])}")
