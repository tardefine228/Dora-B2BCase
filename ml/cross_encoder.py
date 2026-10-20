"""Optional cross-encoder stage over the retrieval top-10 (ML-6).

A cross-encoder scores the (query, candidate) pair in one forward
pass, so it sees both texts at once — strictly more signal than
the bi-encoder cosine the dense features use. The cost is one
forward pass per candidate, which is why it rescoring only the
top-10 of the lexical/dense shortlist, not the whole catalog.

The stage is *optional by construction*: without a model on disk
(``MODELS_DIR``/``cross_encoder`` or ``CROSS_ENCODER_MODEL``) it
reports ``available=False`` and every caller must skip it — the
same graceful-degradation discipline as the embedders. A pilot
with the latency and quality measurement lives in
``eval/cross_encoder_pilot.py``; the stage ships only when the
pilot's decision rule (+2 p.p. top-1 at p95 <= 150 ms) passes.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from ml.config import MODELS_DIR

logger = logging.getLogger("dora.ml.cross_encoder")

# Small Russian cross-encoder; the pilot measures whether the
# quality gain is worth ten forward passes per request.
DEFAULT_MODEL = "cointegrated/rubert-tiny2-cross-encoder"
TOP_K = 10


class CrossEncoderReranker:
    """Rescore ``top_k`` candidates with a cross-encoder, if present."""

    def __init__(self, model_name: str | None = None, top_k: int = TOP_K) -> None:
        self.model_name = model_name or os.getenv(
            "CROSS_ENCODER_MODEL", DEFAULT_MODEL
        )
        self.top_k = max(1, int(top_k))
        self._model = None
        self._checked = False

    # --- availability -----------------------------------------------------
    @property
    def available(self) -> bool:
        """True only when a real cross-encoder model is on disk."""
        if self._checked:
            return self._model is not None
        self._checked = True
        local = Path(self.model_name)
        if local.is_absolute() or self.model_name.startswith((".", "~")):
            candidate = local
        else:
            candidate = MODELS_DIR / "cross_encoder" / self.model_name.replace(
                "/", "_"
            )
        if candidate.is_dir():
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(str(candidate))
                logger.info("Кросс-энкодер загружен из %s", candidate)
            except Exception as exc:  # noqa: BLE001 - stage stays optional
                logger.warning("Кросс-энкодер %s не загрузился: %s", candidate, exc)
        else:
            logger.info(
                "Кросс-энкодер %s не найден — ступень пропускается", self.model_name
            )
        return self._model is not None

    # --- scoring ----------------------------------------------------------
    def rescore(
        self, query: str, candidates: list[str]
    ) -> list[float] | None:
        """Match scores for ``candidates``, or ``None`` when unavailable.

        Scores are raw model logits (higher = better match); they are
        NOT calibrated probabilities and must not be shown to users.
        """
        if not self.available or not candidates:
            return None
        pairs = [[query, title] for title in candidates]
        return [float(score) for score in self._model.predict(pairs)]

    def reorder(
        self, query: str, candidates: list[tuple[str, object]]
    ) -> list[tuple[str, object]]:
        """Sort ``(title, payload)`` candidates by cross-encoder score.

        Falls back to the input order when the stage is unavailable,
        so callers can invoke it unconditionally.
        """
        scores = self.rescore(query, [title for title, _payload in candidates])
        if scores is None:
            return candidates
        order = sorted(range(len(candidates)), key=lambda i: -scores[i])
        return [candidates[index] for index in order]
