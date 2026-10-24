"""Optional third stage: an LLM confirms the top-3 hypotheses for weak rows.

Why this stage exists (``eval/experiments.md``, эксп. 30): after the lexicon and
the purchase-line pass, the catalog backend still stays silent on goods that
*are* in the list but whose wording is far from the official one — «Внешний
жёсткий диск 2 ТБ» against «Устройства запоминающие и прочие устройства
хранения данных». The lexical model's top-1 is often right there, but the group
confidence is below the floor: `pair_probability` 0.13–0.15, margin 0.03. More
bridges cannot fix that class (it is a wording gap, not a vocabulary one), so a
language model is asked a *narrow* question: "which of these ≤3 positions is the
goods, or none?".

Rules the module enforces, because a verifier must never become a second,
unaccounted classifier:

* it only runs on ``review``/``not_found`` answers — an ``ok`` answer is final;
* it may only choose a code that was offered to it, or answer ``none``;
  anything else (invented code, prose, broken JSON, timeout) is ignored;
* its verdict is reported as ``review`` (a candidate for a human), never ``ok``;
* it fails open: no key, no model, no network or an exhausted budget means the
  answer stays exactly the model's own.

The provider is any OpenAI-compatible chat-completions endpoint (OpenAI,
OpenRouter, vLLM, Ollama). It is off by default: ``LLM_VERIFIER_ENABLED=1``,
``LLM_VERIFIER_MODEL`` and ``LLM_VERIFIER_API_KEY`` turn it on.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass

logger = logging.getLogger("dora.services.llm_verifier")

MAX_CANDIDATES = 3
CACHE_SIZE = 1024

SYSTEM_PROMPT = (
    "Ты помогаешь специалисту по закупкам сопоставить товар с позицией перечня "
    "ПП № 1875 (ОКПД2). Тебе дают название товара и не более трёх позиций перечня "
    "с кодами. Выбери ровно одну позицию, которая описывает товар, или ответь "
    "none, если ни одна не подходит. Нельзя предлагать коды, которых нет в списке, "
    "и нельзя придумывать позиции. Ответ — только JSON без пояснений в формате "
    '{"choice": "<код из списка>"} или {"choice": "none"}.'
)


@dataclass(frozen=True)
class Candidate:
    """One offered hypothesis: a code and the position title it belongs to."""

    code: str
    title: str


@dataclass(frozen=True)
class VerifierVerdict:
    """What the model answered. ``code is None`` means "none of these"."""

    code: str | None
    cached: bool = False


class BudgetExhausted(Exception):
    """Raised internally when the per-minute call budget is spent."""


class LlmVerifier:
    """Constrained, cached, budgeted verifier over the top-3 hypotheses."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 8.0,
        max_candidates: int = MAX_CANDIDATES,
        calls_per_minute: int = 60,
        cache_size: int = CACHE_SIZE,
        transport: Callable[[dict, float], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_candidates = max(1, min(int(max_candidates), MAX_CANDIDATES))
        self.calls_per_minute = max(1, int(calls_per_minute))
        self.cache_size = max(0, int(cache_size))
        self._transport = transport
        self._clock = clock
        self._cache: dict[tuple[str, tuple[str, ...]], VerifierVerdict] = {}
        self._window: deque[float] = deque()
        self._lock = threading.Lock()
        self._stats = {"calls": 0, "cache_hits": 0, "answered": 0, "failed": 0, "skipped": 0}

    # ------------------------------------------------------------------ public
    name = "llm-top3"

    @property
    def enabled(self) -> bool:
        return bool(self.base_url and self.model)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return dict(self._stats)

    def verify(self, query: str, candidates: Sequence[Candidate]) -> VerifierVerdict | None:
        """Ask the model which candidate is the goods; ``None`` = no usable answer.

        ``None`` is returned for every reason that must not change the answer:
        no candidates, a disabled verifier, an exhausted budget, a transport
        error, or a reply that is not one of the offered codes / ``none``.
        """
        offered = _dedupe(candidates)[: self.max_candidates]
        if not self.enabled or not offered or not query.strip():
            return None

        key = (_key(query), tuple(candidate.code for candidate in offered))
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self._stats["cache_hits"] += 1
                return VerifierVerdict(code=cached.code, cached=True)

        try:
            content = self._call(query, offered)
        except BudgetExhausted:
            with self._lock:
                self._stats["skipped"] += 1
            return None
        except Exception:  # noqa: BLE001 - a verifier must never break classification
            logger.warning("LLM-верификатор: запрос не удался, ответ модели остаётся как есть")
            with self._lock:
                self._stats["failed"] += 1
            return None

        verdict = self._parse(content, offered)
        with self._lock:
            if verdict is None:
                self._stats["failed"] += 1
                return None
            self._stats["answered"] += 1
            self._remember(key, verdict)
        return verdict

    # ----------------------------------------------------------------- private
    def _remember(self, key: tuple[str, tuple[str, ...]], verdict: VerifierVerdict) -> None:
        if self.cache_size == 0:
            return
        if len(self._cache) >= self.cache_size:
            # Simple FIFO eviction: the working set is a batch of purchase lines.
            self._cache.pop(next(iter(self._cache)))
        self._cache[key] = verdict

    def _call(self, query: str, offered: Sequence[Candidate]) -> str:
        now = self._clock()
        with self._lock:
            while self._window and now - self._window[0] >= 60.0:
                self._window.popleft()
            if len(self._window) >= self.calls_per_minute:
                raise BudgetExhausted
            self._window.append(now)
            self._stats["calls"] += 1
        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 60,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": _user_prompt(query, offered)},
            ],
        }
        if self._transport is not None:
            return self._transport(payload, self.timeout)
        return self._http(payload)

    def _http(self, payload: dict) -> str:
        """POST to an OpenAI-compatible endpoint and return the message content."""
        import httpx

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/chat/completions", json=payload, headers=headers
            )
            response.raise_for_status()
            body = response.json()
        return body["choices"][0]["message"]["content"]

    @staticmethod
    def _parse(content: str, offered: Sequence[Candidate]) -> VerifierVerdict | None:
        text = (content or "").strip()
        if not text:
            return None
        # A chat model may wrap JSON in a fenced block; the payload itself is fixed.
        if text.startswith("```"):
            text = text.strip("`")
            text = text[text.find("{") :] if "{" in text else text
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        choice = payload.get("choice")
        if not isinstance(choice, str):
            return None
        choice = choice.strip()
        if choice.casefold() == "none":
            return VerifierVerdict(code=None)
        allowed = {candidate.code for candidate in offered}
        if choice not in allowed:
            # The model proposed a code nobody offered: ignore, do not invent.
            logger.info(
            "LLM-верификатор предложил код вне top-3 (%s) — ответ отклонён", choice
        )
            return None
        return VerifierVerdict(code=choice)


def _dedupe(candidates: Sequence[Candidate]) -> list[Candidate]:
    seen: set[str] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        code = (candidate.code or "").strip()
        if not code or code in seen:
            continue
        seen.add(code)
        unique.append(Candidate(code=code, title=(candidate.title or "").strip()))
    return unique


def _key(query: str) -> str:
    return " ".join(query.casefold().split())


def _user_prompt(query: str, offered: Sequence[Candidate]) -> str:
    positions = "\n".join(
        f"{index}. {candidate.code} — {candidate.title}"
        for index, candidate in enumerate(offered, 1)
    )
    return (
        f"Товар: {query.strip()}\n\n"
        f"Позиции перечня:\n{positions}\n\n"
        'Ответь JSON: {"choice": "<код из списка>"} или {"choice": "none"}.'
    )




def build_llm_verifier(settings) -> LlmVerifier | None:
    """Construct the verifier from settings, or ``None`` when it is off.

    Off is the default and also the state of a half-configured deployment: an
    enabled flag without a model would only add latency, so it is treated as off.
    """
    if not getattr(settings, "llm_verifier_enabled", False):
        return None
    model = (getattr(settings, "llm_verifier_model", "") or "").strip()
    if not model:
        logger.warning("LLM_VERIFIER_ENABLED=1, но LLM_VERIFIER_MODEL не задан — ступень выключена")
        return None
    return LlmVerifier(
        base_url=getattr(settings, "llm_verifier_base_url", "https://api.openai.com/v1"),
        model=model,
        api_key=getattr(settings, "llm_verifier_api_key", ""),
        timeout=float(getattr(settings, "llm_verifier_timeout_seconds", 8.0)),
        calls_per_minute=int(getattr(settings, "llm_verifier_calls_per_minute", 60)),
    )
