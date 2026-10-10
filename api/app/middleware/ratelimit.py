"""Per-client rate limiting (token bucket) for the public API.

Two tiers:

* ``/admin/*`` — a deliberately strict bucket (``RATE_LIMIT_ADMIN_RPS``, default 1),
  because catalog uploads and reindexing are heavyweight operations;
* everything else — ``RATE_LIMIT_RPS``/``RATE_LIMIT_BURST`` (default 10 rps, burst 20).

``/health*``, ``/metrics`` and the static ``/ui`` are exempt so that probes,
scrapers and the demo frontend never fight the limiter. The client is identified
by the first ``X-Forwarded-For`` address (a reverse proxy in front of the API)
falling back to the socket peer.

Rejections use the same error shape as the rest of the API (``{"detail": ...}``)
and carry ``Retry-After`` plus ``X-RateLimit-Limit``/``X-RateLimit-Remaining``.
The counter of rejected requests is exposed in ``GET /metrics``.
"""

from __future__ import annotations

import math
import threading
import time

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import settings
from app.core.metrics import metrics as metrics_store

_EXEMPT_PREFIXES = ("/health", "/metrics", "/ui")
_ADMIN_PREFIX = "/admin"
_BUCKET_TTL_SECONDS = 300.0
_MAX_TRACKED_CLIENTS = 10_000


class _Bucket:
    __slots__ = ("tokens", "updated_at")

    def __init__(self, tokens: float, updated_at: float) -> None:
        self.tokens = tokens
        self.updated_at = updated_at


def client_key(scope: Scope) -> str:
    """First ``X-Forwarded-For`` hop when present, otherwise the socket peer."""
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-for":
            forwarded = value.decode("latin-1").split(",")[0].strip()
            if forwarded:
                return forwarded
            break
    client = scope.get("client")
    if client:
        return str(client[0])
    return "unknown"


def is_exempt(path: str) -> bool:
    return path == "/metrics" or path.startswith(_EXEMPT_PREFIXES)


class RateLimitMiddleware:
    """ASGI middleware; settings are read per request so tests can toggle them."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str], _Bucket] = {}

    # --- internals --------------------------------------------------------
    def _cleanup_locked(self, now: float) -> None:
        if len(self._buckets) <= _MAX_TRACKED_CLIENTS:
            return
        stale = [
            key
            for key, bucket in self._buckets.items()
            if now - bucket.updated_at > _BUCKET_TTL_SECONDS
        ]
        for key in stale:
            self._buckets.pop(key, None)

    def _consume(
        self, key: tuple[str, str], *, rate: float, burst: int, now: float
    ) -> tuple[bool, float, int]:
        """Charge one token; returns (allowed, remaining, retry_after_seconds)."""
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(tokens=float(burst), updated_at=now)
                self._buckets[key] = bucket
                self._cleanup_locked(now)
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(float(burst), bucket.tokens + elapsed * rate)
                bucket.updated_at = now

            if bucket.tokens >= 1.0:
                bucket.tokens -= 1.0
                return True, int(bucket.tokens), 0
            retry_after = max(1, math.ceil((1.0 - bucket.tokens) / rate))
            return False, 0, retry_after

    def _limits(self, path: str) -> tuple[float, int]:
        if path.startswith(_ADMIN_PREFIX):
            return float(settings.rate_limit_admin_rps), settings.rate_limit_admin_burst
        return float(settings.rate_limit_rps), settings.rate_limit_burst

    # --- ASGI -------------------------------------------------------------
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or not settings.rate_limit_enabled:
            await self.app(scope, receive, send)
            return

        path = scope.get("path", "")
        if is_exempt(path):
            await self.app(scope, receive, send)
            return

        rate, burst = self._limits(path)
        tier = "admin" if path.startswith(_ADMIN_PREFIX) else "api"
        allowed, remaining, retry_after = self._consume(
            (client_key(scope), tier), rate=rate, burst=burst, now=time.monotonic()
        )

        if not allowed:
            metrics_store.record_rate_limited(path)
            response = JSONResponse(
                status_code=429,
                content={
                    "detail": (
                        "Слишком много запросов: превышен лимит "
                        f"{rate:g} зап/с (burst {burst}). Повторите позже."
                    )
                },
                headers={
                    "Retry-After": str(retry_after),
                    "X-RateLimit-Limit": str(burst),
                    "X-RateLimit-Remaining": "0",
                },
            )
            await response(scope, receive, send)
            return

        limit_header = str(burst).encode()
        remaining_header = str(remaining).encode()

        async def send_with_limit_headers(message: Message) -> None:
            if message["type"] == "http.response.start":
                headers = [
                    (name, value)
                    for name, value in (message.get("headers") or [])
                    if name not in (b"x-ratelimit-limit", b"x-ratelimit-remaining")
                ]
                headers.append((b"x-ratelimit-limit", limit_header))
                headers.append((b"x-ratelimit-remaining", remaining_header))
                message = {**message, "headers": headers}
            await send(message)

        await self.app(scope, receive, send_with_limit_headers)
