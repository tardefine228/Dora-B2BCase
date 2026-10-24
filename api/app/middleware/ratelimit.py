"""Per-client rate limiting (token bucket) for the public API.

Three tiers:

* ``/admin/*`` — a deliberately strict bucket (``RATE_LIMIT_ADMIN_RPS``, default 1),
  because catalog uploads and reindexing are heavyweight operations;
* ``POST /classify/file`` — its own bucket counted in files
  (``RATE_LIMIT_FILE_RPS``/``RATE_LIMIT_FILE_BURST``, default 1 rps, burst 3):
  a single upload turns into up to 100 classifications, so charging it to the
  generic request bucket under-counted the real work by two orders of magnitude;
* everything else — ``RATE_LIMIT_RPS``/``RATE_LIMIT_BURST`` (default 10 rps, burst 20).

``/health*``, ``/metrics`` and the static ``/ui`` are exempt so that probes,
scrapers and the demo frontend never fight the limiter.

The client is the socket peer unless the request arrives from a proxy listed in
``TRUSTED_PROXIES``. Only then is ``X-Forwarded-For`` read, and the address taken
is the **right-most hop that is not itself a trusted proxy**: that is the entry
appended by our own proxy, so a client cannot forge its way into a fresh bucket.
Reading the left-most hop (the old behaviour) let anyone send a random header and
reset their budget on every request.

Rejections use the same error shape as the rest of the API (``{"detail": ...}``)
and carry ``Retry-After`` plus ``X-RateLimit-Limit``/``X-RateLimit-Remaining``.
The counter of rejected requests is exposed in ``GET /metrics``.
"""

from __future__ import annotations

import ipaddress
import math
import threading
import time
from functools import lru_cache

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.core.config import settings
from app.core.metrics import metrics as metrics_store

_EXEMPT_PREFIXES = ("/health", "/metrics", "/ui")
_ADMIN_PREFIX = "/admin"
_FILE_PATHS = frozenset({"/classify/file", "/classify/file/"})
_BUCKET_TTL_SECONDS = 300.0
_MAX_TRACKED_CLIENTS = 10_000


class _Bucket:
    __slots__ = ("tokens", "updated_at")

    def __init__(self, tokens: float, updated_at: float) -> None:
        self.tokens = tokens
        self.updated_at = updated_at


def _parse_ip(value: object) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    if value is None:
        return None
    try:
        return ipaddress.ip_address(str(value).strip())
    except ValueError:
        return None


_Network = ipaddress.IPv4Network | ipaddress.IPv6Network


@lru_cache(maxsize=16)
def _networks(entries: tuple[str, ...]) -> tuple[_Network, ...]:
    """Parse the configured entries once per distinct setting value."""
    networks: list[_Network] = []
    for entry in entries:
        networks.append(ipaddress.ip_network(entry, strict=False))
    return tuple(networks)


def _matches_network(address: str | None, entries: tuple[str, ...]) -> bool:
    """Does an address belong to any configured IP/CIDR network?"""
    if not entries or not address:
        return False
    parsed = _parse_ip(address)
    if parsed is None:
        return False
    for network in _networks(entries):
        if parsed.version == network.version and parsed in network:
            return True
    return False


def _peer_is_trusted(address: str | None, entries: tuple[str, ...]) -> bool:
    """May this socket peer be forwarding for someone else?"""
    return _matches_network(address, entries)


def peer_address(scope: Scope) -> str | None:
    """Address of the TCP peer that opened the connection."""
    client = scope.get("client")
    if client and client[0]:
        return str(client[0])
    return None


def _forwarded_hops(scope: Scope) -> list[str]:
    """All ``X-Forwarded-For`` entries, left (oldest) to right (newest)."""
    hops: list[str] = []
    for name, value in scope.get("headers", []):
        if name == b"x-forwarded-for":
            hops.extend(part.strip() for part in value.decode("latin-1").split(","))
    return [hop for hop in hops if hop]


def client_key(scope: Scope, trusted_proxies: tuple[str, ...] | None = None) -> str:
    """Identify the client for the token bucket.

    * no trusted proxies configured → the socket peer; the header is ignored, so
      forging ``X-Forwarded-For`` does not create a new bucket;
    * a trusted peer → the right-most forwarded hop that is not a trusted proxy,
      i.e. the address our own proxy appended. A client can prepend whatever it
      wants; those entries are to the left and never win;
    * every hop trusted (or unparsable) → the socket peer, which is the only
      address the API actually observed.
    """
    trusted = settings.trusted_proxies if trusted_proxies is None else trusted_proxies
    peer = peer_address(scope)
    if not _peer_is_trusted(peer, trusted):
        return peer or "unknown"

    hops = _forwarded_hops(scope)
    for hop in reversed(hops):
        if _parse_ip(hop) is None:
            continue
        if not _matches_network(hop, trusted):
            return hop
    return peer or "unknown"


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
        if path in _FILE_PATHS:
            return float(settings.rate_limit_file_rps), settings.rate_limit_file_burst
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
        if path.startswith(_ADMIN_PREFIX):
            tier = "admin"
        elif path in _FILE_PATHS:
            tier = "file"
        else:
            tier = "api"
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
