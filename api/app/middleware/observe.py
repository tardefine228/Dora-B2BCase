"""Middleware: correlation id, измерение времени запроса и логирование."""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware

from app.core.context import request_id_var
from app.core.metrics import metrics as metrics_store

logger = logging.getLogger("dora.api")


class ObserveMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        rid = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        request_id_var.set(rid)
        started = time.perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            request_id_var.set("-")
            raise

        latency_ms = (time.perf_counter() - started) * 1000
        metrics_store.record(request.url.path, response.status_code, latency_ms)
        response.headers["X-Request-ID"] = rid
        response.headers["X-Process-Time-Ms"] = f"{latency_ms:.2f}"
        logger.info(
            "%s %s status=%s latency_ms=%.2f",
            request.method,
            request.url.path,
            response.status_code,
            latency_ms,
        )
        request_id_var.set("-")
        return response
