"""ASGI middleware for correlation IDs, timings, metrics, and safe 500 responses."""

import logging
import re
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.context import request_id_var
from app.core.metrics import metrics as metrics_store

logger = logging.getLogger("dora.api")
_REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")


class ObserveMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        supplied_id = request.headers.get("X-Request-ID", "")
        request_id = supplied_id if _REQUEST_ID_RE.fullmatch(supplied_id) else uuid.uuid4().hex
        context_token = request_id_var.set(request_id)
        started = time.perf_counter()
        response: Response
        status_code = 500

        try:
            try:
                response = await call_next(request)
                status_code = response.status_code
            except Exception:
                # Never expose implementation details in an unexpected 500 response.
                logger.exception(
                    "Unhandled request exception method=%s path=%s",
                    request.method,
                    request.url.path,
                )
                response = JSONResponse(
                    status_code=500,
                    content={
                        "detail": "Внутренняя ошибка сервера",
                        "request_id": request_id,
                    },
                )

            elapsed_ms = (time.perf_counter() - started) * 1000
            route = request.scope.get("route")
            metric_path = getattr(route, "path", "<unmatched>")
            metrics_store.record(metric_path, status_code, elapsed_ms)
            response.headers["X-Request-ID"] = request_id
            response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"

            content_length = request.headers.get("content-length", "unknown")
            logger.info(
                "%s %s status=%s latency_ms=%.2f content_length=%s",
                request.method,
                metric_path,
                status_code,
                elapsed_ms,
                content_length,
            )
            return response
        finally:
            request_id_var.reset(context_token)
