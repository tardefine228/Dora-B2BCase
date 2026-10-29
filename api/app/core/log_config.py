"""Console logging configuration with correlation IDs on every record.

Two formats share one setup:

* ``text`` — the historical human-readable line for local ``uvicorn`` runs;
* ``json`` — one JSON object per line (``LOG_FORMAT=json``), the default of the
  Docker stack, so log aggregators get structured ``level``/``request_id``/
  ``message`` fields without parsing Russian prose.

Uvicorn installs its own handlers before the application module is imported, so
its loggers are re-pointed at the same formatter instead of keeping the default
access-log layout.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from app.core.context import RequestIDFilter, request_id_var

_TEXT_FORMAT = "%(asctime)s %(levelname)-8s [%(name)s] rid=%(request_id)s %(message)s"
_TEXT_DATEFMT = "%Y-%m-%dT%H:%M:%S%z"

# Standard LogRecord attributes (built from a dummy record) plus request_id,
# which the RequestIDFilter attaches. Everything else is caller-provided extra.
_RESERVED_ATTRS = set(
    logging.LogRecord(
        name="", level=0, pathname="", lineno=0, msg="", args=None, exc_info=None
    ).__dict__.keys()
) | {"request_id"}

_UVICORN_LOGGERS = ("uvicorn", "uvicorn.error", "uvicorn.access")


class JsonFormatter(logging.Formatter):
    """Render every record as a single compact JSON object on one line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "request_id": getattr(record, "request_id", request_id_var.get()),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)
        extras = {
            key: value for key, value in record.__dict__.items() if key not in _RESERVED_ATTRS
        }
        if extras:
            payload["extra"] = extras
        # default=str keeps unusual payloads (Path, Decimal, ...) serializable.
        return json.dumps(payload, ensure_ascii=False, default=str)


def _build_formatter(log_format: str) -> logging.Formatter:
    if log_format == "json":
        return JsonFormatter()
    if log_format == "text":
        return logging.Formatter(fmt=_TEXT_FORMAT, datefmt=_TEXT_DATEFMT)
    raise ValueError(f"LOG_FORMAT must be 'text' or 'json', got {log_format!r}")


def configure_logging(level: str = "INFO", log_format: str = "text") -> None:
    """Configure the root logger and unify the uvicorn loggers on the same format."""
    root = logging.getLogger()
    root.setLevel(level)
    formatter = _build_formatter(log_format)

    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        root.addHandler(logging.StreamHandler())

    # Re-running the configuration (tests, catalog reload) must switch the
    # formatter instead of leaving the first one in place forever.
    for handler in root.handlers:
        if isinstance(handler, logging.StreamHandler):
            handler.setFormatter(formatter)
        if not any(isinstance(item, RequestIDFilter) for item in handler.filters):
            handler.addFilter(RequestIDFilter())

    # Uvicorn's own loggers keep separate handlers; give them our formatter and
    # correlation filter so the whole stream has one machine-parseable shape.
    for name in _UVICORN_LOGGERS:
        logger = logging.getLogger(name)
        for handler in logger.handlers:
            handler.setFormatter(formatter)
            if not any(isinstance(item, RequestIDFilter) for item in handler.filters):
                handler.addFilter(RequestIDFilter())

    # The observer middleware emits its own structured access line.
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
