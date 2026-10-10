"""Console logging configuration with the current correlation ID on every record."""

import logging

from app.core.context import RequestIDFilter


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level)

    if not any(isinstance(handler, logging.StreamHandler) for handler in root.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)-8s [%(name)s] rid=%(request_id)s %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S%z",
            )
        )
        root.addHandler(handler)

    # Uvicorn or test runners may have installed handlers before app import.
    for handler in root.handlers:
        if not any(isinstance(item, RequestIDFilter) for item in handler.filters):
            handler.addFilter(RequestIDFilter())

    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
