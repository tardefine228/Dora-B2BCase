"""Настройка структурированного логирования с correlation id."""

import logging

from app.core.context import RequestIDFilter


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)-8s [%(name)s] rid=%(request_id)s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%S%z",
        )
    )
    handler.addFilter(RequestIDFilter())

    root = logging.getLogger()
    root.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        root.addHandler(handler)

    # Не засоряем логи служебным доступом uvicorn.
    logging.getLogger("uvicorn.access").setLevel("WARNING")
