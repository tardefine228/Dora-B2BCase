"""OPS-6: structured JSON logging and the legacy text format share one setup."""

from __future__ import annotations

import io
import json
import logging

import pytest
from app.core import log_config
from app.core.context import request_id_var
from app.core.log_config import JsonFormatter, configure_logging


@pytest.fixture()
def captured_logger():
    """Restore the global logging state after each test."""
    root = logging.getLogger()
    saved_handlers = list(root.handlers)
    saved_level = root.level
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root.handlers = [handler]
    root.setLevel(logging.INFO)
    try:
        yield root, handler, stream
    finally:
        root.handlers = saved_handlers
        root.setLevel(saved_level)
        request_id_var.set("-")


def test_json_formatter_emits_one_parseable_object_per_record() -> None:
    record = logging.LogRecord(
        name="dora.api",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="Привет, %s",
        args=("мир",),
        exc_info=None,
    )
    payload = json.loads(JsonFormatter().format(record))

    assert payload["message"] == "Привет, мир"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "dora.api"
    assert payload["request_id"] == "-"
    assert payload["ts"].endswith("+00:00")


def test_json_lines_carry_the_correlation_id(captured_logger) -> None:
    root, handler, stream = captured_logger
    configure_logging("INFO", "json")
    token = request_id_var.set("rid-123")
    try:
        logging.getLogger("dora.api").info("запрос обработан")
    finally:
        request_id_var.reset(token)

    payload = json.loads(stream.getvalue().strip())
    assert payload["request_id"] == "rid-123"
    assert payload["message"] == "запрос обработан"


def test_json_formatter_serializes_exceptions_and_extras(captured_logger) -> None:
    root, handler, stream = captured_logger
    configure_logging("INFO", "json")

    try:
        raise ValueError("boom")
    except ValueError:
        logging.getLogger("dora.api").exception("сбой", extra={"route": "/classify"})

    payload = json.loads(stream.getvalue().strip())
    assert payload["level"] == "ERROR"
    assert "ValueError: boom" in payload["exc_info"]
    assert payload["extra"] == {"route": "/classify"}


def test_text_format_keeps_the_historical_layout(captured_logger) -> None:
    root, handler, stream = captured_logger
    configure_logging("INFO", "text")
    token = request_id_var.set("rid-9")
    try:
        logging.getLogger("dora.api").info("старая строка")
    finally:
        request_id_var.reset(token)

    line = stream.getvalue().strip()
    assert "INFO" in line and "[dora.api]" in line
    assert "rid=rid-9" in line and "старая строка" in line
    with pytest.raises(json.JSONDecodeError):
        json.loads(line)


def test_reconfiguring_switches_the_formatter(captured_logger) -> None:
    root, handler, stream = captured_logger

    configure_logging("INFO", "text")
    logging.getLogger("dora.api").info("раз")
    assert "rid=-" in stream.getvalue()

    stream.seek(0)
    stream.truncate(0)
    configure_logging("INFO", "json")
    logging.getLogger("dora.api").info("два")
    assert json.loads(stream.getvalue().strip())["message"] == "два"


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="LOG_FORMAT"):
        log_config._build_formatter("yaml")


def test_uvicorn_loggers_use_the_same_formatter(captured_logger) -> None:
    root, _handler, _stream = captured_logger
    uvicorn_error = logging.getLogger("uvicorn.error")
    own_handler = logging.StreamHandler(io.StringIO())
    uvicorn_error.addHandler(own_handler)
    try:
        configure_logging("INFO", "json")
        # Existing uvicorn handlers are re-pointed at the JSON formatter...
        assert isinstance(own_handler.formatter, JsonFormatter)
        # ...loggers without handlers keep propagating to the JSON-configured root.
        plain = logging.getLogger("uvicorn.plain")
        assert plain.handlers == [] and plain.propagate is True
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
        assert logging.getLogger("uvicorn.access").level == logging.WARNING
    finally:
        uvicorn_error.removeHandler(own_handler)
