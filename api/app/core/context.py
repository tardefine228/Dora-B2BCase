"""Контекст запроса: correlation id, доступный из логов."""

from contextvars import ContextVar
from logging import Filter, LogRecord

# Значение по умолчанию, чтобы formatter не падал вне HTTP-запроса.
request_id_var: ContextVar[str] = ContextVar("request_id", default="-")


class RequestIDFilter(Filter):
    """Добавляет ``request_id`` в каждую запись лога."""

    def filter(self, record: LogRecord) -> bool:
        record.request_id = request_id_var.get()
        return True
