"""Простая внутрипроцессная статистика запросов (без внешних зависимостей)."""

import threading
from collections import defaultdict


class Metrics:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total = 0
        self.by_status: dict[int, int] = defaultdict(int)
        self.latency_sum = 0.0
        self.by_path: dict[str, int] = defaultdict(int)
        self.rate_limited = 0
        self.rate_limited_by_path: dict[str, int] = defaultdict(int)

    def record(self, path: str, status: int, latency_ms: float) -> None:
        with self._lock:
            self.total += 1
            self.by_status[status] += 1
            self.latency_sum += latency_ms
            self.by_path[path] += 1

    def record_rate_limited(self, path: str) -> None:
        """Count a request rejected by the rate limiter (HTTP 429)."""
        with self._lock:
            self.rate_limited += 1
            self.rate_limited_by_path[path] += 1

    def snapshot(self) -> dict:
        with self._lock:
            avg = (self.latency_sum / self.total) if self.total else 0.0
            return {
                "total_requests": self.total,
                "requests_by_status": dict(self.by_status),
                "average_latency_ms": round(avg, 3),
                "requests_by_path": dict(self.by_path),
                "rate_limited_total": self.rate_limited,
                "rate_limited_by_path": dict(self.rate_limited_by_path),
            }


metrics = Metrics()
