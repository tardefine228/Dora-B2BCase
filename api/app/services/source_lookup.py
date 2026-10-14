"""Source fragments for catalog positions: ``clause_url`` and ``GET /source/{record_id}``.

The parsed catalog (``data/data.csv`` / ``data/data.json``) does not contain a URL
to the official publication: ``pp_link`` is a human-readable citation
("ПП № 1875, приложение N, позиция M, стр. X–Y"). The checked-in
``data/source_text.txt`` holds the machine-readable text of the decree with
``=== СТРАНИЦА N ===`` markers, and every catalog row carries
``pdf_page_start``/``pdf_page_end`` inside that file.

Therefore the source of record is exposed by the service itself: every
classification result points at ``/source/{record_id}``, and this module resolves
that route to the pages of the decree the position was parsed from.
"""

from __future__ import annotations

import csv
import json
import logging
import re
import threading
from pathlib import Path
from urllib.parse import quote

from app.schemas import SourceFragment, SourcePages

logger = logging.getLogger("dora.api.source")

SOURCE_TEXT_NAME = "source_text.txt"
CATALOG_CSV_NAME = "data.csv"
CATALOG_JSON_NAME = "data.json"

# ``=== СТРАНИЦА 12 ===`` in data/source_text.txt (see data/README of the parser).
_PAGE_MARKER_RE = re.compile(r"^===+\s*СТРАНИЦА\s+(\d+)\s*===+\s*$", re.IGNORECASE)
# One decree page is normally a few thousand characters; cap the payload so a
# bad page range cannot return megabytes.
MAX_FRAGMENT_CHARS = 20_000

_CATALOG_COLUMNS = (
    "record_id",
    "appendix",
    "position",
    "name",
    "okpd2",
    "pdf_page_start",
    "pdf_page_end",
)


class SourceUnavailableError(RuntimeError):
    """The record exists in the catalog, but its source text cannot be read."""


def clause_url_for(record_id: str | None) -> str | None:
    """Public URL of the source fragment for a catalog record (relative to the API)."""
    record_id = (record_id or "").strip()
    if not record_id:
        return None
    return f"/source/{quote(record_id, safe='')}"


def _clause_ref(appendix: str, position: str) -> str | None:
    if appendix and position:
        return f"Приложение {appendix}, позиция {position}"
    if appendix:
        return f"Приложение {appendix}"
    return None


def _as_int(value: object) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


class SourceLookup:
    """Resolves catalog ``record_id`` to a fragment of the decree text.

    Both the catalog rows and the page index are cached and re-read when the
    underlying files change (``mtime``): the catalog can be replaced at runtime
    through ``POST /admin/catalog``.
    """

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self._lock = threading.Lock()
        self._rows: dict[str, dict] | None = None
        self._rows_stamp: float | None = None
        self._pages: dict[int, str] | None = None
        self._pages_stamp: float | None = None

    # --- catalog rows -----------------------------------------------------
    def _catalog_stamp(self) -> tuple[float, Path] | None:
        stamps = [
            (path.stat().st_mtime, path)
            for path in (
                self.data_dir / CATALOG_CSV_NAME,
                self.data_dir / CATALOG_JSON_NAME,
            )
            if path.is_file()
        ]
        if not stamps:
            return None
        stamp, path = max(stamps)
        return stamp, path

    def _load_rows(self) -> dict[str, dict]:
        csv_path = self.data_dir / CATALOG_CSV_NAME
        json_path = self.data_dir / CATALOG_JSON_NAME
        rows: dict[str, dict] = {}
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle, delimiter=";"):
                    record_id = (row.get("record_id") or "").strip()
                    if record_id:
                        rows[record_id] = row
        elif json_path.is_file():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            for row in payload if isinstance(payload, list) else []:
                record_id = str(row.get("record_id") or "").strip()
                if record_id:
                    rows[record_id] = row
        return rows

    def _rows_cached(self) -> dict[str, dict]:
        stamp_info = self._catalog_stamp()
        if stamp_info is None:
            raise SourceUnavailableError("Каталог не найден: нет data.csv или data.json")
        stamp, _path = stamp_info
        with self._lock:
            if self._rows is not None and self._rows_stamp == stamp:
                return self._rows
            rows = self._load_rows()
            self._rows = rows
            self._rows_stamp = stamp
            logger.info("Каталог для источников обновлён: %d записей", len(rows))
            return rows

    # --- page index -------------------------------------------------------
    def _source_text_path(self) -> Path:
        return self.data_dir / SOURCE_TEXT_NAME

    def _load_pages(self) -> dict[int, str]:
        path = self._source_text_path()
        if not path.is_file():
            raise SourceUnavailableError(
                f"Текст постановления не найден: {path}; положите source_text.txt в DATA_DIR"
            )
        pages: dict[int, list[str]] = {}
        current: int | None = None
        for line in path.read_text(encoding="utf-8").splitlines():
            marker = _PAGE_MARKER_RE.match(line)
            if marker:
                current = int(marker.group(1))
                pages.setdefault(current, [])
                continue
            if current is not None:
                pages[current].append(line)
        return {page: "\n".join(lines).strip() for page, lines in pages.items()}

    def _pages_cached(self) -> dict[int, str]:
        path = self._source_text_path()
        if not path.is_file():
            raise SourceUnavailableError(
                f"Текст постановления не найден: {path}; положите source_text.txt в DATA_DIR"
            )
        stamp = path.stat().st_mtime
        with self._lock:
            if self._pages is not None and self._pages_stamp == stamp:
                return self._pages
            pages = self._load_pages()
            self._pages = pages
            self._pages_stamp = stamp
            logger.info("Текст постановления прочитан: %d страниц", len(pages))
            return pages

    # --- public API -------------------------------------------------------
    def record(self, record_id: str) -> dict | None:
        return self._rows_cached().get(record_id)

    def fragment(self, record_id: str) -> SourceFragment | None:
        """Return the page fragment for ``record_id``; ``None`` if the id is unknown.

        Raises ``SourceUnavailableError`` when the record exists but the decree
        text file is missing or does not cover the recorded pages.
        """
        row = self.record(record_id)
        if row is None:
            return None

        from app.services.catalog_bundle import source_is_current, source_metadata

        if not source_is_current(self.data_dir):
            raise SourceUnavailableError(
                "Источник не соответствует каталогу. "
                "Загрузите ZIP с каталогом и текстом одной редакции."
            )

        pages = self._pages_cached()
        start = _as_int(row.get("pdf_page_start"))
        end = _as_int(row.get("pdf_page_end")) or start
        if start is not None and end is not None and start > end:
            start, end = end, start
        if start is not None and end is not None and (start < 1 or end - start > 200):
            raise SourceUnavailableError("Некорректный диапазон страниц в каталоге.")

        selected: list[str] = []
        if start is not None and end is not None:
            selected = [pages[page] for page in range(start, end + 1) if page in pages]
        if (
            not selected
            or start is None
            or end is None
            or any(page not in pages for page in range(start, end + 1))
        ):
            raise SourceUnavailableError(
                "Нужные страницы источника недоступны; "
                "обновите текст постановления вместе с каталогом."
            )

        text = "\n\n".join(part for part in selected if part)
        truncated = len(text) > MAX_FRAGMENT_CHARS
        if truncated:
            text = text[:MAX_FRAGMENT_CHARS]

        appendix = str(row.get("appendix") or "").strip()
        position = str(row.get("position") or "").strip()
        edition, source_url = source_metadata(self.data_dir)
        return SourceFragment(
            record_id=record_id,
            name=(row.get("name") or "").strip() or None,
            okpd2=(row.get("okpd2") or "").strip() or None,
            clause_ref=_clause_ref(appendix, position),
            pages=SourcePages(start=start, end=end),
            text=text,
            truncated=truncated,
            source_file=SOURCE_TEXT_NAME,
            clause_url=clause_url_for(record_id),
            edition=edition,
            source_url=source_url,
        )


_lookup: SourceLookup | None = None
_lookup_lock = threading.Lock()


def get_source_lookup(data_dir: Path | None = None) -> SourceLookup:
    """Process-wide lookup instance for the configured ``DATA_DIR``."""
    global _lookup
    with _lookup_lock:
        if _lookup is None or (data_dir is not None and Path(data_dir) != _lookup.data_dir):
            if data_dir is None:
                from app.services.catalog_io import resolve_data_dir

                data_dir = resolve_data_dir()
            _lookup = SourceLookup(Path(data_dir))
        return _lookup


def reset_source_lookup() -> None:
    """Drop the cached lookup (used by tests and after a catalog upload)."""
    global _lookup
    with _lookup_lock:
        _lookup = None
