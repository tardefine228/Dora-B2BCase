"""Catalog file parsing, validation and atomic replacement for admin updates.

The API keeps two synchronized views of the PP-1875 catalog in ``DATA_DIR``:

* ``data.csv`` — consumed by the offline ``catalog`` classifier;
* ``data.json`` — consumed by the ML indexer (``ml.catalog_loader``).

An upload may be in either format; both files are regenerated from the accepted
rows so the two views can never drift apart. Writes go to a temporary file in
the same directory and are then atomically renamed over the previous version.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.schemas import CatalogInfo, CatalogIssue

logger = logging.getLogger("dora.catalog_io")

CSV_DELIMITER = ";"
CSV_COLUMNS = [
    "record_id",
    "appendix",
    "position",
    "name",
    "okpd2",
    "code_role",
    "min_share_pct",
    "max_foreign_share_pct",
    "confirming_document",
    "pdf_page_start",
    "pdf_page_end",
    "source_code_cell",
    "review_required",
]
REQUIRED_COLUMNS = ("name", "okpd2")
MAX_ROWS = 50_000
MAX_NAME_LENGTH = 500
APPENDIX_VALUES = {"1", "2", "3", "4"}
OKPD2_PATTERN = r"^\d{2}(\.\d{1,2}){0,3}$"

CODE_ROLES_REVIEW = {"Не указан", "Исключение"}


def resolve_data_dir() -> Path:
    """Resolve DATA_DIR the same way the classifier resolves the catalog path."""
    from app.core.config import settings

    project_root = Path(__file__).resolve().parents[3]
    data_dir = Path(settings.data_dir or "data").expanduser()
    return data_dir if data_dir.is_absolute() else project_root / data_dir


@dataclass
class ParsedCatalog:
    rows: list[dict] = field(default_factory=list)
    source_format: str = "csv"  # csv | json
    columns: list[str] = field(default_factory=list)


@dataclass
class ValidationReport:
    rows_total: int = 0
    rows_valid: int = 0
    errors: list[CatalogIssue] = field(default_factory=list)
    warnings: list[CatalogIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and self.rows_valid > 0


def parse_catalog_bytes(filename: str, content: bytes) -> ParsedCatalog:
    """Parse an uploaded CSV or JSON catalog. Raises ValueError on bad syntax."""
    if len(content) > 5_000_000:
        raise ValueError("Файл каталога не должен превышать 5 МБ")
    lowered = (filename or "").lower()
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Файл должен быть в кодировке UTF-8") from exc

    if lowered.endswith(".json") or text.lstrip().startswith("["):
        rows = json.loads(text)
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise ValueError("JSON-каталог должен быть массивом объектов")
        return ParsedCatalog(rows=rows, source_format="json", columns=_columns_of(rows))

    reader = csv.DictReader(io.StringIO(text), delimiter=CSV_DELIMITER)
    if not reader.fieldnames:
        raise ValueError("CSV-каталог не содержит заголовка")
    missing = [name for name in REQUIRED_COLUMNS if name not in reader.fieldnames]
    if missing:
        raise ValueError(f"В CSV-каталоге нет обязательных колонок: {missing}")
    rows = [dict(row) for row in reader]
    return ParsedCatalog(rows=rows, source_format="csv", columns=list(reader.fieldnames))


def _columns_of(rows: list[dict]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    return columns or list(CSV_COLUMNS)


def validate_rows(parsed: ParsedCatalog) -> ValidationReport:
    """Check the business rules of the catalog; errors block the upload."""
    report = ValidationReport(rows_total=len(parsed.rows))
    if not parsed.rows:
        report.errors.append(
            CatalogIssue(row=None, field=None, message="Каталог пуст", severity="error")
        )
        return report
    if len(parsed.rows) > MAX_ROWS:
        report.errors.append(
            CatalogIssue(
                row=None,
                field=None,
                message=f"Слишком много строк: {len(parsed.rows)} > {MAX_ROWS}",
                severity="error",
            )
        )
        return report

    seen_record_ids: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()
    import re

    okpd2_re = re.compile(OKPD2_PATTERN)

    for index, row in enumerate(parsed.rows, start=1):
        row_errors = 0

        name = str(row.get("name") or "").strip()
        if not name:
            report.errors.append(
                CatalogIssue(
                    row=index, field="name", message="Пустое наименование позиции", severity="error"
                )
            )
            row_errors += 1
        elif len(name) > MAX_NAME_LENGTH:
            report.errors.append(
                CatalogIssue(
                    row=index,
                    field="name",
                    message=f"Наименование длиннее {MAX_NAME_LENGTH} символов",
                    severity="error",
                )
            )
            row_errors += 1

        okpd2 = str(row.get("okpd2") or "").strip()
        code_role = str(row.get("code_role") or "").strip()
        if not okpd2:
            if code_role in CODE_ROLES_REVIEW:
                report.warnings.append(
                    CatalogIssue(
                        row=index,
                        field="okpd2",
                        message=(
                            f"Код ОКПД2 пуст (code_role={code_role}) — "
                            "позиция помечена для ручной проверки"
                        ),
                        severity="warning",
                    )
                )
            else:
                report.errors.append(
                    CatalogIssue(
                        row=index,
                        field="okpd2",
                        message="Пустой код ОКПД2 без признака исключения",
                        severity="error",
                    )
                )
                row_errors += 1
        elif not okpd2_re.match(okpd2):
            report.errors.append(
                CatalogIssue(
                    row=index,
                    field="okpd2",
                    message=f"Код ОКПД2 не похож на код (получено «{okpd2}»)",
                    severity="error",
                )
            )
            row_errors += 1

        appendix = str(row.get("appendix") or "").strip()
        if appendix and appendix not in APPENDIX_VALUES:
            report.warnings.append(
                CatalogIssue(
                    row=index,
                    field="appendix",
                    message=f"Неожиданное значение приложения: «{appendix}»",
                    severity="warning",
                )
            )

        record_id = str(row.get("record_id") or "").strip()
        if record_id:
            if record_id in seen_record_ids:
                report.warnings.append(
                    CatalogIssue(
                        row=index,
                        field="record_id",
                        message=f"Повторяющийся record_id: «{record_id}»",
                        severity="warning",
                    )
                )
            seen_record_ids.add(record_id)

        pair = (okpd2, name)
        if name and pair in seen_pairs:
            report.warnings.append(
                CatalogIssue(
                    row=index,
                    field="name",
                    message="Повторяющаяся пара (ОКПД2, наименование)",
                    severity="warning",
                )
            )
        seen_pairs.add(pair)

        if row_errors == 0:
            report.rows_valid += 1

    return report


def normalize_rows(parsed: ParsedCatalog) -> list[dict]:
    """Map parsed rows onto the canonical CSV/JSON column set."""
    normalized: list[dict] = []
    for row in parsed.rows:
        item = {column: "" for column in CSV_COLUMNS}
        for key, value in row.items():
            if key in item:
                item[key] = "" if value is None else str(value)
        item["name"] = item["name"].strip()
        item["okpd2"] = item["okpd2"].strip()
        normalized.append(item)
    return normalized


def write_catalog(rows: list[dict], data_dir: Path) -> CatalogInfo:
    """Atomically replace data.csv and data.json, return the resulting facts."""
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "data.csv"
    json_path = data_dir / "data.json"

    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=CSV_COLUMNS, delimiter=CSV_DELIMITER)
    writer.writeheader()
    writer.writerows(rows)
    json_text = json.dumps(rows, ensure_ascii=False, indent=1)

    _atomic_write(csv_path, csv_buffer.getvalue().encode("utf-8"))
    _atomic_write(json_path, json_text.encode("utf-8"))
    logger.info("Каталог обновлён: %d строк в %s", len(rows), data_dir)
    return catalog_facts(data_dir)


def _atomic_write(path: Path, content: bytes) -> None:
    descriptor, temp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        try:
            os.unlink(temp_name)
        except OSError:
            pass
        raise


def catalog_facts(data_dir: Path) -> CatalogInfo:
    """Describe the currently installed catalog (for /admin/catalog)."""
    csv_path = data_dir / "data.csv"
    json_path = data_dir / "data.json"
    existing = [path for path in (csv_path, json_path) if path.is_file()]
    if not existing:
        return CatalogInfo(rows=0, columns=[], sha256="", modified_at=None, source_files=[])

    digest = hashlib.sha256()
    rows = 0
    columns: list[str] = []
    latest_mtime = 0.0
    for path in existing:
        content = path.read_bytes()
        digest.update(content)
        latest_mtime = max(latest_mtime, path.stat().st_mtime)
        if path == csv_path:
            reader = csv.reader(io.StringIO(content.decode("utf-8-sig")), delimiter=CSV_DELIMITER)
            header = next(reader, [])
            columns = list(header)
            rows = sum(1 for _ in reader)

    return CatalogInfo(
        rows=rows,
        columns=columns,
        sha256=digest.hexdigest(),
        modified_at=datetime.fromtimestamp(latest_mtime, tz=timezone.utc).isoformat(
            timespec="seconds"
        ),
        source_files=[path.name for path in existing],
    )
