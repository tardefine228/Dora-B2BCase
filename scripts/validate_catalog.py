"""Проверка файла каталога ПП № 1875 без установки (см. docs/catalog-update.md).

Использование:
    python scripts/validate_catalog.py path/to/data.csv [--json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "api"))

from app.services.catalog_io import parse_catalog_bytes, validate_rows  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path)
    parser.add_argument("--json", action="store_true", help="вывод в JSON")
    args = parser.parse_args()

    content = args.file.read_bytes()
    try:
        parsed = parse_catalog_bytes(args.file.name, content)
    except ValueError as exc:
        print(f"Файл не разобран: {exc}")
        return 2

    report = validate_rows(parsed)
    if args.json:
        print(
            json.dumps(
                {
                    "rows_total": report.rows_total,
                    "rows_valid": report.rows_valid,
                    "errors": [issue.model_dump() for issue in report.errors],
                    "warnings": [issue.model_dump() for issue in report.warnings],
                },
                ensure_ascii=False,
                indent=1,
            )
        )
    else:
        print(f"Строк: {report.rows_total}, валидных: {report.rows_valid}")
        for issue in report.errors:
            print(f"  [ошибка] строка {issue.row}: {issue.field}: {issue.message}")
        for issue in report.warnings:
            print(f"  [предупреждение] строка {issue.row}: {issue.field}: {issue.message}")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
