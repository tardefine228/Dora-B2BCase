"""Compare an external PP-1875 snapshot with the working catalog.

The user periodically exports the raw decree list (``pp_1875_dataframe.csv``)
from their own parser. That snapshot and ``data/data.csv`` describe the same
1126 positions but are not byte-identical by design:

* the snapshot zero-pads ``record_id`` (``1-001-01`` vs ``1-1-01``);
* share columns are written as integers (``80`` vs ``80.0``);
* the working catalog carries the QA-only ``review_required`` column;
* ten names were corrected against the source PDF in two review rounds — the
  four position-402 volume bounds (``docs/pdf-source-review-2026-10-25.md``) and
  the position-362 ``Х$`` → ``XS`` / position-394 ``опоры- ходунки`` →
  ``опоры-ходунки`` fixes (``eval/data-review-2026-10-26/source-verification.json``),
  so the raw OCR text in a fresh snapshot is expected to differ there and nowhere
  else.

This script makes that contract executable: every row must align on the
normalized id, codes/roles/pages must be identical, numeric shares must hold
the same values, and the only allowed name diffs are the whitelisted PDF
corrections. Anything else fails with a non-zero exit code.

Usage::

    python scripts/verify_catalog_snapshot.py \\
        eval/source-snapshots/2026-10-26/pp_1875_dataframe.csv \\
        --json eval/source-snapshots/2026-10-26/verification.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Columns that must be character-identical once both files are parsed.
EXACT_COLUMNS = (
    "appendix",
    "position",
    "name",
    "okpd2",
    "code_role",
    "confirming_document",
    "pdf_page_start",
    "pdf_page_end",
    "source_code_cell",
)
# Percent columns: "80" and "80.0" are the same value.
NUMERIC_COLUMNS = ("min_share_pct", "max_foreign_share_pct")
# Columns the working catalog maintains on top of the raw decree snapshot.
CATALOG_ONLY_COLUMNS = ("review_required",)
# Rows whose catalog name carries a PDF correction, so the raw OCR snapshot is
# expected to differ there (and only there):
# * 2-402-1…4 — volume bounds ``≥ 500 мл ≤ 1000 мл`` (visual review 2026-10-25);
# * 2-362-1…4 — ``Х$`` → ``XS`` and 2-394-1…2 — ``опоры- ходунки`` →
#   ``опоры-ходунки`` (line-by-line source verification 2026-10-26,
#   eval/data-review-2026-10-26/source-verification.json).
EXPECTED_NAME_CORRECTIONS = (
    "2-362-1",
    "2-362-2",
    "2-362-3",
    "2-362-4",
    "2-394-1",
    "2-394-2",
    "2-402-1",
    "2-402-2",
    "2-402-3",
    "2-402-4",
)


def read_catalog(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def normalize_record_id(record_id: str) -> str:
    """``1-001-01`` and ``1-1-01`` name the same position."""
    return "-".join(
        str(int(part)) if part.lstrip("-").isdigit() else part for part in record_id.split("-")
    )


def _as_float(value: str) -> float | None:
    value = (value or "").strip().replace(",", ".")
    if not value:
        return None
    return float(value)


def compare(snapshot_rows: list[dict], catalog_rows: list[dict]) -> dict:
    """Return a structured comparison; ``ok`` is False on any contract break."""
    snapshot = {normalize_record_id(row["record_id"]): row for row in snapshot_rows}
    catalog = {normalize_record_id(row["record_id"]): row for row in catalog_rows}
    if len(snapshot) != len(snapshot_rows) or len(catalog) != len(catalog_rows):
        raise ValueError("дублирующиеся record_id после нормализации")

    only_snapshot = sorted(set(snapshot) - set(catalog))
    only_catalog = sorted(set(catalog) - set(snapshot))
    snapshot_columns = set(snapshot_rows[0])
    catalog_columns = set(catalog_rows[0])

    exact_diffs: dict[str, list[dict]] = {column: [] for column in EXACT_COLUMNS}
    numeric_diffs: dict[str, list[dict]] = {}
    numeric_format_only = 0
    for record_id in sorted(set(snapshot) & set(catalog)):
        source = snapshot[record_id]
        working = catalog[record_id]
        for column in EXACT_COLUMNS:
            if (source.get(column) or "") != (working.get(column) or ""):
                exact_diffs[column].append(
                    {
                        "record_id": record_id,
                        "snapshot": (source.get(column) or "")[:200],
                        "catalog": (working.get(column) or "")[:200],
                    }
                )
        for column in NUMERIC_COLUMNS:
            left = (source.get(column) or "").strip()
            right = (working.get(column) or "").strip()
            if left == right:
                continue
            if _as_float(left) == _as_float(right):
                numeric_format_only += 1
                continue
            numeric_diffs.setdefault(column, []).append(
                {"record_id": record_id, "snapshot": left, "catalog": right}
            )

    name_diffs = {item["record_id"] for item in exact_diffs["name"]}
    unexpected_name_diffs = sorted(name_diffs - set(EXPECTED_NAME_CORRECTIONS))
    missing_expected = sorted(set(EXPECTED_NAME_CORRECTIONS) - name_diffs)
    other_exact_diffs = {
        column: items for column, items in exact_diffs.items() if column != "name" and items
    }

    review_required = sum(
        1 for row in catalog_rows if (row.get("review_required") or "").strip().lower() == "true"
    )
    ok = not (
        only_snapshot
        or only_catalog
        or unexpected_name_diffs
        or missing_expected
        or any(other_exact_diffs.values())
        or numeric_diffs
    )
    return {
        "ok": ok,
        "rows": {"snapshot": len(snapshot_rows), "catalog": len(catalog_rows)},
        "aligned_ids": len(set(snapshot) & set(catalog)),
        "only_in_snapshot": only_snapshot,
        "only_in_catalog": only_catalog,
        "columns": {
            "snapshot_only": sorted(snapshot_columns - catalog_columns),
            "catalog_only": sorted(catalog_columns - snapshot_columns),
        },
        "name_diffs": exact_diffs["name"],
        "expected_name_corrections": list(EXPECTED_NAME_CORRECTIONS),
        "unexpected_name_diffs": unexpected_name_diffs,
        "missing_expected_corrections": missing_expected,
        "other_exact_diffs": other_exact_diffs,
        "numeric_value_diffs": numeric_diffs,
        "numeric_format_only_cells": numeric_format_only,
        "catalog_review_required_true": review_required,
    }


def render_markdown(report: dict, snapshot_path: Path) -> str:
    lines = [
        "# Сверка снапшота перечня ПП №1875 с рабочим каталогом",
        "",
        f"Снапшот: `{snapshot_path}`; каталог: `data/data.csv`.",
        "",
        "| Показатель | Значение |",
        "| --- | --- |",
        f"| Строк в снапшоте / каталоге | "
        f"{report['rows']['snapshot']} / {report['rows']['catalog']} |",
        f"| Совпавших id после нормализации | {report['aligned_ids']} |",
        f"| Только в снапшоте | {len(report['only_in_snapshot'])} |",
        f"| Только в каталоге | {len(report['only_in_catalog'])} |",
        "| Отличий долей только в формате (`80` vs `80.0`) | "
        f"{report['numeric_format_only_cells']} |",
        "| Реальных отличий значений долей | "
        f"{sum(len(v) for v in report['numeric_value_diffs'].values())} |",
        "| Отличий в кодах/ролях/страницах | "
        f"{sum(len(v) for v in report['other_exact_diffs'].values())} |",
        f"| Текстовых отличий (все — whitelist PDF-правок) | {len(report['name_diffs'])} |",
        "| `review_required=true` в каталоге (колонки нет в снапшоте) | "
        f"{report['catalog_review_required_true']} |",
        "",
        "Вердикт: **"
        + ("снапшот соответствует каталогу" if report["ok"] else "ОБНАРУЖЕНЫ РАСХОЖДЕНИЯ")
        + "**.",
        "",
        "## Текстовые отличия (ожидаемые PDF-правки)",
        "",
    ]
    for item in report["name_diffs"]:
        lines += [
            f"### `{item['record_id']}`",
            "",
            f"- снапшот (сырой OCR): {item['snapshot']}",
            f"- каталог (после визуальной сверки PDF): {item['catalog']}",
            "",
        ]
    if not report["name_diffs"]:
        lines.append("Нет — whitelist нужно пересмотреть.\n")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "data.csv")
    parser.add_argument("--json", dest="json_path", type=Path, default=None)
    parser.add_argument("--markdown", type=Path, default=None)
    args = parser.parse_args(argv)

    report = compare(read_catalog(args.snapshot), read_catalog(args.catalog))
    if args.json_path is not None:
        args.json_path.parent.mkdir(parents=True, exist_ok=True)
        args.json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", "utf-8")
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(render_markdown(report, args.snapshot), "utf-8")
    summary = {
        "ok": report["ok"],
        "rows": report["rows"],
        "name_diffs": len(report["name_diffs"]),
    }
    print(json.dumps(summary, ensure_ascii=False))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
