"""Loads the PP-1875 catalog from the parsed JSON file."""

import hashlib
import json

from ml.config import DEBUG, RAW_DATA_DIR
from ml.preprocessing import excluded_codes

CATALOG_FILE = "data.json"


def catalog_fingerprint(items: list[dict]) -> str:
    """Stable identity of the catalog actually seen by training and retrieval."""
    rows = [
        {key: item.get(key) for key in ("record_id", "name", "pp_code", "excluded_codes")}
        for item in items
    ]
    rows.sort(key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True))
    payload = json.dumps(rows, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_catalog() -> list[dict]:
    """
    Loads the catalog in the unified internal format.

    Returns:
        List of dicts with fields: id, record_id, name, group, position,
        description, pp_code, pp_link, code_role.
    Raises ``FileNotFoundError`` rather than silently indexing a dummy catalog.
    """
    path = RAW_DATA_DIR / CATALOG_FILE
    if not path.is_file():
        raise FileNotFoundError(f"Catalog JSON not found: {path}")

    with path.open(encoding="utf-8") as f:
        rows = json.load(f)
    if not isinstance(rows, list):
        raise ValueError(f"Catalog JSON must contain an array: {path}")
    return _rows_to_items(rows)


def _rows_to_items(rows: list[dict]) -> list[dict]:
    """Maps raw parsed rows to the internal catalog format."""
    items: list[dict] = []
    skipped = 0
    exceptions = excluded_codes(rows)

    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"Catalog row {idx} is not an object")
        name = (row.get("name") or "").strip()
        if not name:
            skipped += 1
            continue
        if row.get("code_role") in {"Исключение", "Не указан"} or not row.get("okpd2"):
            skipped += 1
            continue

        appendix = row.get("appendix")
        position = row.get("position")
        okpd2 = row.get("okpd2") or ""

        items.append(
            {
                "id": idx,
                "record_id": row.get("record_id", ""),
                "name": name,
                "group": f"Приложение {appendix}",
                "position": f"Приложение {appendix}, позиция {position}",
                "description": f"ОКПД2: {okpd2}",
                "pp_code": okpd2,
                "pp_link": (
                    f"ПП № 1875, приложение {appendix}, позиция {position}, "
                    f"стр. {row.get('pdf_page_start')}–{row.get('pdf_page_end')}"
                ),
                "code_role": row.get("code_role"),
                "excluded_codes": exceptions.get((str(appendix), str(position)), []),
            }
        )

    if DEBUG:
        print(f"   Parsed {len(items)} items, skipped {skipped}")

    return items


def item_to_text(item: dict) -> str:
    """
    Builds the text to vectorize.

    Name carries the semantics; OKPD2 code helps literal code queries.
    Position numbers are kept out of the text on purpose: they add noise
    to semantic search and already live in the payload.
    """
    parts = [item["name"]]

    if item.get("description"):
        parts.append(item["description"])

    return ". ".join(parts)
