"""Loads the PP-1875 catalog from the parsed JSON file."""

import json

from ml.config import DEBUG, RAW_DATA_DIR

CATALOG_FILE = "data.json"

# Keep only these code_role values. Empty set = keep everything.
# Fill it in after running ml/check_catalog.py, e.g. {"Включён"}.
ACTIVE_CODE_ROLES: set[str] = set()


def load_catalog() -> list[dict]:
    """
    Loads the catalog in the unified internal format.

    Returns:
        List of dicts with fields: id, record_id, name, group, position,
        description, pp_code, pp_link, code_role.
    Falls back to dummy data if the real file is missing (for offline tests).
    """
    path = RAW_DATA_DIR / CATALOG_FILE

    if path.exists():
        with path.open(encoding="utf-8") as f:
            rows = json.load(f)
        return _rows_to_items(rows)

    return _get_dummy_catalog()


def _rows_to_items(rows: list[dict]) -> list[dict]:
    """Maps raw parsed rows to the internal catalog format."""
    items: list[dict] = []
    skipped = 0

    for idx, row in enumerate(rows, start=1):
        name = (row.get("name") or "").strip()
        if not name:
            skipped += 1
            continue
        if ACTIVE_CODE_ROLES and row.get("code_role") not in ACTIVE_CODE_ROLES:
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
            }
        )

    if DEBUG:
        print(f"   Parsed {len(items)} items, skipped {skipped}")

    return items


def _get_dummy_catalog() -> list[dict]:
    """Tiny fallback catalog for offline tests."""
    return [
        {
            "id": 1,
            "record_id": "dummy-1",
            "name": "Электрический перфоратор Bosch GBH 2-26",
            "group": "Приложение 1",
            "position": "Приложение 1, позиция 1",
            "description": "ОКПД2: 28.24",
            "pp_code": "28.24",
            "pp_link": "dummy",
            "code_role": "Включён",
        },
    ]


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