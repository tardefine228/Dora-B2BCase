"""Loads product catalog from files."""

import json
from ml.config import RAW_DATA_DIR


def load_catalog() -> list[dict]:
    """
    Loads the product catalog.

    First tries to read data/raw/pp1875_catalog.json.
    If the file doesn't exist — returns dummy data for testing.

    Returns:
        List of dicts with fields: id, name, group, description, pp_code, pp_link.
    """
    catalog_path = RAW_DATA_DIR / "pp1875_catalog.json"

    if catalog_path.exists():
        with catalog_path.open(encoding="utf-8") as f:
            return json.load(f)

    # Dummy data for testing
    return _get_dummy_catalog()


def _get_dummy_catalog() -> list[dict]:
    """Returns a toy catalog for tests."""
    return [
        {
            "id": 1,
            "name": "Электрический перфоратор Bosch GBH 2-26",
            "group": "Электроинструмент",
            "description": "Мощность 830 Вт, энергия удара 2.7 Дж",
            "pp_code": "142.1",
            "pp_link": "https://example.com/pp1875/142.1",
        },
        {
            "id": 2,
            "name": "Ударная дрель Makita HP1640",
            "group": "Электроинструмент",
            "description": "Мощность 680 Вт",
            "pp_code": "142.2",
            "pp_link": "https://example.com/pp1875/142.2",
        },
        {
            "id": 3,
            "name": "Смеситель для кухни Grohe Eurosmart",
            "group": "Сантехника",
            "description": "Однорычажный, с поворотным изливом",
            "pp_code": "220.5",
            "pp_link": "https://example.com/pp1875/220.5",
        },
        {
            "id": 4,
            "name": "Силовой медный кабель ВВГнг 3х2.5",
            "group": "Электротовары",
            "description": "Трёхжильный, сечение 2.5 мм²",
            "pp_code": "305.1",
            "pp_link": "https://example.com/pp1875/305.1",
        },
        {
            "id": 5,
            "name": "Ноутбук Lenovo ThinkPad X1 Carbon",
            "group": "Оргтехника",
            "description": "14 дюймов, Intel Core i7",
            "pp_code": "410.3",
            "pp_link": "https://example.com/pp1875/410.3",
        },
    ]


def item_to_text(item: dict) -> str:
    """
    Converts a product into text for vectorization.

    Concatenates name, group and description — this gives better search quality
    than just the name alone.
    """
    parts = [item["name"]]

    if item.get("group"):
        parts.append(f"Группа: {item['group']}")

    if item.get("description"):
        parts.append(f"Описание: {item['description']}")

    return ". ".join(parts)


if __name__ == "__main__":
    catalog = load_catalog()
    print(f"Loaded products: {len(catalog)}")
    print("\nExample text for vectorization:")
    print(item_to_text(catalog[0]))