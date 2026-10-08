from ml.catalog_loader import item_to_text, load_catalog


def test_ml_indexer_uses_the_checked_in_real_json_catalog() -> None:
    catalog = load_catalog()
    assert len(catalog) == 1126
    assert catalog[0]["name"] == "Ткани текстильные"
    assert catalog[0]["pp_code"] == "13.2"
    assert "ОКПД2: 13.2" in item_to_text(catalog[0])
