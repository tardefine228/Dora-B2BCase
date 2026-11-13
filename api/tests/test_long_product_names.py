"""Official composite titles must survive the public request contract intact."""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import pytest
from app.main import app
from app.schemas import PRODUCT_NAME_MAX_LENGTH
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[2]
client = TestClient(app)


def _long_titles() -> list[tuple[str, set[str]]]:
    groups: dict[str, set[str]] = defaultdict(set)
    with (ROOT / "data" / "data.csv").open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter=";"):
            if len(row["name"]) > 500:
                groups[row["name"]].add(row["okpd2"])
    return [(name, codes) for name, codes in groups.items()]


LONG_TITLES = _long_titles()


def test_long_title_fixture_covers_all_three_official_groups() -> None:
    assert len(LONG_TITLES) == 3
    assert sorted(len(name) for name, _ in LONG_TITLES) == [961, 963, 1164]
    assert max(len(name) for name, _ in LONG_TITLES) == 1164
    assert max(len(name) for name, _ in LONG_TITLES) < PRODUCT_NAME_MAX_LENGTH


@pytest.mark.parametrize(
    ("name", "expected_codes"),
    LONG_TITLES,
    ids=[f"long-title-{len(name)}" for name, _ in LONG_TITLES],
)
def test_official_long_title_returns_all_catalog_interpretations(
    name: str,
    expected_codes: set[str],
) -> None:
    response = client.post("/classify", json={"name": name})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "needs_clarification"
    assert body["code"] is None
    option_codes = {option["code"] for option in body["clarification"]["options"]}
    assert expected_codes <= option_codes


def test_batch_accepts_official_long_title_and_rejects_only_beyond_new_cap() -> None:
    name, expected_codes = LONG_TITLES[0]
    response = client.post("/classify/batch", json={"items": [name]})
    assert response.status_code == 202
    status = client.get(response.json()["status_url"]).json()
    assert status["status"] == "completed"
    result = status["results"][0]
    assert result["status"] == "needs_clarification"
    option_codes = {option["code"] for option in result["clarification"]["options"]}
    assert expected_codes <= option_codes

    at_cap = client.post("/classify", json={"name": "x" * PRODUCT_NAME_MAX_LENGTH})
    assert at_cap.status_code == 200
    beyond_cap = client.post(
        "/classify",
        json={"name": "x" * (PRODUCT_NAME_MAX_LENGTH + 1)},
    )
    assert beyond_cap.status_code == 422
