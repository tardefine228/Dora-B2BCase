"""The decree fragment endpoint must be readable for people, not only parsers."""

from app.main import app
from fastapi.testclient import TestClient

RECORD_ID = "2-195-01"  # ноутбуки: the row the labeled set and the UI link to


def test_source_endpoint_serves_a_readable_page_to_browsers() -> None:
    with TestClient(app) as client:
        page = client.get(
            f"/source/{RECORD_ID}", headers={"Accept": "text/html,application/xhtml+xml"}
        )
        assert page.status_code == 200
        assert page.headers["content-type"].startswith("text/html")
        assert "<!doctype html>" in page.text
        assert "Компьютеры портативные" in page.text
        # The decree entry is highlighted and the official sources are linked.
        assert 'class="pos hit"' in page.text
        assert 'class="code">26.20.11<' in page.text
        assert "publication.pravo.gov.ru" in page.text
        assert "government.ru" in page.text
        # Raw JSON must not leak into the browser view.
        assert '"record_id"' not in page.text


def test_source_endpoint_keeps_json_for_api_clients() -> None:
    with TestClient(app) as client:
        answer = client.get(f"/source/{RECORD_ID}", headers={"Accept": "application/json"})
        assert answer.status_code == 200
        assert answer.headers["content-type"].startswith("application/json")
        payload = answer.json()
        assert payload["record_id"] == RECORD_ID
        assert payload["text"]
