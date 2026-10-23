"""API contract v1.0 tests executed through a real HTTP client.

These tests pin the result format agreed with the frontend team
(``docs/api-contract.md``): breaking them means breaking ``frontend/app.js``.
"""

from __future__ import annotations

import asyncio

import httpx
from app.main import app

REQUIRED_RESULT_FIELDS = {
    "query",
    "code",
    "title",
    "confidence",
    "status",
    "clause_ref",
    "clause_url",
    "alternatives",
    "latency_ms",
    "contract_version",
}


async def _call(coro_factory):
    """Run one request against the ASGI app through httpx.AsyncClient."""
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        return await coro_factory(client)


def call(coro_factory):
    return asyncio.run(_call(coro_factory))


def test_classify_result_matches_contract_v1() -> None:
    response = call(lambda c: c.post("/classify", json={"name": "Ноутбук Lenovo"}))
    assert response.status_code == 200
    body = response.json()
    assert REQUIRED_RESULT_FIELDS <= set(body)
    assert body["contract_version"] == "1.0"
    assert body["status"] in {"ok", "review", "not_found"}
    assert 0.0 <= body["confidence"] <= 1.0
    for alternative in body["alternatives"]:
        assert {"code", "title", "confidence", "clause_ref"} <= set(alternative)


def test_batch_job_flow_over_http() -> None:
    async def flow(client):
        accepted = await client.post(
            "/classify/batch",
            json={"items": ["Ноутбук Lenovo", "Цемент", "абвгдеёжз qwerty"]},
        )
        assert accepted.status_code == 202
        job = accepted.json()
        assert {"job_id", "status", "total", "status_url", "contract_version"} <= set(job)
        assert job["contract_version"] == "1.0"

        status = await client.get(job["status_url"])
        assert status.status_code == 200
        state = status.json()
        assert state["contract_version"] == "1.0"
        return state

    state = call(flow)
    assert state["status"] == "completed"
    assert state["progress_percent"] == 100.0
    assert len(state["results"]) == 3
    for result in state["results"]:
        assert REQUIRED_RESULT_FIELDS <= set(result)
    # The not-found contract: no code/title when the status is not_found.
    for result in state["results"]:
        if result["status"] == "not_found":
            assert result["code"] is None
            assert result["title"] is None


def test_error_details_are_json_and_sanitized() -> None:
    response = call(lambda c: c.post("/classify", json={"name": ""}))
    assert response.status_code == 422
    assert isinstance(response.json()["detail"], list)

    missing = call(lambda c: c.get("/classify/batch/does-not-exist"))
    assert missing.status_code == 404
    assert isinstance(missing.json()["detail"], str)


def test_admin_endpoints_require_key() -> None:
    for path in ("/admin/catalog", "/admin/reindex"):
        get_response = call(lambda c, p=path: c.get(p))
        # 405 = the path exists only for other methods; auth is still enforced there.
        assert get_response.status_code in {403, 405, 503}
        post_response = call(lambda c, p=path: c.post(p))
        assert post_response.status_code in {403, 405, 422, 503}


def test_openapi_serves_documented_paths() -> None:
    schema = call(lambda c: c.get("/openapi.json")).json()["paths"]
    for path in ("/classify", "/classify/batch", "/classify/batch/{job_id}", "/admin/catalog"):
        assert path in schema


def test_static_ui_is_served() -> None:
    response = call(lambda c: c.get("/ui/"))
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "1875" in response.text


def test_cors_allows_browser_clients() -> None:
    response = call(lambda c: c.get("/health", headers={"Origin": "http://example.com"}))
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in {"*", "http://example.com"}
