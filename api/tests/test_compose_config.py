"""Sanity checks for docker-compose.yml and the API image definition.

Docker itself may be unavailable in the development sandbox; these tests pin
the deployment contract that the target-machine checklist
(``eval/docker-target-checklist.md``) verifies end to end.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML не установлен")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_compose_defines_api_and_qdrant_with_healthchecks(compose) -> None:
    services = compose["services"]
    assert {"api", "qdrant"} <= set(services)
    for name in ("api", "qdrant"):
        healthcheck = services[name].get("healthcheck") or {}
        assert healthcheck.get("test"), f"{name}: healthcheck.test отсутствует"
        assert healthcheck.get("interval")
        assert healthcheck.get("retries")


def test_qdrant_data_survives_restart(compose) -> None:
    volumes = compose.get("volumes") or {}
    assert "qdrant_data" in volumes
    qdrant_volumes = " ".join(compose["services"]["qdrant"].get("volumes") or [])
    assert "qdrant_data" in qdrant_volumes


def test_catalog_data_has_a_named_volume(compose) -> None:
    # Uploaded catalogs must survive `docker compose restart`.
    api_volumes = " ".join(compose["services"]["api"].get("volumes") or [])
    assert "catalog_data" in api_volumes
    assert "catalog_data" in (compose.get("volumes") or {})


def test_install_ml_flag_controls_ml_stack(compose) -> None:
    api = compose["services"]["api"]
    build_args = (api.get("build") or {}).get("args") or {}
    assert "INSTALL_ML" in build_args
    assert api["environment"]["ML_ENABLED"]


def test_dockerfile_installs_ml_only_on_request() -> None:
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"ARG INSTALL_ML", dockerfile)
    assert "requirements-ml.txt" in dockerfile
    assert "COPY --chown=app:app frontend" in dockerfile or "frontend" in dockerfile
    assert "USER app" in dockerfile
