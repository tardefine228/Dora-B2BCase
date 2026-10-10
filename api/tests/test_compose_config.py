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


def test_compose_healthchecks_api_and_starts_minimal_qdrant(compose) -> None:
    services = compose["services"]
    assert {"api", "qdrant"} <= set(services)

    api_healthcheck = services["api"].get("healthcheck") or {}
    assert api_healthcheck.get("test"), "api: healthcheck.test отсутствует"
    assert api_healthcheck.get("interval")
    assert api_healthcheck.get("retries")

    # The official Qdrant image has no curl/wget binary, so don't define an
    # in-container HTTP healthcheck that will always fail. Check its HTTP endpoint
    # from the host, and only require that Qdrant has started before launching API.
    assert "healthcheck" not in services["qdrant"]
    dependency = services["api"]["depends_on"]["qdrant"]
    assert dependency["condition"] == "service_started"


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


def test_job_store_survives_restart(compose) -> None:
    api = compose["services"]["api"]
    assert api["environment"]["JOB_STORE"] == "${JOB_STORE:-sqlite}"
    assert api["environment"]["JOB_STORE_PATH"].endswith("jobs.sqlite")
    api_volumes = " ".join(api.get("volumes") or [])
    assert "jobs_data" in api_volumes
    assert "jobs_data" in (compose.get("volumes") or {})


def test_install_ml_flag_controls_ml_stack(compose) -> None:
    api = compose["services"]["api"]
    build_args = (api.get("build") or {}).get("args") or {}
    assert "INSTALL_ML" in build_args
    assert api["environment"]["ML_ENABLED"]


def test_dockerfile_installs_ml_only_on_request() -> None:
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert re.search(r"ARG INSTALL_ML", dockerfile)
    assert "requirements-ml.txt" in dockerfile
    assert "USER app" in dockerfile


def test_dockerfile_builds_ui_and_ships_only_dist() -> None:
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    # Sources must not be copied raw: the browser cannot run .tsx or bare imports.
    assert "COPY --chown=app:app frontend ./frontend" not in dockerfile
    assert "npm run build" in dockerfile
    assert "COPY --chown=app:app --from=ui-build /ui/dist ./frontend" in dockerfile


def test_docker_build_context_keeps_ui_and_source_assets() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")

    # Docker cannot COPY a file that the build context has excluded.
    assert "frontend" not in dockerignore
    assert "!data/source_text.txt" in dockerignore
    assert "frontend/node_modules" in dockerignore
    assert "frontend/dist" in dockerignore
    assert "COPY frontend/ ./" in dockerfile
    assert "data/source_text.txt" in dockerfile


def test_sqlite_job_volume_has_a_writable_mount_point() -> None:
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert "mkdir -p /app/jobs" in dockerfile
    assert "chown app:app /app/jobs" in dockerfile
