"""Sanity checks for docker-compose.yml and the API image definition.

Docker itself may be unavailable in the development sandbox; these tests pin
the deployment contract that the target-machine checklist
(``eval/docker-target-checklist.md``) verifies end to end.
"""

from __future__ import annotations

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


def test_stack_has_postgres_redis_and_a_celery_worker(compose) -> None:
    services = compose["services"]
    assert {"api", "worker", "postgres", "redis", "qdrant"} <= set(services)
    # The worker runs the same image as the API and consumes the Celery queue.
    assert services["worker"]["image"] == services["api"]["image"]
    assert "app.worker.celery_app" in " ".join(services["worker"]["command"])
    # Both wait for a healthy database and broker instead of racing their start.
    for name in ("api", "worker"):
        depends = services[name]["depends_on"]
        assert depends["postgres"]["condition"] == "service_healthy"
        assert depends["redis"]["condition"] == "service_healthy"
    assert services["postgres"]["healthcheck"]["test"]
    assert services["redis"]["healthcheck"]["test"]


def test_worker_sees_the_ml_package(compose) -> None:
    # The celery console script does not put the working directory on sys.path
    # (uvicorn does, for the api), so without PYTHONPATH the worker cannot import
    # the sibling `ml` package and every batch job dies with ModuleNotFoundError.
    environment = compose["services"]["worker"]["environment"]
    if isinstance(environment, list):
        environment = dict(item.split("=", 1) for item in environment)
    assert environment.get("PYTHONPATH") == "/app"


def test_jobs_live_in_postgres_and_batches_go_through_redis(compose) -> None:
    for name in ("api", "worker"):
        env = compose["services"][name]["environment"]
        assert env["JOB_STORE"] == "postgres"
        assert env["DATABASE_URL"].startswith("postgresql://")
        assert env["QUEUE_BACKEND"] == "celery"
        assert env["REDIS_URL"].startswith("redis://")
    assert "postgres_data" in (compose.get("volumes") or {})
    assert "postgres_data" in " ".join(compose["services"]["postgres"].get("volumes") or [])


def test_api_and_worker_share_the_catalog_volume(compose) -> None:
    # A catalog uploaded through the API must be visible to the worker process.
    for name in ("api", "worker"):
        assert "catalog_data" in " ".join(compose["services"][name].get("volumes") or [])


def test_vector_search_is_on_and_served_from_qdrant(compose) -> None:
    env = compose["services"]["api"]["environment"]
    assert env["VECTOR_SEARCH"].endswith("true}")
    assert env["VECTOR_STORE"].endswith("qdrant}")
    assert env["QDRANT_URL"] == "http://qdrant:6333"
    build_args = (compose["services"]["api"].get("build") or {}).get("args") or {}
    assert "INSTALL_TRANSFORMER" in build_args and "EMBEDDER_MODEL" in build_args


def test_dockerfile_bakes_the_embedder_with_a_safe_fallback() -> None:
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG INSTALL_TRANSFORMER" in dockerfile
    assert "requirements-ml.txt" in dockerfile
    assert "scripts/download_model.py" in dockerfile
    assert "build_model.py --dense --gate" in dockerfile
    # A failed transformer step must not fail the whole build.
    assert "|| echo" in dockerfile
    assert "USER app" in dockerfile
    # Fail-fast preflight: a blocked Hugging Face must be diagnosed in seconds,
    # not after ten minutes of dead downloads, and name the step that died.
    assert "huggingface.co/api/models" in dockerfile
    assert "download.pytorch.org" in dockerfile
    assert "WARNING: torch install failed" in dockerfile


def test_dockerfile_builds_ui_and_ships_only_dist() -> None:
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    # Sources must not be copied raw: the browser cannot run .tsx or bare imports.
    assert "COPY --chown=app:app frontend ./frontend" not in dockerfile
    assert "npm run build" in dockerfile
    assert "COPY --chown=app:app --from=ui-build /ui/dist ./frontend" in dockerfile


def test_dockerfile_keeps_the_big_downloads_out_of_the_code_edit_path() -> None:
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    lines = dockerfile.splitlines()

    def line_of(needle: str) -> int:
        return next(i for i, line in enumerate(lines) if needle in line)

    downloads = line_of("download.pytorch.org")
    app_copy = line_of("COPY --chown=app:app api/app")
    training = line_of("build_model.py --dense --gate")
    # The ~1 GB toolchain/model layer must not sit behind application-code layers:
    # otherwise every code edit invalidates it and rebuilds pay ~10 minutes of
    # downloads. Training (offline, cheap) may and must see the app code.
    assert downloads < app_copy < training
    # A re-run of the download layer must reuse local caches, not the network.
    assert "--mount=type=cache,target=/root/.cache/pip" in dockerfile
    assert "--mount=type=cache,target=/root/.cache/huggingface" in dockerfile


def test_docker_build_context_keeps_ui_and_source_assets() -> None:
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")

    # Docker cannot COPY a file that the build context has excluded.
    assert "frontend" not in dockerignore
    assert "!data/source_text.txt" in dockerignore
    assert "frontend/node_modules" in dockerignore
    assert "frontend/dist" in dockerignore
    # The transformer build step scores itself on the labeled set.
    assert "!eval/run_eval.py" in dockerignore
    assert "!eval/labeled_examples.csv" in dockerignore
    assert "COPY frontend/ ./" in dockerfile
    assert "data/source_text.txt" in dockerfile


def test_sqlite_job_volume_has_a_writable_mount_point() -> None:
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert "mkdir -p /app/jobs" in dockerfile
    assert "chown app:app /app/jobs" in dockerfile
