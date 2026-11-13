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


def test_catalog_decision_floors_are_passed_to_api_and_worker(compose) -> None:
    """The shared env anchor feeds both services without rebuilding the model."""
    common = compose["x-app-env"]
    assert common["CATALOG_OK_FLOOR"] == "${CATALOG_OK_FLOOR:-}"
    assert common["CATALOG_REVIEW_FLOOR"] == "${CATALOG_REVIEW_FLOOR:-}"
    assert compose["services"]["api"]["environment"] == common
    assert compose["services"]["worker"]["environment"]["CATALOG_OK_FLOOR"] == (
        "${CATALOG_OK_FLOOR:-}"
    )


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


def test_dockerfile_bakes_the_embedder_with_explicit_fallback() -> None:
    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert "ARG INSTALL_TRANSFORMER" in dockerfile
    assert "requirements-ml.txt" in dockerfile
    assert "scripts/download_model.py" in dockerfile
    assert "build_model.py --dense --gate" in dockerfile
    # A broken ML build must not silently pass the acceptance checks.
    assert "ARG ALLOW_DEGRADE=false" in dockerfile
    assert dockerfile.count('if [ "$ALLOW_DEGRADE" = "true" ]') == 2
    assert "ERROR: transformer step failed (ALLOW_DEGRADE=false)" in dockerfile
    assert "USER app" in dockerfile
    # Fail-fast preflight: a blocked Hugging Face must be diagnosed in seconds,
    # not after ten minutes of dead downloads, and name the step that died.
    assert "huggingface.co/api/models" in dockerfile
    assert "download.pytorch.org" in dockerfile
    assert "WARNING: torch install failed" in dockerfile


def test_corporate_ca_is_wired_from_env_into_the_build(compose) -> None:
    """TLS interception (antivirus/proxy) must be fixable without forking the Dockerfile."""
    build_args = (compose["services"]["api"].get("build") or {}).get("args") or {}
    assert build_args["CORP_CA_PEM_B64"] == "${CORP_CA_PEM_B64:-}"

    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    assert 'ARG CORP_CA_PEM_B64=""' in dockerfile
    # Empty by default: the stock bundle must be used when nothing is passed.
    assert 'if [ -n "$CORP_CA_PEM_B64" ]' in dockerfile
    assert "update-ca-certificates" in dockerfile
    # Both export flavours from certmgr.msc (PEM with headers and raw DER) are accepted.
    assert "BEGIN CERTIFICATE" in dockerfile
    # Python HTTP stacks have to read the merged bundle, and the Rust xet downloader
    # keeps its own root store, so the plain HTTPS path is used instead.
    for variable in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE"):
        assert f"{variable}=" in dockerfile
    assert "HF_HUB_DISABLE_XET=1" in dockerfile


def test_ml_requirements_do_not_reinstall_the_base_stack() -> None:
    # Regression: requirements-ml.txt used to start with `-r requirements.txt`,
    # so the transformer layer re-resolved the whole base stack and re-downloaded
    # the 42 MB ru_core_news_md wheel from github.com — after the base layer had
    # already installed it. A truncated re-download ("Wheel 'ru-core-news-md'
    # ... is invalid", ~47 kB/s to the GitHub CDN) failed otherwise healthy
    # builds after ~20 minutes. The ML file layers on top of the base install.
    ml_requirements = (ROOT / "api" / "requirements-ml.txt").read_text(encoding="utf-8")
    requirement_lines = [
        line.strip()
        for line in ml_requirements.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert not any(line.startswith("-r") for line in requirement_lines), requirement_lines
    assert any(line.startswith("sentence-transformers") for line in requirement_lines)


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


# --- сборка образа ------------------------------------------------------------------


def test_docker_image_ships_every_eval_module_the_training_step_imports() -> None:
    """`build_model.py --dense --gate` runs *inside* the image (api/Dockerfile).

    It imports ``eval.run_eval``, ``eval.dense_gate`` and ``eval.quality_gate``, but
    ``.dockerignore`` drops ``eval/*``. For one release only two of the three were
    copied, so the training step died on ModuleNotFoundError in 0.4 s — swallowed by
    ``|| echo`` — and every image quietly shipped without the dense reranker
    (the log then says «Нет реранкера для эмбеддера … используем лексический»).
    This test derives the required files from the imports themselves.
    """
    import ast

    dockerfile = (ROOT / "api" / "Dockerfile").read_text(encoding="utf-8")
    ignored = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()

    imported: set[str] = set()
    for script in (ROOT / "scripts" / "build_model.py", ROOT / "eval" / "run_eval.py"):
        tree = ast.parse(script.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.startswith("eval."):
                    imported.add(module.split(".", 1)[1])
    assert {"dense_gate", "quality_gate", "run_eval"} <= imported

    for module in sorted(imported):
        path = f"eval/{module}.py"
        assert (ROOT / path).is_file(), f"{path} нет в репозитории"
        assert f"!{path}" in ignored, f"{path} отфильтрован .dockerignore и не попадёт в контекст"
        assert path in dockerfile, f"{path} не копируется в образ (api/Dockerfile)"


def test_worker_builds_from_the_local_definition_instead_of_pulling(compose) -> None:
    """`docker compose up --build` must not start with a failed pull of `dora-app`.

    The worker shares the api image by tag; without its own `build` section compose
    treats that tag as a remote image and prints
    «✘ Image dora-app Error pull access denied for dora-app» before the api build
    provides it. Both services must therefore carry the same build definition.
    """
    services = compose["services"]
    api_build = services["api"]["build"]
    worker_build = services["worker"].get("build")
    assert worker_build == api_build
    assert services["worker"]["image"] == services["api"]["image"]


def test_qdrant_client_pin_matches_the_pinned_server() -> None:
    """A client/server minor gap of more than one makes Qdrant warn on every start.

    The stack logged «Qdrant client version 1.20.0 is incompatible with server
    version 1.14.1» on every boot: the image pins the server, the Python client was
    open-ended. Keep them in one minor, and bump both together (a server jump needs
    a volume backup — see eval/docker-target-checklist.md).
    """
    import re

    compose_text = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    server = re.search(r"image:\s*qdrant/qdrant:v(\d+)\.(\d+)", compose_text)
    assert server, "в docker-compose.yml не найден образ qdrant/qdrant:vX.Y"
    server_minor = int(server.group(2))

    requirements = (ROOT / "api" / "requirements.txt").read_text(encoding="utf-8")
    client = re.search(r"qdrant-client(?:\[[^\]]*\])?>=1\.(\d+),<1\.(\d+)", requirements)
    assert client, "qdrant-client должен быть закреплён как >=1.X,<1.Y"
    low, high = int(client.group(1)), int(client.group(2))
    # The pin must describe the server's own minor and nothing else.
    assert low <= server_minor < high, (low, server_minor, high)
