"""OPS-6: Prometheus alert rules, Grafana dashboard and the monitoring profile.

The strongest guarantee here is the cross-check: every ``dora_*`` metric name
referenced by an alert or a dashboard panel must be emitted by
``api/app/core/metrics.py``. A renamed metric then breaks this test instead of
silently emptying a panel or disabling an alert.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml", reason="PyYAML не установлен")

ROOT = Path(__file__).resolve().parents[2]
MONITORING = ROOT / "deploy" / "monitoring"
METRIC_RE = re.compile(r"dora_[a-z_]+")
# Histograms expose <name>_bucket/_sum/_count series although only the base name
# appears in the Counter/Histogram declaration in metrics.py.
_HISTOGRAM_SUFFIXES = ("_bucket", "_sum", "_count")


def _base_metric(name: str) -> str:
    for suffix in _HISTOGRAM_SUFFIXES:
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


@pytest.fixture(scope="module")
def emitted_metrics() -> set[str]:
    source = (ROOT / "api" / "app" / "core" / "metrics.py").read_text(encoding="utf-8")
    return set(METRIC_RE.findall(source))


@pytest.fixture(scope="module")
def rules() -> dict:
    return yaml.safe_load(
        (MONITORING / "prometheus" / "rules" / "dora-alerts.yml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def prometheus_config() -> dict:
    return yaml.safe_load(
        (MONITORING / "prometheus" / "prometheus.yml").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def dashboard() -> dict:
    return json.loads((MONITORING / "grafana" / "dashboards" / "dora.json").read_text("utf-8"))


@pytest.fixture(scope="module")
def compose() -> dict:
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_every_alert_rule_metric_is_emitted(rules, emitted_metrics) -> None:
    alert_names = []
    for group in rules["groups"]:
        for rule in group["rules"]:
            alert_names.append(rule["alert"])
            referenced = {_base_metric(n) for n in METRIC_RE.findall(rule["expr"])}
            unknown = referenced - emitted_metrics
            assert not unknown, f"{rule['alert']}: неизвестные метрики {unknown}"
    # The seven alerts of the documented runbook.
    assert set(alert_names) == {
        "DoraClassifierDown",
        "DoraEmbedderDegraded",
        "DoraVectorStoreDown",
        "DoraHighErrorRate",
        "DoraHighLatencyP95",
        "DoraBatchJobsFailing",
        "DoraRateLimitSaturated",
    }
    # Every alert carries severity and a human description.
    for group in rules["groups"]:
        for rule in group["rules"]:
            assert rule["labels"]["severity"] in {"critical", "warning", "info"}
            assert rule["annotations"]["summary"]
            assert rule["annotations"]["description"]


def test_prometheus_scrapes_the_api_metrics_endpoint(prometheus_config) -> None:
    scrape = prometheus_config["scrape_configs"][0]
    assert scrape["job_name"] == "dora-api"
    assert scrape["metrics_path"] == "/metrics"
    assert "api:8000" in scrape["static_configs"][0]["targets"]
    rule_files = prometheus_config["rule_files"]
    assert any("dora-alerts.yml" in path for path in rule_files)


def test_dashboard_json_is_valid_and_metrics_exist(dashboard, emitted_metrics) -> None:
    assert dashboard["uid"] == "dora-b2b-overview"
    panels = dashboard["panels"]
    assert len([p for p in panels if p["type"] != "row"]) >= 8

    referenced: set[str] = set()
    for panel in panels:
        for target in panel.get("targets", []):
            referenced |= {_base_metric(n) for n in METRIC_RE.findall(target.get("expr", ""))}
    assert referenced, "дашборд без единого Prometheus-запроса"
    assert referenced <= emitted_metrics, f"неизвестные метрики: {referenced - emitted_metrics}"

    # Every data panel points at the provisioned datasource uid.
    for panel in panels:
        if panel["type"] == "row":
            continue
        assert panel["datasource"]["uid"] == "dora-prometheus"


def test_grafana_provisioning_points_at_prometheus() -> None:
    datasource = yaml.safe_load(
        (MONITORING / "grafana" / "provisioning" / "datasources" / "prometheus.yml").read_text(
            "utf-8"
        )
    )
    assert datasource["datasources"][0]["uid"] == "dora-prometheus"
    assert datasource["datasources"][0]["url"] == "http://prometheus:9090"

    provider = yaml.safe_load(
        (MONITORING / "grafana" / "provisioning" / "dashboards" / "dashboards.yml").read_text(
            "utf-8"
        )
    )
    assert provider["providers"][0]["options"]["path"] == "/var/lib/grafana/dashboards"


def test_monitoring_services_are_profile_gated_and_loopback_only(compose) -> None:
    services = compose["services"]
    for name in ("prometheus", "grafana"):
        service = services[name]
        assert service["profiles"] == ["monitoring"], f"{name} должен быть за профилем"
        for mapping in service["ports"]:
            assert mapping.startswith("127.0.0.1:"), f"{name} не должен слушать 0.0.0.0"
        assert service["mem_limit"]
    # The default stack must not gain monitoring dependencies.
    core = {name for name in services if "profiles" not in services[name]}
    assert {"api", "worker", "postgres", "redis", "qdrant"} <= core
    assert "prometheus" not in core and "grafana" not in core
    # State survives restarts on named volumes.
    assert "prometheus_data" in compose["volumes"]
    assert "grafana_data" in compose["volumes"]
    # Grafana provisioning and dashboards are mounted read-only.
    grafana_volumes = " ".join(services["grafana"]["volumes"])
    assert ":ro" in grafana_volumes


def test_runtime_policy_and_log_format_reach_the_app_environment(compose) -> None:
    env = compose["x-app-env"]
    assert env["ALLOW_DEGRADE"] == "${ALLOW_DEGRADE:-false}"
    assert env["LOG_FORMAT"] == "${LOG_FORMAT:-json}"
    for service in ("api", "worker"):
        assert compose["services"][service]["environment"]["ALLOW_DEGRADE"] == (
            "${ALLOW_DEGRADE:-false}"
        )
