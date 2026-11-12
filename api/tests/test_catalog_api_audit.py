"""Pure checks for the production API audit; the CI job exercises HTTP/E5."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _module():
    path = ROOT / "scripts" / "catalog_api_audit.py"
    spec = importlib.util.spec_from_file_location("catalog_api_audit", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_audit_covers_the_complete_catalog_and_reviewed_generic_heads():
    module = _module()
    rows = module.catalog_rows(ROOT / "data" / "data.csv")
    exact = module.exact_queries(rows)
    generic = module.generic_queries(rows)
    assert len(exact) == 746
    assert sum(len(codes) > 1 for _, codes in exact) == 87
    assert sum(not codes for _, codes in exact) == 6
    assert len(generic) == 103


def test_exact_audit_requires_every_sibling_code_and_real_single_code_answer():
    module = _module()
    pending = {
        "status": "needs_clarification",
        "code": None,
        "options": [{"code": "26.11.3"}, {"code": "26.11.30"}],
    }
    assert module.check_exact(pending, {"26.11.3", "26.11.30"}) is None
    assert "sibling" in module.check_exact(pending, {"26.11.3", "26.11.30", "26.11.31"})
    assert module.check_exact({"status": "ok", "code": "15.2", "options": []}, {"15.2"}) is None
    assert module.check_exact({"status": "review", "code": "15.2", "options": []}, {"15.2"})
    assert module.check_exact({"status": "ok", "code": "26.11.3", "options": []}, set())
    long_title = {
        "query": "x" * 1164,
        "status": "needs_clarification",
        "code": None,
        "options": [{"code": "26.11.3"}, {"code": "26.11.30"}],
    }
    assert module.check_exact(long_title, {"26.11.3", "26.11.30"}) is None
    assert module.check_exact({**long_title, "status": "validation_error"}, {"26.11.3"})


def test_generic_and_quality_audit_do_not_accept_arbitrary_clarifications():
    module = _module()
    pending = {
        "query": "Кабель",
        "status": "needs_clarification",
        "code": None,
        "options": [{"code": "27.32"}, {"code": "27.31.1"}],
    }
    assert module.check_generic(pending) is None
    assert module.check_generic({**pending, "options": [{"code": "27.32"}]})
    assert module.check_generic({**pending, "query": "Одежда", "status": "ok"}) is None
    assert module.check_generic({**pending, "query": "Одежда"})
    medical = {**pending, "query": "Медицинские", "status": "not_found"}
    assert module.check_generic(medical) is None
    assert module.check_generic({**medical, "status": "ok"})

    row = {"expected_okpd2": "27.31.1", "label_kind": "code"}
    assert module.check_quality(pending, row) is True
    assert module.check_quality({**pending, "options": [{"code": "27.32"}]}, row) is False
    absent = {"expected_okpd2": "", "label_kind": "absent"}
    assert module.check_quality({**pending, "status": "not_found"}, absent) is True
    assert module.check_quality(pending, absent) is False


def test_github_comment_contains_runtime_identity_and_actionable_failures():
    module = _module()
    failure = {
        "query": "Генератор",
        "expected": "27.90.40.150",
        "status": "ok",
        "code": "27.11.3",
        "options": [],
    }
    report = {
        "summary": {
            "readiness": {
                "vector_search": {
                    "embedder": "st-multilingual-e5-test",
                    "store": "qdrant",
                    "positions": 984,
                },
                "job_store": "postgres",
                "queue_backend": "celery",
            },
            "calls": 1123,
            "seconds": 145.0,
            "exact": {"total": 746, "statuses": {"ok": 653}, "failures": 0},
            "generic": {"total": 103, "statuses": {"ok": 7}, "failures": 0},
            "quality": {"scored": 233, "correct": 232, "failures": 1},
        },
        "exact_failures": [],
        "generic_failures": [],
        "quality_failures": [failure],
    }
    comment = module.render_comment(report)
    assert "st-multilingual-e5-test" in comment
    assert "postgres" in comment and "celery" in comment
    assert "Генератор" in comment
    assert "27.90.40.150" in comment and "27.11.3" in comment
    assert len(comment) < 5000

    report["quality_failures"] = []
    report["summary"]["quality"] = {"scored": 233, "correct": 233, "failures": 0}
    assert "Расхождений не обнаружено" in module.render_comment(report)
