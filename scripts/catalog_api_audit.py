#!/usr/bin/env python3
"""Exhaustive catalog audit through a running API (including production E5).

Run inside the API container after ``docker_smoke.sh`` is healthy. The script
uses only the public ``/classify`` contract, so lexical, TF-IDF and transformer
backends are measured through exactly the same serving path. It writes every
answer for review and fails on any scored discrepancy; generic heads have an
explicit reviewed doctrine rather than accepting every clarification as success.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ml.features import stems, tokens  # noqa: E402

GENERIC_OK = {"ткани", "одежда", "обувь", "лифты", "трубы", "видеокамеры", "вина"}
GENERIC_NOT_FOUND = {"медицинские"}
GENERIC_STOP = {
    "для",
    "и",
    "прочие",
    "прочий",
    "прочая",
    "не",
    "в",
    "с",
    "из",
    "по",
    "на",
    "их",
    "или",
    "без",
    "всех",
    "видов",
    "соответствующие",
    "соответствующий",
    "соответствующая",
    "том",
    "числе",
    "назначения",
    "общего",
}


def norm(text: str) -> str:
    return " ".join(text.casefold().split())


def codes_match(predicted: str | None, expected: str) -> bool:
    if not predicted or not expected:
        return False
    return (
        predicted == expected
        or predicted.startswith(expected + ".")
        or expected.startswith(predicted + ".")
    )


def catalog_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter=";"))


def exact_queries(rows: list[dict]) -> list[tuple[str, set[str]]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if row.get("name"):
            grouped[norm(row["name"])].append(row)
    return [
        (group[0]["name"], {row["okpd2"].strip() for row in group if row.get("okpd2")})
        for group in grouped.values()
    ]


def generic_queries(rows: list[dict]) -> list[str]:
    grouped: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for row in rows:
        words = tokens(row.get("name") or "")
        if words and words[0] not in GENERIC_STOP and len(words[0]) >= 4:
            grouped[stems(words[0])[0]].append((words[0], row))
    queries = []
    for items in grouped.values():
        codes = {row["okpd2"] for _, row in items if row.get("okpd2")}
        names = {row["name"] for _, row in items}
        if len(codes) > 1 and len(names) > 1:
            queries.append(Counter(word for word, _ in items).most_common(1)[0][0])
    return queries


class ApiAudit:
    def __init__(self, base_url: str, requests_per_second: float = 8.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.min_interval = 1.0 / requests_per_second
        self.last_request = 0.0
        self.calls = 0

    def request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode()
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        for attempt in range(7):
            elapsed = time.monotonic() - self.last_request
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_request = time.monotonic()
            try:
                with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code != 429 or attempt == 6:
                    raise
                retry_after = float(error.headers.get("Retry-After", "1"))
                time.sleep(max(retry_after, self.min_interval))
        raise AssertionError("unreachable")

    def classify(self, query: str) -> dict:
        self.calls += 1
        if self.calls % 50 == 0:
            print(f"  classified {self.calls}", flush=True)
        try:
            answer = self.request("/classify", {"name": query})
        except urllib.error.HTTPError as error:
            if error.code != 422 or len(query) <= 500:
                raise
            # ProductName deliberately caps public input at 500 characters.
            # Three composite medical titles are longer; record the validation
            # contract explicitly instead of hiding them or weakening the cap.
            answer = {"status": "validation_error", "code": None, "title": None}
        return {
            "query": query,
            "status": answer["status"],
            "code": answer.get("code"),
            "title": answer.get("title"),
            "options": [
                {
                    "option_id": option["option_id"],
                    "code": option.get("code"),
                    "title": option.get("title"),
                }
                for option in (answer.get("clarification") or {}).get("options", [])
            ],
        }


def check_exact(answer: dict, expected_codes: set[str]) -> str | None:
    if len(answer.get("query", "")) > 500 and answer["status"] == "validation_error":
        return None
    if not expected_codes:
        if answer["status"] == "not_found":
            return None
        return "ожидался not_found для бескодовой строки"
    option_codes = {option["code"] for option in answer["options"] if option.get("code")}
    if len(expected_codes) > 1:
        if answer["status"] != "needs_clarification":
            return f"ожидалось уточнение кодов {sorted(expected_codes)}"
        if not expected_codes <= option_codes:
            return f"не все sibling-коды: нет {sorted(expected_codes - option_codes)}"
        return None
    expected = next(iter(expected_codes))
    if answer["status"] != "ok" or answer["code"] != expected:
        return f"ожидался ok {expected}"
    return None


def check_generic(answer: dict) -> str | None:
    query = norm(answer["query"])
    if query in GENERIC_OK:
        expected = "ok"
    elif query in GENERIC_NOT_FOUND:
        expected = "not_found"
    else:
        expected = "needs_clarification"
    if answer["status"] != expected:
        return f"ожидался {expected}"
    if expected == "needs_clarification" and len(answer["options"]) < 2:
        return "уточнение содержит меньше двух options"
    return None


def check_quality(answer: dict, row: dict) -> bool | None:
    kind = row.get("label_kind") or ("code" if row.get("expected_okpd2") else "absent")
    if kind == "unclear":
        return None
    if kind == "absent":
        return answer["status"] == "not_found"
    expected = row["expected_okpd2"].strip()
    if codes_match(answer.get("code"), expected):
        return True
    return any(codes_match(option.get("code"), expected) for option in answer["options"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--requests-per-second", type=float, default=8.0)
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "data.csv")
    parser.add_argument("--labeled", type=Path, default=ROOT / "eval" / "labeled_examples.csv")
    parser.add_argument("--output", type=Path, default=Path("/tmp/catalog-production-audit.json"))
    parser.add_argument("--require-e5", action="store_true")
    args = parser.parse_args()

    if args.requests_per_second <= 0:
        parser.error("--requests-per-second должен быть положительным")
    api = ApiAudit(args.base_url, args.requests_per_second)
    readiness = api.request("/health/ready")
    if args.require_e5:
        vector = readiness.get("vector_search") or {}
        embedder = vector.get("embedder") or ""
        if not vector.get("enabled") or "e5" not in embedder.casefold():
            raise SystemExit(f"Ожидался production E5, получен {vector}")
        if vector.get("store") != "qdrant":
            raise SystemExit(f"Ожидался Qdrant, получен {vector}")
        if readiness.get("job_store") != "postgres" or readiness.get("queue_backend") != "celery":
            raise SystemExit(f"Ожидались PostgreSQL/Celery, получен {readiness}")

    rows = catalog_rows(args.catalog)
    with args.labeled.open(encoding="utf-8", newline="") as handle:
        labeled = list(csv.DictReader(handle, delimiter=";"))
    started = time.monotonic()
    exact = []
    exact_failures = []
    for query, expected_codes in exact_queries(rows):
        answer = api.classify(query)
        problem = check_exact(answer, expected_codes)
        record = {**answer, "expected_codes": sorted(expected_codes), "problem": problem}
        exact.append(record)
        if problem:
            exact_failures.append(record)

    generic = []
    generic_failures = []
    for query in generic_queries(rows):
        answer = api.classify(query)
        problem = check_generic(answer)
        record = {**answer, "problem": problem}
        generic.append(record)
        if problem:
            generic_failures.append(record)

    quality = []
    quality_failures = []
    for row in labeled:
        answer = api.classify(row["query"])
        correct = check_quality(answer, row)
        record = {
            **answer,
            "expected": row.get("expected_okpd2") or "(not_found)",
            "label_kind": row.get("label_kind"),
            "correct": correct,
        }
        quality.append(record)
        if correct is False:
            quality_failures.append(record)

    scored_quality = [row for row in quality if row["correct"] is not None]
    summary = {
        "readiness": readiness,
        "calls": api.calls,
        "seconds": round(time.monotonic() - started, 3),
        "exact": {
            "total": len(exact),
            "statuses": dict(Counter(row["status"] for row in exact)),
            "validation_only": sum(row["status"] == "validation_error" for row in exact),
            "failures": len(exact_failures),
        },
        "generic": {
            "total": len(generic),
            "statuses": dict(Counter(row["status"] for row in generic)),
            "failures": len(generic_failures),
        },
        "quality": {
            "scored": len(scored_quality),
            "correct": sum(row["correct"] is True for row in scored_quality),
            "failures": len(quality_failures),
        },
    }
    args.output.write_text(
        json.dumps(
            {
                "summary": summary,
                "exact_failures": exact_failures,
                "generic_failures": generic_failures,
                "quality_failures": quality_failures,
                "exact": exact,
                "generic": generic,
                "quality": quality,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Audit JSON: {args.output}")
    return int(bool(exact_failures or generic_failures or quality_failures))


if __name__ == "__main__":
    raise SystemExit(main())
