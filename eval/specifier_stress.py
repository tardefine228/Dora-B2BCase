"""Stress measurement: a realistic purchase line around a catalog position.

Why this exists. The labeled set (`eval/labeled_examples.csv`) is human-curated
and small: 201 rows with a code. A change that helps real ERP lines can move it
by one row either way, and a change that hurts it is invisible until it is large.
The fifth lexicon wave (эксп. 34) showed the asymmetry: the lines that fail in
production are not "краткие названия", they are names plus attributes nobody
could have listed — "Осциллограф цифровой запоминающий 100 МГц; 2 канала".

This tool therefore *synthesizes* such lines from the catalog itself, with the
true position known by construction:

    <2-4 words of the position name> + <1-3 attribute strings>

Three shapes are generated per sampled row, all deterministic (seed 1875):

* ``name`` — the words alone, the control: what the labels already measure;
* ``plain`` — words followed by attributes, as typed in a request;
* ``erp`` — ``name; attribute; attribute``, as exported by 1C/ERP.

Reported per shape: top-1 (the answer carries the expected code), top-3 (the
code is the answer or one of the alternatives), silence, and ``silent_hint`` —
silent while the expected position *is* among the shown hints. The last number
is the one the weak-lead rescue (эксп. 34) exists for: the rank was right, the
answer was empty.

The set is a **measurement**, not a training set: `ml/train_reranker.py` never
reads it, and a change is kept only if it improves this set *and* does not
regress the labeled one (`eval/experiments.md`, эксп. 35).

Usage::

    PYTHONPATH=api EMBEDDER_BACKEND=tfidf VECTOR_SEARCH=false \
        python eval/specifier_stress.py --limit 300
    PYTHONPATH=api python eval/specifier_stress.py --check eval/baselines/specifier_stress.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from eval.run_eval import codes_match  # noqa: E402
from ml.catalog_loader import load_catalog  # noqa: E402
from ml.specifiers import SPEC_EXAMPLES  # noqa: E402
from ml.train_reranker import GENERIC_WORDS  # noqa: E402

SEED = 1875
# Bases (what the user types) and shapes (base alone / base + attributes). The
# control pairs (``name``, ``short``, ``bridge``) say how well the position is
# found at all; ``+attr`` is the same query with 1-3 technical attributes, and
# the difference is the cost of the specification.
BASES = ("name", "short", "bridge")
SHAPES = tuple(base for base in BASES) + tuple(f"{base}+attr" for base in BASES)
ATTR_RATE = 2  # attributes added per line (rng.randint(1, ATTR_RATE))


def _content_words(name: str) -> list[str]:
    """Product words of a position name, in the order the name writes them."""
    words = [word for word in name.replace(",", " ").split() if word]
    product = [
        word for word in words if word.lower() not in GENERIC_WORDS and not word.isdigit()
    ]
    return product or words


def _add_attributes(text: str, specs: list[str]) -> str:
    """A line as a specialist types it: the name, then its attributes."""
    return " ".join([text, *specs])


def build_cases(limit: int | None = None) -> list[dict]:
    """Deterministic synthetic lines: three bases × (alone, with attributes).

    * ``name`` — the position name exactly as the catalog writes it: the control
      that the labels already measure (a literal name is answered ~100 %);
    * ``short`` — the first 3-4 product words of that name, i.e. how a specialist
      who does not copy the list would type it;
    * ``bridge`` — the everyday product word the lexicon bridges to this position
      ("осциллограф" → «Осциллоскопы и осциллографы…»), the shape the fifth wave
      was about.

    The truth is the **set** of codes of that position: 87 of 746 names are
    repeated in several appendices with different codes, and any of them is a
    correct answer.
    """
    rng = random.Random(SEED)
    catalog = load_catalog()
    rows = list(catalog)
    if limit is not None and limit < len(rows):
        rows = rng.sample(rows, limit)
    else:
        rng.shuffle(rows)

    codes_by_name: dict[str, list[str]] = {}
    for item in catalog:
        title = (item["name"] or "").strip()
        code = (item.get("pp_code") or "").strip()
        if title and code and code not in codes_by_name.setdefault(title, []):
            codes_by_name[title].append(code)

    from ml.train_reranker import bridge_sources_by_name

    sources_by_name = bridge_sources_by_name(catalog)
    cases: list[dict] = []

    def add(base: str, text: str, codes: list[str], name: str) -> None:
        specs = rng.sample(SPEC_EXAMPLES, rng.randint(1, ATTR_RATE))
        cases.append({"query": text, "expected": codes, "shape": base, "row": name})
        cases.append(
            {
                "query": _add_attributes(text, specs),
                "expected": codes,
                "shape": f"{base}+attr",
                "row": name,
            }
        )

    for item in rows:
        name = (item["name"] or "").strip()
        if not name or not (item.get("pp_code") or "").strip():
            continue
        codes = codes_by_name[name]
        add("name", name.rstrip(";").strip(), codes, name)

        words = _content_words(name)
        keep = min(len(words), rng.randint(3, 4))
        short = " ".join(words[:keep]).rstrip(";").strip()
        if short and short.lower() != name.lower():
            add("short", short, codes, name)

        sources = sources_by_name.get(name) or []
        if sources:
            add("bridge", rng.choice(sources), codes, name)

    # Bridge rows are rare in a sample, but they are the shape that matters most
    # for the lexicon waves: always take every canonical bridged position.
    sampled_names = {case["row"] for case in cases}
    for name, sources in sources_by_name.items():
        if name in sampled_names or not name:
            continue
        codes = codes_by_name.get(name)
        if not codes or not sources:
            continue
        add("bridge", rng.choice(sources), codes, name)
        sampled_names.add(name)
    return cases


def _matches(predicted: str | None, expected: list[str]) -> bool:
    """Whether a code names the same position as any code of the expected row."""
    return any(codes_match(predicted, code) or codes_match(code, predicted) for code in expected)


def measure(cases: list[dict], classifier) -> dict:
    """Run the classifier over the cases and count the four outcomes per shape."""
    stats: dict[str, dict[str, int]] = {
        shape: {"rows": 0, "top1": 0, "top3": 0, "silent": 0, "silent_hint": 0}
        for shape in SHAPES
    }
    failures: list[dict] = []
    for case in cases:
        shape = case["shape"]
        expected = case["expected"]
        response = classifier.classify(case["query"])
        row = stats[shape]
        row["rows"] += 1
        answered = response.status in {"ok", "review"} and bool(response.code)
        hint_codes = [alt.code for alt in (response.alternatives or []) if alt.code]
        top1 = answered and _matches(response.code, expected)
        top3 = top1 or any(_matches(code, expected) for code in hint_codes)
        if top1:
            row["top1"] += 1
        if top3:
            row["top3"] += 1
        if not answered:
            row["silent"] += 1
            if any(_matches(code, expected) for code in hint_codes):
                row["silent_hint"] += 1
            if len(failures) < 40:
                failures.append(
                    {
                        "query": case["query"],
                        "shape": shape,
                        "expected": expected[0],
                        "expected_name": case["row"],
                        "status": response.status,
                        "hints": hint_codes[:3],
                    }
                )
        elif not top1 and len(failures) < 40:
            failures.append(
                {
                    "query": case["query"],
                    "shape": shape,
                    "expected": expected[0],
                    "expected_name": case["row"],
                    "status": response.status,
                    "got": response.code,
                    "got_name": response.title,
                    "hints": hint_codes[:3],
                }
            )
    return {"stats": stats, "failures": failures}


def _rate(row: dict, key: str) -> float:
    return round(100.0 * row[key] / row["rows"], 1) if row["rows"] else 0.0


def summarize(result: dict) -> dict:
    stats = result["stats"]
    return {
        shape: {
            "rows": row["rows"],
            "top1": _rate(row, "top1"),
            "top3": _rate(row, "top3"),
            "silence": _rate(row, "silent"),
            "silent_hint": _rate(row, "silent_hint"),
            "top1_correct": row["top1"],
            "silent_correct": row["silent"],
            "silent_hint_correct": row["silent_hint"],
        }
        for shape, row in stats.items()
    }


def _config() -> dict:
    """Provenance: which artifact and catalog the numbers describe."""
    import hashlib

    from ml import config as ml_config

    artifact = ml_config.ARTIFACTS_DIR / "reranker.json"
    try:
        sha = hashlib.sha256(artifact.read_bytes()).hexdigest()
    except OSError:  # pragma: no cover - only when the artifact is missing
        sha = None
    return {
        "reranker": artifact.name,
        "artifact_sha256": sha,
        "catalog_rows": len(load_catalog()),
        "labeled_rows": 274,
    }


def _print(summary: dict, detail: bool) -> None:
    header = (
        f"{'форма':6} {'строк':>6} {'top-1':>7} {'top-3':>7} {'тишина':>7} "
        f"{'молчит, но ранг 1':>18}"
    )
    print(header)
    for shape in SHAPES:
        row = summary[shape]
        print(
            f"{shape:6} {row['rows']:6d} {row['top1']:6.1f}% {row['top3']:6.1f}% "
            f"{row['silence']:6.1f}% {row['silent_hint']:6.1f}%"
        )
    if detail:
        print()
        for shape in SHAPES:
            row = summary[shape]
            print(
                f"{shape}: top-1 {row['top1_correct']}/{row['rows']}, "
                f"тишина {row['silent_correct']}, "
                f"из них ранг 1 верный {row['silent_hint_correct']}"
            )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=300, help="сколько позиций каталога взять")
    parser.add_argument("--json", type=Path, help="куда записать замер")
    parser.add_argument("--detail", action="store_true", help="печатать счётчики, а не проценты")
    parser.add_argument(
        "--check",
        type=Path,
        help="сравнить с записанным замером и упасть при падении больше чем на допуск",
    )
    parser.add_argument(
        "--failures", type=int, default=0, help="сколько разобранных строк напечатать"
    )
    args = parser.parse_args(argv)

    from app.services.classifier import build_classifier

    cases = build_cases(args.limit)
    result = measure(cases, build_classifier("catalog"))
    summary = summarize(result)
    _print(summary, args.detail)

    for failure in result["failures"][: args.failures]:
        print(
            f"  {failure['shape']:5} {failure['status']:9} {failure.get('got') or '-':14} "
            f"ждём {failure['expected']:14} {failure['query'][:78]}"
        )

    payload = {
        "seed": SEED,
        "limit": args.limit,
        "config": _config(),
        "shapes": summary,
    }
    if args.json:
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"Замер: {args.json}")

    if args.check:
        baseline = json.loads(args.check.read_text(encoding="utf-8"))["shapes"]
        return _compare(summary, baseline)
    return 0


# A shape may lose this many points before the run is a regression. The set is
# deterministic, so the number is not noise; the tolerance only absorbs a
# deliberately changed generator or catalog.
TOLERANCE = 0.5


def _compare(current: dict, baseline: dict) -> int:
    problems = []
    for shape in SHAPES:
        for key in ("top1", "top3", "silence"):
            delta = round(current[shape][key] - baseline[shape][key], 1)
            # Silence may not grow, the other two may not fall.
            if (key == "silence" and delta > TOLERANCE) or (
                key != "silence" and delta < -TOLERANCE
            ):
                problems.append(
                    f"{shape}.{key}: было {baseline[shape][key]:.1f}, "
                    f"стало {current[shape][key]:.1f}"
                )
    if problems:
        print("РЕГРЕССИЯ на стресс-наборе:")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("Стресс-набор: не хуже записанного замера")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
