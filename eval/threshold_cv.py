"""Перекрёстная проверка порога `review`: реальная точка или подгонка под набор.

Зачем. `eval/run_eval.py --sweep` показывает сетку «порог → метрики» на всём
размеченном наборе. Соблазн взять лучшую клетку велик, но эта клетка выбрана на
тех же 274 строках, которыми потом меряем: часть выигрыша — шум конкретных строк.
Так, в отчёте видна точка `review = 0.655` с теми же top-1/top-3, что у 0.622, но
на 4 строки больше верной тишины (`eval/experiments.md`, эксп. 24 и 25).

Что делает инструмент. Делит набор на фолды и повторяет процедуру честно: порог
выбирается **на обучающих фолдах**, метрики снимаются **на отложенных**. Сравниваются
три стратегии:

* ``artifact`` — порог из артефакта (обучен на валидации из названий каталога,
  размеченный набор не видел вовсе);
* ``cv`` — лучший порог на обучающих фолдах (то, что мы получили бы, будя у нас
  только эти данные);
* ``grid`` — каждая клетка сетки, снятая на отложенных фолдах: это честный фронт,
  без всякого выбора.

Если ``cv`` не выигрывает у ``artifact``, значит подбирать порог на 274 строках
нечего: клетки, которые «лучше», — это шум, а не закономерность.

Почему порог вообще можно двигать пересчётом. Ранжирование (какой код первый,
какие альтернативы) от порога `review` не зависит — он решает только, ответить или
промолчать. Поэтому дорогое (ретрив + реранк) считается один раз, а метрики
пересчитываются для любой сетки мгновенно.

Запуск::

    EMBEDDER_BACKEND=tfidf VECTOR_SEARCH=true PYTHONPATH=api \
        python eval/threshold_cv.py [--folds 5] [--repeats 20] [--objective top1_all]
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from app.services.classifier import (  # noqa: E402
    _distinct_alternatives,
    _query_content_stems,
    build_classifier,
)

from eval.run_eval import LABELED, codes_match, label_kind, load_labeled  # noqa: E402
from ml.features import normalize  # noqa: E402

OBJECTIVES = {
    # Everything the product can observe about one row, counted the same way.
    "top1_all": lambda m: m["top1_all"],
    # Answer as much as possible without losing silence below the artifact's level.
    "top1_guarded": lambda m: m["top1_expected"] if m["silence_rate"] >= 90.0 else -1.0,
}


def collect(classifier, rows: list[dict]) -> list[dict]:
    """One row per labeled query: raw confidence, the answer, and both hint sets.

    The ranking does not depend on the review threshold, so it is captured once.
    ``answered_alts`` are the runner-ups shown when the row is answered;
    ``silent_alts`` are the anchored leads shown when it is not
    (``_distinct_alternatives`` with ``status="not_found"``).

    The answer is taken from the ranking, not from ``response.code``: the
    classifier already applied the *configured* floor, so ``response.code`` is
    ``None`` for every row below it and a sweep under that floor would silently
    re-measure the same point (this is how a 0.25 row once looked identical to
    0.45).
    """
    reranker = getattr(classifier, "_reranker", None)
    captured: dict[str, list] = {}
    captured_raw: dict[str, float] = {}
    original = reranker.evaluate

    def spy(query, ranked):
        captured[normalize(query)] = ranked
        result = original(query, ranked)
        captured_raw[normalize(query)] = float(result[0] or 0.0)
        return result

    reranker.evaluate = spy  # type: ignore[method-assign]
    try:
        collected = []
        for row in rows:
            if label_kind(row) == "unclear":
                continue  # contested ground truth: nothing to cross-validate
            query = (row["query"] or "").strip()
            expected = (row["expected_okpd2"] or "").strip()
            response = classifier.classify(query)
            ranked = captured[normalize(query)]
            stems = _query_content_stems(query)
            answered_alts = [
                alt.code
                for alt in _distinct_alternatives(ranked, "review", stems)
                if alt.code
            ]
            silent_alts = [
                alt.code
                for alt in _distinct_alternatives(ranked, "not_found", stems)
                if alt.code
            ]
            collected.append(
                {
                    "query": query,
                    "expected": expected,
                    "kind": label_kind(row),
                    "code": ranked[0].code if ranked else None,
                    # The raw score of the leading candidate, not the response's:
                    # a `not_found` answer zeroes `confidence_raw` by contract, so
                    # reading it would make every point below the configured floor
                    # look like "no answer at all".
                    "raw": captured_raw.get(normalize(query), response.confidence_raw),
                    "answered_alts": answered_alts,
                    "silent_alts": silent_alts,
                    "disputed": (row.get("disputed") or "0").strip() == "1",
                    # Состояние при пороге артефакта — для контроля сходимости с run_eval.
                    "baseline_status": response.status,
                }
            )
        return collected
    finally:
        reranker.evaluate = original  # type: ignore[method-assign]


def metrics_at(rows: list[dict], threshold: float) -> dict:
    """Metrics of the same rows at an arbitrary ``review`` threshold."""
    answered_rows = 0
    top1 = top3 = silence = 0
    expected_total = silence_total = 0
    for row in rows:
        if row.get("kind", "code") == "unclear":
            continue
        answered = row["code"] is not None and row["raw"] >= threshold
        answered_rows += answered
        if not row["expected"]:
            silence_total += 1
            silence += not answered
            continue
        expected_total += 1
        if answered:
            hit = codes_match(row["code"], row["expected"])
            top1 += hit
            top3 += hit or any(
                codes_match(code, row["expected"]) for code in row["answered_alts"]
            )
        else:
            top3 += any(
                codes_match(code, row["expected"]) for code in row["silent_alts"]
            )
    total = sum(1 for row in rows if row.get("kind", "code") != "unclear")
    return {
        "threshold": round(threshold, 3),
        "top1_expected": top1,
        "top3_expected": top3,
        "silence": silence,
        "silence_rate": 100.0 * silence / silence_total if silence_total else 0.0,
        "top1_all": top1 + silence,
        "answered": answered_rows,
        "expected_total": expected_total,
        "silence_total": silence_total,
        "total": total,
    }


def candidate_thresholds(rows: list[dict], own: float | None) -> list[float]:
    """Every distinct decision the rows can produce, plus the artifact's value."""
    values = sorted({round(row["raw"], 4) for row in rows} | {0.0})
    # One step above the strongest answer: "answer nothing" is a decision too, and
    # without this candidate the tool could never choose it.
    if values:
        values.append(round(values[-1] + 0.001, 4))
    return sorted(set(values) | ({round(own, 4)} if own is not None else set()))


def best_threshold(rows: list[dict], objective: str, own: float | None = None) -> float:
    """Threshold maximising ``objective`` on these rows; ties go to higher silence."""
    score = OBJECTIVES[objective]
    best: tuple[float, float, float] | None = None
    for threshold in candidate_thresholds(rows, own):
        metrics = metrics_at(rows, threshold)
        # Tie-break: more correct silence first, then the lower threshold (more answers).
        key = (score(metrics), metrics["silence"], -threshold)
        if best is None or key > best[0]:
            best = (key, threshold, metrics["silence"])
    return best[1] if best else 0.0


def stratified_folds(rows: list[dict], folds: int, rng: random.Random) -> list[list[int]]:
    """Split row indices into folds, keeping in-scope/out-of-scope proportions."""
    buckets: dict[bool, list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        if row.get("kind", "code") == "unclear":
            continue
        buckets[bool(row["expected"])].append(index)
    assignment = [[] for _ in range(folds)]
    for indices in buckets.values():
        rng.shuffle(indices)
        for position, index in enumerate(indices):
            assignment[position % folds].append(index)
    return assignment


def cross_validate(
    rows: list[dict], *, folds: int, repeats: int, objective: str, artifact: float
) -> dict:
    """Out-of-fold metrics of the three strategies."""
    per_strategy: dict[str, list[dict]] = defaultdict(list)
    chosen: list[float] = []
    rng = random.Random(1875)
    for _ in range(repeats):
        assignment = stratified_folds(rows, folds, rng)
        for held_out in assignment:
            held_out_set = set(held_out)
            train = [row for index, row in enumerate(rows) if index not in held_out_set]
            test = [rows[index] for index in held_out]
            selected = best_threshold(train, objective, artifact)
            chosen.append(selected)
            per_strategy["cv"].append(metrics_at(test, selected))
            per_strategy["artifact"].append(metrics_at(test, artifact))
    return {"per_strategy": per_strategy, "chosen": chosen}


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def summarize(metrics_list: list[dict]) -> dict[str, float]:
    return {
        "top1_expected": _mean([m["top1_expected"] for m in metrics_list]),
        "top3_expected": _mean([m["top3_expected"] for m in metrics_list]),
        "silence": _mean([m["silence"] for m in metrics_list]),
        "top1_all": _mean([m["top1_all"] for m in metrics_list]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--objective", choices=sorted(OBJECTIVES), default="top1_all")
    parser.add_argument("--labeled", type=Path, default=LABELED)
    parser.add_argument(
        "--backend",
        default="catalog",
        choices=("catalog", "qdrant"),
        help="бэкенд, на котором меряется порог (у семантики своя сырая шкала)",
    )
    args = parser.parse_args()

    rows = load_labeled(args.labeled)
    classifier = build_classifier(args.backend)
    reranker = getattr(classifier, "_reranker", None)
    artifact_threshold = float(reranker.confidence_model.threshold_review)
    info = getattr(classifier, "vector_info", None) or {}
    print(
        f"Скорит: {info.get('reranker') or '—'} · {info.get('scoring')} · "
        f"порог артефакта review={artifact_threshold:.3f}"
    )

    data = collect(classifier, rows)
    full_artifact = metrics_at(data, artifact_threshold)
    print(
        f"\nВесь набор при пороге артефакта: top-1 {full_artifact['top1_expected']}/"
        f"{full_artifact['expected_total']}, top-3 {full_artifact['top3_expected']}, "
        f"тишина {full_artifact['silence']}/{full_artifact['silence_total']}, "
        f"top-1 всего {full_artifact['top1_all']}"
    )

    print("\nСетка на всём наборе (как в отчёте — клетки выбраны на тех же строках):")
    print(f"  {'порог':>6} {'top-1':>7} {'top-3':>7} {'тишина':>8} {'всего':>7}")
    grid = candidate_thresholds(data, artifact_threshold)
    # Only the cells that change something are interesting; keep it readable.
    readable = sorted({round(value, 2) for value in grid} | {round(artifact_threshold, 2)})
    for threshold in readable:
        metrics = metrics_at(data, threshold)
        print(
            f"  {threshold:6.2f} {metrics['top1_expected']:7d} {metrics['top3_expected']:7d} "
            f"{metrics['silence']:8d} {metrics['top1_all']:7d}"
        )

    # Is the tempting cell special, or just one point of a plateau?
    scored = [(threshold, metrics_at(data, threshold)) for threshold in grid]
    best_all = max(metrics["top1_all"] for _, metrics in scored)
    plateau = [threshold for threshold, metrics in scored if metrics["top1_all"] == best_all]
    print(
        f"\nПлато максимума «top-1 всего» = {best_all}: пороги "
        f"{min(plateau):.2f}…{max(plateau):.2f} ({len(plateau)} точек). "
        "Соблазнительная клетка — одна из них, а не уникальная."
    )

    result = cross_validate(
        data,
        folds=args.folds,
        repeats=args.repeats,
        objective=args.objective,
        artifact=artifact_threshold,
    )
    print(
        f"\nКросс-валидация: {args.folds} фолдов × {args.repeats} повторов, "
        f"цель выбора — {args.objective}"
    )
    cv = summarize(result["per_strategy"]["cv"])
    artifact = summarize(result["per_strategy"]["artifact"])
    print(f"  {'стратегия':22s} {'top-1':>7} {'top-3':>7} {'тишина':>8} {'всего':>8}")
    for name, values in (("порог артефакта", artifact), ("порог, выбранный на фолдах", cv)):
        print(
            f"  {name:22s} {values['top1_expected']:7.1f} {values['top3_expected']:7.1f} "
            f"{values['silence']:8.1f} {values['top1_all']:8.1f}"
        )
    delta = {key: cv[key] - artifact[key] for key in cv}
    chosen = result["chosen"]
    print(
        f"\nВыигрыш выбора порога вне выборки (в строках на фолд ≈ {delta['top1_all']:+.2f}, "
        f"на весь набор ≈ {delta['top1_all'] * args.folds:+.1f}): "
        f"top-1 {delta['top1_expected']:+.1f}, тишина {delta['silence']:+.1f}"
    )
    matched = sum(1 for value in chosen if abs(value - artifact_threshold) < 1e-9)
    print(
        f"Выбранный порог: медиана {statistics.median(chosen):.3f}, "
        f"разброс {min(chosen):.3f}…{max(chosen):.3f}, "
        f"совпал с артефактом в {matched} из {len(chosen)} фолдов"
    )
    # One row per fold is the smallest change that is not rounding noise: folds hold
    # ~55 rows, so a tenth of a row cannot be a real improvement.
    if delta["top1_all"] < 1.0:
        print(
            "\nВывод: подбирать порог на размеченном наборе нечего. Плато на всём наборе "
            "широкое, но его выигрыш не переносится на отложенные строки — прирост "
            "меньше одной строки на фолд."
        )
    else:
        print(
            "\nВывод: выбор порога переносится на отложенные строки — точка из сетки "
            "не шум, её стоит предложить владельцу как настройку."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
