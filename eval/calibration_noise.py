"""Is an ECE difference a finding or a coin flip? (ML-debt)

The vector artifact and the lexical one are measured on the same 274-row labeled
set, of which ~200 answers are answered and not contested. ECE over that set is
quantised: five bins, one row moves |accuracy − confidence| by ~0.005, and the
same *code* trained with a different seed moves ECE between 0.012 and 0.038
(``eval/experiments.md``, эксп. 36; ``dense-gate.json`` keeps the per-run values
— ``ece_per_run``).

That is why a single build's ECE cannot rank two artifacts: the difference
between 0.012 and 0.031 is smaller than the spread of either family across
seeds. This tool answers the only question worth asking about such a difference:

    **is it larger than the resampling noise of this very set?**

It bootstraps the answered rows (paired: the same resampled rows are scored by
both configurations), reports the 95 % interval of each ECE and of the
difference, and prints the share of resamples in which the candidate is better.
The gate pools reliability tables over training runs for the same reason
(``eval/dense_gate.pool_bins``); this tool does the complementary job for a
single build — it says whether one number may be read at all.

Usage::

    PYTHONPATH=api python eval/calibration_noise.py --samples 2000
    PYTHONPATH=api python eval/calibration_noise.py --output eval/calibration_noise.md

Each configuration runs in its own process (the backend and the embedder are
read at import time), so the numbers come from the same code the service uses.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "api"))

from eval.dense_gate import ece_from_bins, reliability_bins  # noqa: E402

LABELED = ROOT / "eval" / "labeled_examples.csv"
DEFAULT_SAMPLES = 2000
DEFAULT_SEED = 1875


def answered_pairs() -> dict[str, tuple[float, bool]]:
    """``query -> (confidence, correct_top1)`` for answered, uncontested rows.

    Exactly the rows ``run_eval.calibration`` scores: silence is excluded (its
    correctness is another question) and so are contested labels. Keyed by the
    query, because two configurations may answer a different number of rows and
    a paired comparison must score the same queries.
    """
    from app.services.classifier import build_classifier

    from eval.run_eval import _classify_rows, _row_kind, load_labeled

    classifier = build_classifier()
    results = _classify_rows(classifier, load_labeled(LABELED))
    return {
        row["query"]: (float(row["confidence"]), bool(row["correct_top1"]))
        for row in results
        if row["status"] != "not_found" and _row_kind(row) != "unclear"
    }


def ece(pairs: list[tuple[float, bool]]) -> float:
    """ECE as the gate computes it: |accuracy − confidence| over 5 bins."""
    return ece_from_bins(reliability_bins(pairs))


def _resample(
    pairs: list[tuple[float, bool]], rng: random.Random
) -> list[tuple[float, bool]]:
    return [pairs[rng.randrange(len(pairs))] for _ in range(len(pairs))]


def bootstrap(
    base: list[tuple[float, bool]],
    candidate: list[tuple[float, bool]],
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict:
    """Paired bootstrap of two ECEs measured on the same rows.

    ``base`` and ``candidate`` are row-aligned: index *i* of both lists is the
    same query answered by the two configurations. One resample of indices
    scores both, so the difference is not inflated by two independent draws.
    """
    if len(base) != len(candidate):
        raise ValueError("Конфигурации измерены на разных наборах строк")
    rng = random.Random(seed)
    base_values: list[float] = []
    candidate_values: list[float] = []
    differences: list[float] = []
    for _ in range(samples):
        indices = [rng.randrange(len(base)) for _ in range(len(base))]
        left = ece([base[index] for index in indices])
        right = ece([candidate[index] for index in indices])
        base_values.append(left)
        candidate_values.append(right)
        differences.append(right - left)
    return {
        "rows": len(base),
        # Point estimate on the full set; the intervals come from the resamples
        # (ECE is convex in the sample, so the *mean* of resampled ECEs sits above
        # the estimate and must not be printed as one).
        "base": ece(base),
        "candidate": ece(candidate),
        "base_resampled_mean": statistics.fmean(base_values),
        "candidate_resampled_mean": statistics.fmean(candidate_values),
        "base_interval": _interval(base_values),
        "candidate_interval": _interval(candidate_values),
        "difference_interval": _interval(differences),
        "candidate_better_share": sum(1 for value in differences if value < 0)
        / len(differences),
        "samples": samples,
    }


def _interval(values: list[float]) -> tuple[float, float]:
    ordered = sorted(values)
    low = ordered[int(0.025 * (len(ordered) - 1))]
    high = ordered[int(0.975 * (len(ordered) - 1))]
    return round(low, 4), round(high, 4)


def _run_mode(config: str) -> dict[str, tuple[float, bool]]:
    """Measure one configuration in its own process (env is read at import)."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "api"), os.environ.get("PYTHONPATH", "")]
    )
    if config == "lexical":
        env.update(VECTOR_SEARCH="false", EMBEDDER_BACKEND="tfidf")
    elif config == "dense":
        env.update(VECTOR_SEARCH="true", EMBEDDER_BACKEND="tfidf")
    else:
        raise SystemExit(f"неизвестная конфигурация: {config}")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--pairs"],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )
    if result.returncode != 0:
        raise SystemExit(f"конфигурация {config} упала:\n{result.stderr}")
    return {
        query: (float(value), bool(hit))
        for query, (value, hit) in json.loads(result.stdout).items()
    }


def _contributions(pairs: list[tuple[float, bool]]) -> dict[tuple[float, float], dict]:
    """Per-bin share of ECE, so a difference can be attributed to a band."""
    table = reliability_bins(pairs)
    total = sum(row.count for row in table) or 1
    return {
        (row.low, row.high): {
            "count": row.count,
            "confidence": row.confidence,
            "accuracy": row.accuracy,
            "contribution": row.count / total * abs(row.accuracy - row.confidence),
        }
        for row in table
        if row.count
    }


def _dominant_difference(
    base: list[tuple[float, bool]], candidate: list[tuple[float, bool]]
) -> str | None:
    """Band that explains most of the ECE gap, in words."""
    left, right = _contributions(base), _contributions(candidate)
    bands = set(left) | set(right)
    if not bands:
        return None
    band = max(
        bands,
        key=lambda key: abs(
            right.get(key, {}).get("contribution", 0.0)
            - left.get(key, {}).get("contribution", 0.0)
        ),
    )
    old, new = left.get(band), right.get(band)
    if not old or not new:
        return (
            f"разница объясняется строками уверенности {band[0]:.1f}–{band[1]:.1f}, "
            f"которые отвечает только одна из конфигураций"
        )
    direction = _direction(new)
    return (
        f"почти вся разница — в интервале {band[0]:.1f}–{band[1]:.1f}: "
        f"{old['count']} строк против {new['count']}, точность "
        f"{old['accuracy']:.3f} против {new['accuracy']:.3f}, средняя уверенность "
        f"{old['confidence']:.3f} против {new['confidence']:.3f} — у вектора это "
        f"{direction}"
    )


def _direction(bin_stats: dict) -> str:
    """Name the sign of a miscalibration, because the two signs are not equal."""
    gap = bin_stats["confidence"] - bin_stats["accuracy"]
    if abs(gap) < 0.02:
        return "почти точное попадание"
    return "переуверенность (обещает больше, чем даёт)" if gap > 0 else (
        "недоуверенность (обещает меньше, чем даёт)"
    )


def _reliability_table(pairs: list[tuple[float, bool]]) -> list[str]:
    rows = [
        "| Интервал уверенности | Строк | Средняя уверенность | Точность |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in reliability_bins(pairs):
        if not row.count:
            continue
        rows.append(
            f"| {row.low:.1f} – {row.high:.1f} | {row.count} | "
            f"{row.confidence:.3f} | {row.accuracy:.3f} |"
        )
    return rows


def _markdown(
    stats: dict,
    base: list[tuple[float, bool]],
    candidate: list[tuple[float, bool]],
) -> str:
    dominant = _dominant_difference(base, candidate)
    low, high = stats["base_interval"]
    cand_low, cand_high = stats["candidate_interval"]
    diff_low, diff_high = stats["difference_interval"]
    share = stats["candidate_better_share"]
    if diff_high < 0:
        verdict = "вектор калиброван лучше, и это больше шума пересэмплинга"
    elif diff_low > 0:
        verdict = "вектор калиброван хуже, и это больше шума пересэмплинга"
    elif abs(share - 0.5) < 0.15:
        verdict = "различие неотличимо от шума пересэмплинга"
    else:
        verdict = (
            f"интервал разницы включает ноль, но пересэмплинг смещён "
            f"({share * 100:.0f} % против {100 - share * 100:.0f} %): на 200 строках различие "
            f"не доказуемо, для решения нужен больший набор или пул прогонов"
        )
    return "\n".join(
        [
            "# Шум ECE на размеченном наборе (ML-долг)",
            "",
            f"Отвеченных uncontested строк в обеих конфигурациях: **{stats['rows']}**; "
            f"только у лексики: {stats['only_base']}, только у вектора: "
            f"{stats['only_candidate']} (в парный интервал не входят).",
            f"Бутстрэп: {stats['samples']} пересэмплингов, парный по строкам.",
            "",
            "| Конфигурация | ECE | ECE, 95 % интервал |",
            "| --- | ---: | --- |",
            f"| лексика | {stats['base']:.4f} | {low:.4f} … {high:.4f} |",
            f"| вектор | {stats['candidate']:.4f} | {cand_low:.4f} … {cand_high:.4f} |",
            "",
            f"Разница (вектор − лексика): **{stats['candidate'] - stats['base']:+.4f}**, "
            f"95 % интервал **{diff_low:+.4f} … {diff_high:+.4f}**.",
            "",
            f"Доля пересэмплингов, где вектор ближе к калибровке: "
            f"{stats['candidate_better_share'] * 100:.1f} %.",
            "",
            f"**Вывод:** {verdict}.",
            "",
            *([f"Где именно: {dominant}."] if dominant else []),
            "",
            "Таблицы надёжности, по которым считается ECE:",
            "",
            "**Лексика**",
            "",
            *_reliability_table(base),
            "",
            "**Вектор**",
            "",
            *_reliability_table(candidate),
            "",
            "Переуверенность опасна: она превращает подсказку в обещание. Недоуверенность",
            "безопасна: статус `review` и так значит «проверьте этот пункт». Поэтому знак",
            "разницы важен не меньше её модуля.",
            "",
            "Это не оправдание допуска, а граница разрешения набора: один и тот же код",
            "с другим сидом обучения даёт ECE от 0.012 до 0.038 (эксп. 36, `ece_per_run`",
            "в `dense-gate.json`). Гейт сравнивает не два одиночных ECE, а таблицы",
            "надёжности, объединённые по прогонам (`eval/dense_gate.pool_bins`).",
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", action="store_true", help="вывести пары (внутреннее)")
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path, default=None, help="записать markdown-отчёт")
    args = parser.parse_args()

    if args.pairs:
        print(json.dumps(answered_pairs(), ensure_ascii=False))
        return 0

    measured = {"lexical": _run_mode("lexical"), "dense": _run_mode("dense")}
    common = sorted(set(measured["lexical"]) & set(measured["dense"]))
    base = [measured["lexical"][query] for query in common]
    candidate = [measured["dense"][query] for query in common]
    only_base = len(measured["lexical"]) - len(common)
    only_candidate = len(measured["dense"]) - len(common)
    stats = bootstrap(base, candidate, samples=args.samples, seed=args.seed)
    stats["only_base"] = only_base
    stats["only_candidate"] = only_candidate
    report = _markdown(stats, base, candidate)
    print(report)
    if args.output is not None:
        args.output.write_text(report, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
