"""Decision rule for replacing the lexical reranker with a vector-feature one.

The rule answers one question: **may the artifact trained with dense features
serve decisions instead of the lexical artifact?** It is deliberately a pure
function over *counts*, so it can be tested without a model, a catalog or an
embedder, and so its verdict on a 274-row measurement set is a matter of
arithmetic rather than of comparing rounded percentages.

Why counts and not percentages
------------------------------
The labeled set has 166 rows with an expected code and 108 rows where silence is
the right answer, so one row is 0.6 / 0.9 p.p. The previous rule compared
*rounded rates* with ``new >= base`` and rejected a vector artifact that improved
top-1 by 1.8 p.p. (82.5 → 84.3), top-3 by 2.5 p.p. and calibration ECE by 0.018,
because ``ok_precision`` moved by one row (131/143 → 133/146). A strict
comparison of a rate computed over a model-chosen subset is noise, not a verdict.

Policy
------
* **Guards — not a single row may be lost** (``correct_silence``, ``exact_top1``):
  silence is the safety property of a legal assistant (a wrong code is worse than
  no code), and literal catalog names must never regress.
* **Headline — improvements are the reason to take the risk**: ``top1_expected``,
  ``top3_expected``, ``top1_all``. Each may lose at most one row (noise), and at
  least one must gain one row, otherwise there is nothing to trade the change for.
* **Rate over a model-chosen subset** (``ok_precision``): at most one more wrong
  confident answer than the lexical artifact. Counted as wrong ``ok`` answers,
  because that is what a user sees: "уверенный результат", which turned out wrong.
* **Calibration**: ECE may not grow by more than ``ECE_TOLERANCE``.

The rule is intentionally conservative: it never accepts an artifact that trades
away silence and never accepts one that is merely "not worse".
"""

from __future__ import annotations

from dataclasses import dataclass

ROW_TOLERANCE = 1
ECE_TOLERANCE = 0.01

# metric key -> human label
GUARD_METRICS: dict[str, str] = {
    "correct_silence": "верный not_found",
    "exact_top1": "точные названия, top-1",
}
HEADLINE_METRICS: dict[str, str] = {
    "top1_expected": "top-1 (с кодом)",
    "top3_expected": "top-3 (с кодом)",
    "top1_all": "top-1 по всем",
}
RATE_METRICS: dict[str, str] = {
    "ok_precision": "точность статуса ok",
}
METRIC_LABELS = {**GUARD_METRICS, **HEADLINE_METRICS, **RATE_METRICS}


@dataclass(frozen=True)
class Counts:
    """``correct`` out of ``total``; the only input the rule trusts."""

    correct: int
    total: int

    def __post_init__(self) -> None:
        if self.total < 0 or self.correct < 0 or self.correct > self.total:
            raise ValueError(f"Некорректные счётчики: {self.correct}/{self.total}")

    @property
    def wrong(self) -> int:
        return self.total - self.correct

    @property
    def rate(self) -> float:
        return 100.0 * self.correct / self.total if self.total else 0.0

    def as_dict(self) -> dict:
        return {"correct": self.correct, "total": self.total, "rate": round(self.rate, 1)}


@dataclass(frozen=True)
class GateRow:
    key: str
    label: str
    base: Counts
    new: Counts
    verdict: str  # better | equal | worse | tolerated | fatal

    @property
    def delta(self) -> int:
        return self.new.correct - self.base.correct

    @property
    def delta_wrong(self) -> int:
        """Shift in wrong answers — what the verdict uses for model-chosen subsets."""
        return self.new.wrong - self.base.wrong

    def as_dict(self) -> dict:
        return {
            "metric": self.key,
            "label": self.label,
            "base": self.base.as_dict(),
            "new": self.new.as_dict(),
            "delta_correct": self.delta,
            "delta_wrong": self.delta_wrong,
            "verdict": self.verdict,
        }


@dataclass(frozen=True)
class GateVerdict:
    keep: bool
    rows: tuple[GateRow, ...]
    reasons: tuple[str, ...]
    base_ece: float
    new_ece: float
    labels: tuple[str, str] = ("ПРИНЯТ", "ОТКЛОНЁН")
    columns: tuple[str, str] = ("лексика", "вектор")

    @property
    def headline_gains(self) -> dict[str, int]:
        return {
            row.key: row.delta
            for row in self.rows
            if row.key in HEADLINE_METRICS and row.delta > 0
        }

    def as_dict(self) -> dict:
        return {
            "keep": self.keep,
            "reasons": list(self.reasons),
            "ece": {"base": round(self.base_ece, 4), "new": round(self.new_ece, 4)},
            "rows": [row.as_dict() for row in self.rows],
        }

    def table(self) -> str:
        """Fixed-width comparison, readable in a build log without a markdown renderer."""
        left, right = self.columns
        lines = [
            f"  {'метрика':26s} {left:>22s} {right:>22s} {'Δ':>12s}  итог",
            "  " + "-" * 88,
        ]
        for row in self.rows:
            if row.key in RATE_METRICS:
                # Its denominator is chosen by the model itself, so the delta in
                # correct answers (e.g. -16) is not a regression at all: show the
                # thing the verdict is actually computed from.
                base = f"{row.base.rate:.1f}% ({row.base.wrong} неверн./{row.base.total})"
                new = f"{row.new.rate:.1f}% ({row.new.wrong} неверн./{row.new.total})"
                delta = f"{row.delta_wrong:+d} неверн."
            else:
                base = f"{row.base.rate:.1f}% ({row.base.correct}/{row.base.total})"
                new = f"{row.new.rate:.1f}% ({row.new.correct}/{row.new.total})"
                delta = f"{row.delta:+d}"
            lines.append(
                f"  {row.label:26s} {base:>22s} {new:>22s} {delta:>12s}  {row.verdict}"
            )
        lines.append(
            f"  {'ECE (калибровка)':26s} {self.base_ece:>22.3f} {self.new_ece:>22.3f}"
            f" {self.new_ece - self.base_ece:>12.3f}  "
            f"{'ok' if self.new_ece <= self.base_ece + ECE_TOLERANCE else 'fatal'}"
        )
        verdict = self.labels[0] if self.keep else self.labels[1]
        lines.append(f"  гейт: {verdict}" + ("" if self.keep else f" — {'; '.join(self.reasons)}"))
        return "\n".join(lines)


def evaluate_gate(
    base: dict[str, Counts],
    new: dict[str, Counts],
    *,
    base_ece: float,
    new_ece: float,
) -> GateVerdict:
    """Apply the policy above to two measurements of the same labeled set."""
    missing = (set(METRIC_LABELS) - set(base)) | (set(METRIC_LABELS) - set(new))
    if missing:
        raise ValueError(f"Нет измерений для метрик: {sorted(missing)}")
    # Guards and headline metrics are defined by the dataset, so their denominators
    # must match. ``ok_precision`` is counted over the answers the model chose to
    # mark as confident: a different artifact legitimately produces a different
    # number of them, which is why it is compared by wrong answers, not by rate.
    for key in GUARD_METRICS | HEADLINE_METRICS:
        if base[key].total != new[key].total:
            raise ValueError(f"Метрика {key} измерена на разных наборах: {base[key]} vs {new[key]}")

    rows: list[GateRow] = []
    reasons: list[str] = []

    for key, label in {**GUARD_METRICS, **HEADLINE_METRICS, **RATE_METRICS}.items():
        old, fresh = base[key], new[key]
        delta = fresh.correct - old.correct
        if key in GUARD_METRICS:
            verdict = "better" if delta > 0 else ("equal" if delta == 0 else "fatal")
            if verdict == "fatal":
                reasons.append(f"{label}: −{abs(delta)} верных строк")
        elif key in HEADLINE_METRICS:
            if delta < -ROW_TOLERANCE:
                verdict = "fatal"
                reasons.append(f"{label}: −{abs(delta)} строк (> допуска {ROW_TOLERANCE})")
            elif delta < 0:
                verdict = "tolerated"
            else:
                verdict = "better" if delta > 0 else "equal"
        else:  # ok_precision — compare wrong "confident" answers, not the rate
            extra_wrong = fresh.wrong - old.wrong
            if extra_wrong > ROW_TOLERANCE:
                verdict = "fatal"
                reasons.append(
                    f"{label}: неверных ok {old.wrong} → {fresh.wrong} "
                    f"(допуск +{ROW_TOLERANCE})"
                )
            elif extra_wrong > 0:
                verdict = "tolerated"
            else:
                verdict = "better" if delta > 0 or extra_wrong < 0 else "equal"
        rows.append(GateRow(key, label, old, fresh, verdict))

    if new_ece > base_ece + ECE_TOLERANCE:
        reasons.append(f"ECE вырос {base_ece:.3f} → {new_ece:.3f} (допуск +{ECE_TOLERANCE})")

    if not any(row.key in HEADLINE_METRICS and row.delta > 0 for row in rows):
        reasons.append("ни одна из ключевых метрик не улучшилась — менять нечего")

    return GateVerdict(
        keep=not reasons,
        rows=tuple(rows),
        reasons=tuple(reasons),
        base_ece=base_ece,
        new_ece=new_ece,
    )
