"""Reproducible offline experiment. Never installs the candidate automatically.

Training pairs are synthetic derivatives of the catalog, NOT procurement data.
The existing labeled evaluation set is excluded from training and calibration
by normalized query. It is a regression benchmark, not a blind external test.
"""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "api")]
from app.services.classifier import CatalogClassifier, _tokenize
from ml.features import normalize
from ml.reranker import LogisticReranker
from ml.train_reranker import train
from ml.dense import DenseIndex
from eval.run_eval import (_classify_rows, load_labeled, LABELED, summarize,
                           evaluate_exact_names, load_exact_names, codes_match)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dense", action="store_true")
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    catalog = ROOT / "data/data.csv"
    labeled = load_labeled(LABELED)
    excluded = {normalize(row["query"]) for row in labeled}
    baseline = LogisticReranker.load(ROOT / "ml/artifacts/reranker.json")
    base = CatalogClassifier(catalog, reranker=baseline, dense=None)
    dense = DenseIndex.build([row["name"] for row in base.index])
    dense.warm_queries([row["query"] for row in labeled])
    base._dense = dense
    exact_names = load_exact_names(catalog)
    def measure(classifier):
        results = _classify_rows(classifier, labeled)
        exact = evaluate_exact_names(classifier, exact_names)
        summary = summarize({"results": results, "exact": exact})
        confident = [row for row in results if row["status"] == "ok"]
        summary["ok_precision"] = sum(r["correct_top1"] for r in confident)/len(confident) if confident else 1.0
        return {"summary": summary, "results": results, "exact": exact}
    before = measure(base)
    (out / "baseline.json").write_text(json.dumps(before, ensure_ascii=False, indent=2), encoding="utf-8")
    hits = {"lexical20": 0, "dense20": 0, "union20": 0}
    count = 0
    misses = []
    for row in labeled:
        if not row["expected_okpd2"]: continue
        count += 1
        q = normalize(row["query"])
        lex = [r for _,r in base._retrieve(q, _tokenize(q))][:20]
        semantic = [base.index[i] for i,_ in dense.top(q,20)]
        good = lambda rows: any(codes_match(r["okpd2"], row["expected_okpd2"]) for r in rows)
        hits["lexical20"] += good(lex)
        hits["dense20"] += good(semantic)
        hits["union20"] += good(lex+semantic)
        if not good(lex+semantic): misses.append(row["query"])
    print("BASELINE",json.dumps(before["summary"]),flush=True)
    print("RETRIEVAL",count,hits,flush=True)
    candidate_path = out / "reranker.json"
    train(candidate_path, use_dense=args.dense, dataset_output=out / "pairs.csv", excluded_queries=excluded)
    candidate = CatalogClassifier(catalog, reranker=LogisticReranker.load(candidate_path), dense=dense)
    after = measure(candidate)
    checks = {k: after["summary"][k] >= before["summary"][k]
              for k in ["top1_expected", "top3_expected", "top1_all", "correct_silence", "exact_top1", "ok_precision"]}
    checks["calibration"] = after["summary"]["ece"] <= before["summary"]["ece"]
    with (out / "pairs.csv").open(encoding="utf-8") as handle: pairs = list(csv.DictReader(handle))
    training = {normalize(r["query"]) for r in pairs if r["split"] == "train"}
    validation = {normalize(r["query"]) for r in pairs if r["split"] == "validation"}
    checks["no_split_overlap"] = not (training & validation)
    checks["no_benchmark_overlap"] = not ((training | validation) & excluded)
    report = {"source": "catalog_synthetic", "neural_encoder_finetuned": False,
              "catalog_sha256": hashlib.sha256(catalog.read_bytes()).hexdigest(),
              "evaluation_sha256": hashlib.sha256(LABELED.read_bytes()).hexdigest(),
              "before": before, "after": after, "checks": checks,
              "accepted": all(checks.values()),
              "retrieval": {"count":count, "hits":hits, "misses":misses}}
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2),encoding="utf-8")
    print("AFTER", json.dumps(after["summary"]), flush=True)
    print("CHECKS", json.dumps(checks), "ACCEPTED", report["accepted"], flush=True)


if __name__ == "__main__": main()
