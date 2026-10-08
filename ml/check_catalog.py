"""Data quality checks for the real PP-1875 catalog."""

import json
from collections import Counter

from ml.config import RAW_DATA_DIR

path = RAW_DATA_DIR / "data.json"
rows = json.load(path.open(encoding="utf-8"))

names = [(r.get("name") or "").strip() for r in rows]
ids = [r.get("record_id") for r in rows]
pairs = [(r.get("okpd2"), r.get("name")) for r in rows]

print("rows:", len(rows))
print("code_role:", Counter(r.get("code_role") for r in rows))
print("appendix:", Counter(r.get("appendix") for r in rows))
print("empty names:", sum(1 for n in names if not n))
print("duplicate record_id:", len(ids) - len(set(ids)))
print("duplicate (okpd2, name):", len(pairs) - len(set(pairs)))
name_len_min, name_len_avg, name_len_max = (
    min(map(len, names)),
    sum(map(len, names)) // len(names),
    max(map(len, names)),
)
print("name len min/avg/max:", name_len_min, name_len_avg, name_len_max)
