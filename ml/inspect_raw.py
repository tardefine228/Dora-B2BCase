"""Quick inspection of the raw PP-1875 catalog file."""

import csv
import json

from ml.config import RAW_DATA_DIR

FILE_NAME = "data.json"  # change to .csv if needed


def inspect() -> None:
    path = RAW_DATA_DIR / FILE_NAME
    if not path.exists():
        print(f"File not found: {path}")
        return

    if path.suffix == ".json":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        print(f"Rows: {len(data)}")
        if isinstance(data, list) and data:
            print(f"Keys: {sorted(data[0].keys())}")
            for row in data[:5]:
                print(row)

    elif path.suffix == ".csv":
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"Rows: {len(rows)}")
        print(f"Columns: {reader.fieldnames}")
        for row in rows[:5]:
            print(row)

    else:
        print(f"Unsupported format: {path.suffix} (if it is .xlsx — tell me, we add pandas)")


if __name__ == "__main__":
    inspect()