"""Import display-only code names from the classifikators.ru XLSX download.

Usage: python scripts/import_okpd_reference.py path/to/okpd.xlsx
Requires openpyxl. No network requests and no changes to the PP1875 catalog.
"""
import argparse
import hashlib
import json
import re
from datetime import date
from pathlib import Path

import openpyxl


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xlsx", type=Path)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]
                        / "api/app/reference/okpd2.json")
    args = parser.parse_args()
    workbook = openpyxl.load_workbook(args.xlsx, read_only=True, data_only=True)
    rows = list(workbook.active.values)
    codes = {}
    for row in rows[7:]:
        if not row[1] or not row[2]:
            continue
        code, title = str(row[1]).strip(), str(row[2]).strip()
        if not re.fullmatch(r"(?:[A-U]|\d{2}(?:\.\d{1,3}){0,3})", code) or code in codes:
            raise ValueError(f"Invalid or duplicate code: {code}")
        codes[code] = title
    if len(codes) < 20000:
        raise ValueError("Incomplete reference: fewer than 20000 entries")
    payload = {
        "source": "https://classifikators.ru/assets/downloads/okpd/okpd.xlsx",
        "edition": rows[1][0], "retrieved": date.today().isoformat(),
        "sha256": hashlib.sha256(args.xlsx.read_bytes()).hexdigest(), "codes": codes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Imported {len(codes)} code descriptions into {args.output}")


if __name__ == "__main__":
    main()
