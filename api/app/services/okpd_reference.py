"""Display-only OKPD2 reference: never changes ranking or assigns a code."""
import json
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def reference() -> dict:
    path = Path(__file__).resolve().parents[1] / "reference" / "okpd2.json"
    return json.loads(path.read_text(encoding="utf-8"))


def code_description(code: str) -> str | None:
    return reference()["codes"].get(code)


def code_context(code: str) -> str | None:
    """Nearest named ancestor with a distinct title (e.g. string vs wind altos)."""
    codes = reference()["codes"]
    title = codes.get(code)
    parts = code.split(".")
    parents = []
    if len(parts) == 4:
        last = parts[-1]
        for n in (1, 2):
            candidate = ".".join(parts[:-1] + [last[:-n] + "0" * n])
            if candidate != code:
                parents.append(candidate)
    for length in range(len(parts) - 1, 0, -1):
        parents.append(".".join(parts[:length]))
    for parent in parents:
        if codes.get(parent) and codes[parent] != title:
            return codes[parent]
    return None
