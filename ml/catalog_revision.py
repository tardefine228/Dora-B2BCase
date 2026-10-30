"""Resolve one immutable catalog revision across API and worker processes."""
import json
from pathlib import Path


def active_catalog_dir(root: Path) -> Path:
    pointer = root / "active-catalog.json"
    if not pointer.exists():
        return root
    version = json.loads(pointer.read_text(encoding="utf-8"))["revision"]
    if not isinstance(version, str) or not version.isalnum():
        raise ValueError("Invalid catalog revision")
    target = root / "revisions" / version
    if not target.is_dir():
        raise ValueError("Catalog revision is missing")
    return target
