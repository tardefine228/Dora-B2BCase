"""Structural OKPD2 hierarchy helpers shared by serving and HTTP audits.

A one-digit subgroup can be expanded without another dot (26.12.2 → 26.12.20),
while deeper levels use a dot (27.20.22 → 27.20.22.000). String prefix alone is
not enough: 27.20.22 is not a parent of 27.20.220000.
"""

from __future__ import annotations

from collections.abc import Iterable


def is_code_ancestor(parent: str, child: str) -> bool:
    """Return whether ``parent`` is a strict structural ancestor of ``child``."""
    parent = parent.strip()
    child = child.strip()
    if not parent or not child or parent == child:
        return False
    if child.startswith(parent + "."):
        return True
    last_segment = parent.rsplit(".", 1)[-1]
    if len(last_segment) != 1 or not child.startswith(parent):
        return False
    suffix = child[len(parent) :]
    return bool(suffix) and suffix[0].isdigit() and (len(suffix) == 1 or suffix[1] == ".")


def unique_detailed_code(codes: Iterable[str]) -> str | None:
    """Return the only leaf when all distinct codes form one hierarchy branch."""
    distinct = {code.strip() for code in codes if code and code.strip()}
    if len(distinct) < 2:
        return None
    leaves = [
        code
        for code in distinct
        if not any(is_code_ancestor(code, other) for other in distinct if other != code)
    ]
    return leaves[0] if len(leaves) == 1 else None
