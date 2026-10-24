"""Purchase-line parsing: what in a real request is a product name, and what is a spec.

A classifier is asked about purchase lines, not about product names. A line from
an ERP export looks like this::

    Машина стиральная LG F2J3NS0W; бытовая; загрузка 6 кг; отжим 1000 об/мин; 1 шт

The identity of the goods is the part before the first separator ("Машина
стиральная LG F2J3NS0W"). Everything after it is an attribute: colour, voltage,
brand model, quantity. The catalog describes positions by *name*, so the
attribute tail adds nothing to retrieval and actively hurts the confidence
model: its coverage features count how many query words the catalog knows, and
`DDR4`, `1000 об/мин`, `1 шт` are unknown by construction. Measured on 200 real
purchase lines (`eval/experiments.md`, experiment 30): scoring only the name
moves 25 of them from "review" to "ok" while changing 3 codes.

The split is deliberately conservative: it never *replaces* the query. Callers
score both variants and keep the more confident answer, so a line whose
disambiguating information sits in the tail ("Плита Kronospan MDF;
древесно-волокнистая") keeps the answer that uses it. 125 of the 1126 catalog
position names themselves contain a semicolon ("Насосы воздушные или вакуумные;
воздушные или прочие газовые компрессоры"), which is exactly why the split must
not be applied blindly.
"""

from __future__ import annotations

import re

# Separators used by ERP/1C exports between the name and its attributes. The
# vertical bar is included because spreadsheets exported with a semicolon column
# delimiter often carry it inside the cell.
SEPARATORS = (";", "|")

# A head must look like a name: at least this many letters and this many
# characters. Otherwise "12; 5 шт" is a quantity table, not a product.
_MIN_LETTERS = 3
_MIN_LENGTH = 3

_LETTER_RE = re.compile(r"[^\W\d_]", re.UNICODE)
_SPACE_RE = re.compile(r"\s+")


def _looks_like_name(text: str) -> bool:
    """Whether ``text`` can stand alone as a product name."""
    stripped = text.strip()
    if len(stripped) < _MIN_LENGTH:
        return False
    return len(_LETTER_RE.findall(stripped)) >= _MIN_LETTERS


def split_purchase_query(query: str) -> tuple[str, str]:
    """Split a purchase line into ``(name, spec)``.

    ``name`` is always a non-empty string: when the query has no separator, or
    the part before the first separator does not look like a name, the whole
    query is returned as the name and ``spec`` is empty. Callers use
    ``spec == ""`` to skip the second scoring pass.
    """
    text = _SPACE_RE.sub(" ", (query or "").strip())
    if not text:
        return "", ""

    positions = [text.find(separator) for separator in SEPARATORS if text.find(separator) > 0]
    if not positions:
        return text, ""
    cut = min(positions)
    head, tail = text[:cut].strip(), text[cut + 1 :].strip()
    if not _looks_like_name(head):
        # "1 шт; 2 кг" — the separator is inside a specification, not after a name.
        return text, ""
    return head, tail


def name_variants(query: str) -> tuple[str, ...]:
    """Distinct texts worth scoring for one purchase line, longest first.

    The full line is scored first because it is what the user typed; the name
    alone is scored second. The caller picks by confidence, so the order only
    decides ties and the shape of logs.
    """
    text = (query or "").strip()
    head, spec = split_purchase_query(text)
    if not spec:
        return (text,)
    return (text, head)
