"""Conservative input normalization shared by single, batch and file requests.

Never remove brands, measurements, materials, negations or arbitrary unknown words.
The original request is retained by the response/cache layer.
"""

import re
import unicodedata

OKPD2_PATTERN = r"^\d{2}(?:\.\d{1,2}){0,2}(?:\.\d{1,3})?$"

_PREFIX = re.compile(
    r"^(?:(?:пожалуйста|срочно)\b[, :]*\s*|"
    r"(?:прошу|просим)\s+(?:определить|найти|подобрать)\s+(?:код\s+)?(?:окпд\s*2\s+)?(?:для\s+)?|"
    r"(?:(?:закупка|поставка)(?:\s+товаров?)?|наименование\s+товара|название\s+товара)\s*[:—–-]\s*)",
    re.IGNORECASE,
)
_LIST = re.compile(r"^\s*(?:[•●▪]+\s*|\d{1,4}[.)]\s+)")


def clean_query(text: str) -> str:
    """Remove formatting noise and explicit request wrappers, not product meaning."""
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    text = "".join(c for c in text if unicodedata.category(c) not in {"Cf", "Cc"})
    while text:
        # Each successful step consumes characters; dotted codes are not list markers.
        updated = _PREFIX.sub("", _LIST.sub("", text), count=1).strip()
        if updated == text:
            break
        text = updated
    return text


def excluded_codes(rows: list[dict]) -> dict[tuple[str, str], list[str]]:
    """Exceptions are scoped to a position, never globally to the entire decree."""
    result: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        if row.get("code_role") == "Исключение" and row.get("okpd2"):
            key = (str(row.get("appendix", "")), str(row.get("position", "")))
            result.setdefault(key, []).append(str(row["okpd2"]).strip())
    return result


def query_hits_exclusion(query: str, exclusions: list[str]) -> bool:
    codes = re.findall(r"(?<!\d)\d{2}(?:\.\d{1,3}){1,4}(?!\d)", query)
    return any(c == ex or c.startswith(ex + ".") for c in codes for ex in exclusions)
