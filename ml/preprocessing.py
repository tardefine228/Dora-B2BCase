"""Conservative input normalization shared by single, batch and file requests.

Never remove brands, measurements, materials, negations or arbitrary unknown words.
The original request is retained by the response/cache layer.
"""

import re
import unicodedata
from dataclasses import dataclass

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


@dataclass(frozen=True)
class SearchQuery:
    """A separate scoring view; the response always retains the user's text."""

    text: str
    prefixes: tuple[str, ...] = ()
    clarify: str | None = None


_QUANTITY = re.compile(
    r"\b(?:кол(?:ичество|[-.]?во)\s*[:=]?\s*)?\d+(?:[.,]\d+)?\s*"
    r"(?:шт(?:ук[аи]?)?\.?|упаков(?:ок|ки|ка)|упак\.?|уп\.)(?=\s|[,;.)]|$)",
    re.IGNORECASE,
)
_ELECTRONICS = re.compile(
    r"\b(?:ноутбук\w*|нетбук\w*|планшет\w*|смартфон\w*|телефон\w*|"
    r"монитор\w*|проектор\w*|принтер\w*|мфу|компьютер\w*)\b",
    re.IGNORECASE,
)
_SPEC_FIELD = re.compile(
    r"^(?:экран|диагональ|память|оперативная память|встроенная память|"
    r"процессор|частота процессора|разрешение|цвет|артикул|модель|"
    r"ram|rom|ssd|hdd|cpu|core|intel|amd)\b",
    re.IGNORECASE,
)
_TECH_VALUE = re.compile(
    r"\b(?:(?:экран|диагональ|память|ram|rom|ssd|hdd)\s*[:=]?\s*)?"
    r"\d+(?:[.,/]\d+)?\s*(?:гб|мб|тб|gb|mb|tb|дюйм\w*|ггц|ghz)\b",
    re.IGNORECASE,
)
_LATIN_TOKEN = re.compile(r"(?<!\w)[a-z][a-z0-9+./-]*(?!\w)", re.IGNORECASE)
_KEEP_LATIN = {"usb", "usb-c", "type-c", "hdmi", "vga", "lan", "led", "oled", "lcd"}
_FURNITURE = re.compile(
    r"\b(?:стул(?:а|у|ом|е)?|стулья|стульев|стульями|столы?|стола|столов|"
    r"кресл[аоы]?|шкаф\w*|стеллаж\w*|тумб\w*)\b",
    re.IGNORECASE,
)
_SPECIAL_FURNITURE = re.compile(
    r"медицин|больнич|санитар|стоматолог|операцион|гинеколог|инвалид|"
    r"лаборатор|вытяжн|эндоскоп|парикмахер|массаж|чертеж|пеленаль|"
    r"холодильн|морозильн|сушильн|духов|электр|управлен|распредел|"
    r"сервер|телеком|пожар|теплов",
    re.IGNORECASE,
)


def search_query(text: str) -> SearchQuery:
    """Remove explicit commercial metadata, then resolve common product names.

    Never drop an arbitrary unknown Russian word: it can be the product head,
    a material, a restriction or a negation. Aliases are anchored at the start,
    so a cartridge for a printer or a mouse pad cannot turn into the device.
    """
    text = clean_query(text)
    text = _QUANTITY.sub(" ", text)
    electronics = bool(_ELECTRONICS.search(text))
    if electronics:
        fields = re.split(r"\s*;\s*", text)
        # Only explicit fields are metadata. Keep purpose/material/negation fields.
        fields = [
            field
            for i, field in enumerate(fields)
            if i == 0
            or not _SPEC_FIELD.match(field)
            or re.search(r"\b(?:не|без|кроме)\b", field, re.IGNORECASE)
        ]
        text = "; ".join(fields)
        text = _TECH_VALUE.sub(" ", text)
        text = _LATIN_TOKEN.sub(
            lambda match: match[0] if match[0].lower() in _KEEP_LATIN else " ", text
        )
        text = re.sub(r"\b(?:wi[ -]?fi|bluetooth)\b", " ", text, flags=re.IGNORECASE)
    text = " ".join(text.split()).strip(" ;,")
    lowered = text.casefold().replace("ё", "е")

    # Furniture is specified by purpose and material, not by the word 'chair'.
    furniture = _FURNITURE.match(lowered)
    if furniture and not _SPECIAL_FURNITURE.search(lowered):
        suffix = text[furniture.end() :].strip()
        if re.search(r"офис|для офис|торгов", lowered):
            prefix = "31.01.1"
            if re.search(r"деревян|из дерева", lowered):
                prefix = "31.01.12"
            elif re.search(r"металл|стальн", lowered):
                prefix = "31.01.11"
            return SearchQuery("Мебель для офисов " + suffix, (prefix,))
        if re.search(r"кухон", lowered):
            return SearchQuery("Мебель кухонная " + suffix, ("31.02",))
        if re.search(r"металл|стальн", lowered):
            return SearchQuery("Мебель металлическая " + suffix, ("31.09.11",))
        if re.search(r"пластик|пластмасс", lowered):
            return SearchQuery("Мебель из пластмассовых материалов " + suffix, ("31.09.14",))
        if re.search(r"деревян|из дерева", lowered):
            return SearchQuery(
                "Мебель деревянная " + suffix,
                ("31.09.12", "31.09.13"),
                "Уточните назначение мебели: для спальни, столовой или другое.",
            )
        return SearchQuery(
            "Мебель " + suffix,
            ("31",),
            "Уточните назначение и материал: офисный, кухонный, металлический и т. п.",
        )

    # These are terminology bridges, not fixed answers: search still selects a
    # real, non-excluded row from the currently loaded catalog.
    if re.match(r"^мфу\b|^многофункциональн\w*\s+устройств", lowered):
        return SearchQuery("Устройства периферийные печать копирование сканирование", ("26.20.18",))
    if re.match(r"^(?:лазерн\w*\s+|струйн\w*\s+)?принтер\b", lowered) and not re.search(
        r"3\s*[-dд]|трехмер|объемн", lowered
    ):
        return SearchQuery("Устройства ввода или вывода", ("26.20.16",))
    if re.match(r"^(?:мышь|мышка|клавиатура)\b", lowered) and not re.search(
        r"музык|фортепиан|синтезатор|животн|лаборатор", lowered
    ):
        return SearchQuery("Устройства ввода или вывода", ("26.20.16",))
    if re.match(r"^(?:usb[- ]*)?(?:флеш\w*|флэш\w*)\b", lowered):
        return SearchQuery("Устройства запоминающие хранения данных", ("26.20.2",))
    if re.match(r"^бумага\s+(?:офисн|для\s+(?:печат|принтер|ксерокс))", lowered):
        return SearchQuery("Бумага", ("17.12",))
    return SearchQuery(text)


def compatible_purpose(query: str, title: str) -> bool:
    """Do not confuse an accessory for a computer with one for a prosthesis."""
    if not re.match(r"^(?:чехол|держатель|подставка|картридж|крепление)\b", query, re.I):
        return True
    requested = re.search(r"\bдля\s+([а-яё-]+)", query, re.I)
    purposes = re.findall(r"\bдля\s+([а-яё-]+)", title, re.I)
    if not requested or not purposes:
        return True
    from ml.features import match_weight, stem

    return any(match_weight(stem(requested[1]), stem(purpose)) > 0 for purpose in purposes)


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
