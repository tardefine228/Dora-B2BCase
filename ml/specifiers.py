"""Technical specifiers: the part of a purchase line that is not the goods.

A line from an ERP export carries the product *and* its attributes::

    Осциллограф цифровой запоминающий 100 МГц; 2 канала; 220 В

The catalog names the position by the goods alone ("Осциллоскопы и осциллографы
электронно-лучевые"). Every attribute word the catalog does not know is an
unknown word for the coverage features, and unknown words are evidence *against*
a match: measured on the same candidate, `pair_probability` is 0.485 for
"Осциллограф" and 0.084 for "Осциллограф цифровой запоминающий"
(`eval/experiments.md`, эксперимент 34).

This module is the vocabulary of such attributes, split in two parts:

* ``SPECIFIER_STEMS`` — Russian adjectives and participles that describe a
  *version* of a product rather than the product itself ("цифровой",
  "трёхфазный", "фланцевый"). They are used by ``strip_specifiers`` and are
  deliberately limited to words that no position name *needs* to be recognised
  by — a word that carries identity ("аккумуляторный", "дизельный") is **not**
  here, because dropping it changes which position is right;
* ``SPEC_EXAMPLES`` — realistic measurement strings ("IP54", "10 бар",
  "16 ГБ"), used to synthesize realistic stress lines in
  ``eval/specifier_stress.py``.

Nothing here changes the user's query text: the stripper is only used to build
an *additional* scoring variant, and the caller keeps the more confident answer.
"""

from __future__ import annotations

import re

from ml.features import normalize, stem, tokens

# Adjectives/participles that describe the version of a product. Rule for adding
# a word: a position name that contains it stays recognisable without it.
# The tuple is also the training vocabulary (``ml/train_reranker.py`` appends
# these words to synthetic queries, so the reranker learns that an attribute is
# not evidence against a match — эксп. 35).
SPEC_ADJECTIVES: tuple[str, ...] = (
        # Форма и исполнение
        "цифровой",
        "аналоговый",
        "запоминающий",
        "программируемый",
        "перепрограммируемый",
        "многофункциональный",
        "универсальный",
        "профессиональный",
        "бытовой",
        "промышленный",
        "офисный",
        "стационарный",
        "переносной",
        "мобильный",
        "настольный",
        "напольный",
        "настенный",
        "потолочный",
        "встраиваемый",
        "накладной",
        "уличный",
        "внутренний",
        "наружный",
        "герметичный",
        "взрывозащищенный",
        "антивандальный",
        "усиленный",
        "облегченный",
        "компактный",
        "миниатюрный",
        "автоматический",
        "полуавтоматический",
        "ручной",
        "механический",
        "электрический",
        "электронный",
        "пневматический",
        "гидравлический",
        "беспроводной",
        "проводной",
        "сетевой",
        "аккумуляторный",
        "энергосберегающий",
        "светодиодный",
        "люминесцентный",
        "галогенный",
        "индукционный",
        "электромагнитный",
        "ультразвуковой",
        "оптический",
        "лазерный",
        "инфракрасный",
        "мембранный",
        "погружной",
        "центробежный",
        "вихревой",
        "роторный",
        "поршневой",
        "пластинчатый",
        "дисковый",
        "шаровой",
        "фланцевый",
        "резьбовой",
        "приварной",
        # Электрика
        "трехфазный",
        "однофазный",
        "двухполюсный",
        "трехполюсный",
        "четырехполюсный",
        "заземленный",
        "экранированный",
        "бронированный",
        "силовой",
        "слаботочный",
        "высоковольтный",
        "низковольтный",
        "импульсный",
        "стабилизированный",
        # Материалы и покрытия
        "стальной",
        "алюминиевый",
        "чугунный",
        "медный",
        "латунный",
        "бронзовый",
        "пластиковый",
        "полипропиленовый",
        "полиэтиленовый",
        "стеклянный",
        "деревянный",
        "оцинкованный",
        "хромированный",
        "окрашенный",
        "нержавеющий",
        "керамический",
        "композитный",
        # Геометрия и размеры (прилагательные; числа отсекаются отдельно)
        "радиальный",
        "осевой",
        "диагональный",
        "вертикальный",
        "горизонтальный",
        "прямой",
        "угловой",
        "круглый",
        "квадратный",
        "профильный",
        "двухкамерный",
        "однокамерный",
        "многокамерный",
        "двухстворчатый",
        "раздвижной",
        "распашной",
)

SPECIFIER_STEMS: frozenset[str] = frozenset(stem(word) for word in SPEC_ADJECTIVES)

# Words that make a line look like a specification even when the token is not in
# the list above: units and dimensions. A digit-bearing token is a specifier
# unless the digits are part of a model name ("F2J3NS0W", "x200").
_UNIT_RE = re.compile(
    r"^(?:"
    r"\d+(?:[.,]\d+)?"
    r"(?:в|вт|квт|мвт|кв|мв|а|ма|ач|мач|гц|кгц|мгц|ггц|мм|см|дм|м|км|кг|г|мг|т|л|мл|м3|м2|шт|"
    r"гб|тб|мб|кб|гбит|мбит|"
    r"об|мин|с|ч|сут|дн|бар|мпа|кпа|па|кал|ккал|ккалч|втч|квтч|°c|%|руб|тр)"
    r"|(?:дн|ду|dn)\d+"
    r"|ip\d{2}"
    r"|№\d+"
    r")$",
    re.IGNORECASE,
)
_LATIN_RE = re.compile(r"[a-z]", re.IGNORECASE)

# Realistic attribute strings for the synthetic stress lines. The list is
# intentionally generic: a real line carries attributes nobody could predict for
# its position, and that is exactly what the stress measurement has to model.
SPEC_EXAMPLES: tuple[str, ...] = (
    "220 В",
    "380 В",
    "12 В",
    "24 В",
    "1000 Вт",
    "2,5 кВт",
    "50 Гц",
    "IP54",
    "IP65",
    "10 бар",
    "Ду 50",
    "500 мм",
    "1200х600 мм",
    "2,5 мм",
    "А4",
    "24 дюйма",
    "55 дюйма",
    "16 ГБ",
    "512 ГБ",
    "1 ТБ",
    "8 портов",
    "16 портов",
    "3 фазы",
    "60 Ач",
    "100 Ач",
    "класс А",
    "ГОСТ 12.4.107",
    "цвет белый",
    "цвет серый",
    "в комплекте 2 шт",
    "цифровой",
    "запоминающий",
    "трехфазный",
    "радиальный",
    "фланцевый",
    "нержавеющий",
    "силовой",
    "стационарный",
    "автоматический",
    "светодиодный",
    "аккумуляторный",
    "переносной",
    "уличный",
    "взрывозащищенный",
    "антивандальный",
    "усиленный",
    "порошковый",
    "гарантия 24 мес",
    "срок поставки 30 дней",
)


def is_specifier_token(token: str) -> bool:
    """Whether a single token carries no product identity.

    Latin tokens are kept: "ssd", "usb", "ups" and model codes are how a
    specialist names the goods, and the lexicon bridges several of them.
    """
    if not token:
        return True
    lowered = normalize(token)
    if not lowered:
        return True
    if lowered.isdigit():
        return True
    # Units are checked before the Latin skip: "IP54" and "16GB" are attributes,
    # while "ssd" and "usb" are names of the goods.
    if _UNIT_RE.match(lowered):
        return True
    if _LATIN_RE.search(lowered):
        return False
    return stem(lowered) in SPECIFIER_STEMS


def strip_specifiers(text: str) -> str:
    """``text`` without its attribute words, or ``""`` when nothing is left.

    The result is meant as an extra scoring variant next to the untouched query,
    never as a replacement: a line whose only identity word is an attribute
    adjective ("Светодиодный") must keep the answer it already has.
    """
    kept = [word for word in (text or "").split() if not is_specifier_token(word)]
    if len(kept) == len((text or "").split()):
        return ""
    result = " ".join(kept).strip(" ,;")
    if not result or len(tokens(result)) < 1:
        return ""
    return result
