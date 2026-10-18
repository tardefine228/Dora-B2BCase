"""Hand-curated product→category bridges for the PP-1875 catalog.

The official list names abstract positions ("Одежда, кроме одежды из меха",
"Обувь"), while procurement specialists type concrete goods ("кроссовки
мужские", "костюм"). The catalog itself contains no synonymy, and a true
semantic model (see ``ml.embedder`` with SentenceTransformers) closes that gap
only when its weights are provided. Until then this small, editable lexicon
bridges everyday product words to the words the list actually uses.

Rules of the lexicon:

* one source word → one category word that exists in the catalog;
* every entry is a deliberate, reviewable decision (keep it small);
* the mapping only *adds* words to the query for scoring — it never renames the
  user's query and never changes the legal text.

Two helpers consume the table:

* ``expand_query`` *appends* the category words (used by lexical retrieval so the
  shortlist contains the abstract position);
* ``substitute_query`` *replaces* the product words by the category words (used
  by the reranker features so a bridged word is not counted as an unknown
  Russian word that penalizes the match — "костюм мужской" is scored like
  "одежда мужской").

``CANONICAL_CODES`` pins each category word to the OKPD2 position that bridged
queries should resolve to; training uses it to label positives, so an abstract
word that also occurs in narrower positions ("Одежда защитная огнестойкая",
"Части светильников") does not steal the examples.

Edit this file and rebuild the model (``python scripts/build_model.py``) so the
reranker learns the bridged queries too.
"""

from __future__ import annotations

from ml.features import normalize, stem, stems, tokens

# source product word -> catalog category word
BRIDGES: dict[str, str] = {
    # Обувь (15.2)
    "кроссовки": "обувь",
    "кроссовка": "обувь",
    "кеды": "обувь",
    "ботинки": "обувь",
    "ботинок": "обувь",
    "сапоги": "обувь",
    "сапог": "обувь",
    "туфли": "обувь",
    "туфля": "обувь",
    "тапочки": "обувь",
    "шлёпанцы": "обувь",
    "сланцы": "обувь",
    "сандалии": "обувь",
    "мокасины": "обувь",
    "валенки": "обувь",
    "берцы": "обувь",
    # Одежда (14.1)
    "костюм": "одежда",
    "платье": "одежда",
    "брюки": "одежда",
    "рубашка": "одежда",
    "сорочка": "одежда",
    "куртка": "одежда",
    "пальто": "одежда",
    "свитер": "одежда",
    "футболка": "одежда",
    "майка": "одежда",
    "штаны": "одежда",
    "джинсы": "одежда",
    "юбка": "одежда",
    "блузка": "одежда",
    "халат": "одежда",
    "пижама": "одежда",
    "носки": "одежда",
    "носков": "одежда",
    "колготки": "одежда",
    "перчатки": "одежда",
    "варежки": "одежда",
    "шапка": "одежда",
    "шарф": "одежда",
    "галстук": "одежда",
    "жилет": "одежда",
    "комбинезон": "одежда",
    "полукомбинезон": "одежда",
    "спецовка": "одежда",
    "стёганка": "одежда",
    "бельё": "одежда",
    "белье": "одежда",
    "пиджак": "одежда",
    "плащ": "одежда",
    "ветровка": "одежда",
    "пуховик": "одежда",
    "толстовка": "одежда",
    "кофта": "одежда",
    "водолазка": "одежда",
    "шорты": "одежда",
    "сарафан": "одежда",
    "спецодежда": "одежда",
    "унты": "обувь",
    "галоши": "обувь",
    "полуботинки": "обувь",
    "босоножки": "обувь",
    "полусапоги": "обувь",
    # Меховые изделия (14.20)
    "шуба": "меховые",
    "дублёнка": "меховые",
    "дубленка": "меховые",
    # Шины и покрышки (22.11.1)
    "шина": "покрышки",
    "шины": "покрышки",
    "покрышка": "покрышки",
    "покрышки": "покрышки",
    "камера": "покрышки",
    # Лампы (27.40.1) / светильники (27.40.22)
    "лампочка": "лампы",
    "светильник": "светильники",
    "светильники": "светильники",
    "бра": "светильники",
    "торшер": "светильники",
    # Ткани (13.2)
    "ткань": "ткани",
    "ситец": "ткани",
    "сатин": "ткани",
    "бязь": "ткани",
    "лён": "ткани",
    "лен": "ткани",
    "фланель": "ткани",
    "трикотаж": "трикотажные",
    # Электрочайники (27.51.24.110)
    "чайник": "электрочайники",
    # Печи микроволновые (27.51.27.000)
    "микроволновка": "микроволновые",
    # Утюги (27.51.23.130)
    "утюг": "утюги",
    # Телевизоры (26.40.20.*)
    "телевизор": "телевизоры",
    "тв": "телевизоры",
    # Телефоны (26.30.22)
    "смартфон": "телефонные",
    "телефон": "телефонные",
    "мобильник": "телефонные",
    # Компьютеры (26.20.11)
    "ноутбук": "портативные",
    "нетбук": "портативные",
    "планшет": "портативные",
    "ноут": "портативные",
    # Мониторы (26.20.17 / 26.40.34.110)
    "монитор": "мониторы",
    "проектор": "проекторы",
    # Цемент (23.51)
    "цемент": "цемент",
    # Двери и окна (25.12.10.000)
    "дверь": "двери",
    "двери": "двери",
    "окно": "окна",
    "окна": "окна",
    "форточка": "окна",
    "стеклопакет": "окна",
    "ворота": "двери",
    "калитка": "двери",
    # Лакокрасочные материалы (20.30)
    "краска": "лакокрасочные",
    "эмаль": "лакокрасочные",
    "грунтовка": "лакокрасочные",
    "лак": "лакокрасочные",
    # Компьютерная периферия: перечень называет её абстрактно
    # («Устройства ввода или вывода…», 26.20.16/18, 26.20.2, 26.12.2, 26.30.11.110)
    "принтер": "вывода",
    "сканер": "ввода",
    "клавиатура": "ввода",
    "мышь": "ввода",
    "мфу": "периферийные",
    "накопитель": "запоминающие",
    "ssd": "запоминающие",
    "винчестер": "запоминающие",
    "видеокарта": "видеоплаты",
    "коммутатор": "коммутации",
    # Канцелярия и игрушки: официальные названия короткие и буквальные
    "бумага": "бумага",
    "тетрадь": "тетради",
    "блокнот": "блокноты",
    "витрина": "витрины",
    "конструктор": "игрушки",
    # Ручной инструмент: якорные слова официальных названий уникальны
    "дрель": "механизированным",
    "сверло": "сменных",
    "бур": "сменных",
    "паяльник": "пайки",
    # Кухонная теплотехника
    "варочная": "варочные",
    # Мебель: офис против металла различаем словом «офисный»
    "офисный": "офисов",
    # Склад и инженерка: описательные названия перечня
    "погрузчик": "автопогрузчики",
    "насос": "центробежные",
    "клапан": "арматура",
    "компрессор": "газовые",
    "кассовый": "кассовые",
    "чпу": "числового",
    "подошва": "вулканизированной",
    # Платы и моющие средства: ретранкер иначе переворачивает верный топ-1
    "плата": "звуковые",
    "порошок": "моющие",
    # Холодильное оборудование: «глухая дверь» тянет в 25.12.10 без моста
    "холодильный": "холодильные",
    "морозильный": "холодильные",
    # Аккумуляторы и шины: стартерные/восстановленные против общих
    "стартерный": "запуска",
    "восстановленная": "восстановленные",
}

# Product words whose bridge must NOT fire when a vetoing specifier stem is
# present in the query: the specifier points to a neighbouring position.
BRIDGE_VETO: dict[str, set[str]] = {
    "насос": {"роторн", "объемн", "винтов", "шестерен", "вакуумн", "плунжерн"},
}

# Category word -> OKPD2 code of the canonical position it bridges into.
# Training attaches bridged product words only to this row.
CANONICAL_CODES: dict[str, str] = {
    "обувь": "15.2",
    "одежда": "14.1",
    "трикотажные": "14.3",
    "меховые": "14.20",
    "покрышки": "22.11.1",
    "лампы": "27.40.1",
    "светильники": "27.40.22",
    "ткани": "13.2",
    "электрочайники": "27.51.24.110",
    "микроволновые": "27.51.27.000",
    "утюги": "27.51.23.130",
    "телевизоры": "26.40.20",
    "телефонные": "26.30.22",
    "портативные": "26.20.11",
    "мониторы": "26.20.17",
    "проекторы": "26.20.17",
    "цемент": "23.51",
    "двери": "25.12.10.000",
    "окна": "25.12.10.000",
    "лакокрасочные": "20.30",
    "вывода": "26.20.16",
    "ввода": "26.20.16",
    "бумага": "17.12",
    "тетради": "17.23.13.194",
    "блокноты": "17.23.13.191",
    "витрины": "28.25.13.114",
    "игрушки": "32.4",
    "механизированным": "28.24.1",
    "сменных": "25.73.40",
    "пайки": "27.90.31.110",
    "варочные": "27.51.28",
    "офисов": "31.01.1",
    "автопогрузчики": "28.22.15.110",
    "центробежные": "28.13.14",
    "арматура": "28.14.11",
    "газовые": "28.13.2",
    "кассовые": "26.20.12",
    "числового": "26.20.40.150",
    "вулканизированной": "22.19.73",
    "звуковые": "26.12.2",
    "моющие": "20.41.32",
    "холодильные": "28.25.13.111",
    "запуска": "27.20.21",
    "восстановленные": "22.11.2",
    "периферийные": "26.20.18",
    "запоминающие": "26.20.2",
    "видеоплаты": "26.12.2",
    "коммутации": "26.30.11.110",
}

# Reverse index: catalog word -> source words that map to it.
_BY_TARGET: dict[str, list[str]] = {}
for _source, _target in BRIDGES.items():
    _BY_TARGET.setdefault(_target, []).append(_source)

# Stem of a source word -> target word (stems make the lookup inflection-proof).
_BRIDGE_BY_STEM: dict[str, str] = {}
for _source, _target in BRIDGES.items():
    _BRIDGE_BY_STEM.setdefault(normalize(_source), _target)
    _BRIDGE_BY_STEM.setdefault(stems(_source)[0], _target)


def bridged_targets(query: str) -> dict[str, str]:
    """Map each bridged query token (normalized) to its category word."""
    found: dict[str, str] = {}
    query_stems = {stem(token) for token in tokens(query)}
    for token in tokens(query):
        target = _BRIDGE_BY_STEM.get(token) or _BRIDGE_BY_STEM.get(stem(token))
        if not target:
            continue
        # A vetoing specifier ("насос роторный объёмный") means the product
        # belongs to a neighbouring position, not the bridged one.
        vetoes = BRIDGE_VETO.get(token) or BRIDGE_VETO.get(stem(token))
        if vetoes and query_stems & vetoes:
            continue
        found[token] = target
    return found


def expand_query(query: str) -> str:
    """Append category words for recognized product words, keeping the query text."""
    additions = sorted(set(bridged_targets(query).values()))
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def substitute_query(query: str, keep_stems: set[str] | None = None) -> str:
    """Replace bridged product words by their category words (scoring only).

    Unlike ``expand_query`` the product word itself disappears, so it is not
    counted as an unknown Russian word by the coverage features. A product word
    that the candidate position *does* mention ("ноутбуки" inside the
    portable-computers name) is kept next to its category word when its stem is
    in ``keep_stems`` (the candidate's stems). The replacement is word-by-word;
    everything else (modifiers, brands, numbers) stays in place.
    """
    targets = bridged_targets(query)
    if not targets:
        return query
    replaced: list[str] = []
    for word in query.split():
        key = normalize(word)
        core = next(iter(tokens(word)), key)
        target = targets.get(core)
        if target is None:
            replaced.append(word)
            continue
        if keep_stems is not None and stem(core) in keep_stems:
            replaced.append(word)
        if target not in replaced:
            replaced.append(target)
    return " ".join(replaced)


def sources_for_target_word(target_word: str) -> list[str]:
    """Product words that bridge into a catalog word (used by training)."""
    return _BY_TARGET.get(target_word, [])
