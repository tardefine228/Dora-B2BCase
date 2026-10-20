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
    # «конструктор» раньше мостил в «Игры и игрушки» (32.4), но перечень
    # 1875 строительные наборы не содержит — это ложный друг с уверенностью
    # 0.66. Пара запрещена в PAIR_VETO, мост удалён (ML-3).
    # Ручной инструмент: якорные слова официальных названий уникальны
    "дрель": "механизированным",
    "сверло": "сменных",
    "бур": "сменных",
    "паяльник": "пайки",
    # Кухонная теплотехника
    "варочная": "варочные",
    # Мебель: офис против металла различаем словом «офисный»
    "офисный": "офисов",
    "стеллаж": "металлическая",
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

# The opposite of a veto: the bridge fires only together with a confirming
# context stem. (Empty for now: the "кухонный" experiment showed a raw stem
# overlap ("вытяжка кухонная" ~ "Мебель кухонная") bypasses any bridge guard,
# so context-dependent goods stay out of the lexicon until pair-level rules.)
BRIDGE_REQUIRE: dict[str, set[str]] = {}

# --- Pair-level bans (ML-3) -------------------------------------------------
# The five "confident false friends" (raw confidence >= 0.50 on the labeled
# set) cannot be fixed by a threshold or an embedder: the query and the
# position legitimately share words ("Шредер для бумаги" ~ "Бумага и
# картон", "камера" ~ "Шины, покрышки и камеры резиновые"). Each entry bans
# a (query anchor word, candidate position) pair: when the anchor's stem is
# in the query, candidates under the listed OKPD2 prefixes are removed from
# the shortlist before reranking, so the pair can never be suggested.
#
# Rules of the table (same discipline as BRIDGES):
# * one anchor word -> the OKPD2 prefixes it must never resolve to;
# * every entry is a deliberate, reviewable decision with a reason;
# * a veto removes candidates from the answer — it never rewrites the query.
PAIR_VETO: dict[str, frozenset[str]] = {
    # "Шредер для бумаги" -> 17.12 «Бумага и картон» (raw 0.69): a shredder
    # is office equipment, not paper; the word «бумага» in the query is a
    # specifier, not the goods.
    "шредер": frozenset({"17.12"}),
    # "Конструктор детский" -> 32.4 «Игры и игрушки» (raw 0.66): building
    # sets are not enumerated by the decree; the bridge was removed with it.
    "конструктор": frozenset({"32.4"}),
    # "Огнетушитель порошковый" -> 20.41.32 «Средства моющие…» (raw 0.55):
    # the «порошок»->«моющие» bridge fires on the specifier «порошковый».
    "огнетушитель": frozenset({"20.41.32"}),
}

# Context-gated bans: the anchor is ambiguous ("камера" is a tire tube AND
# a surveillance camera), so the ban fires only when a context stem points
# at the *other* meaning. "Камера велосипедная" (22.11.1) and "Камера
# холодильная сборная" (28.25.13.112) stay answerable.
PAIR_VETO_CONTEXT: dict[str, tuple[frozenset[str], frozenset[str]]] = {
    "камера": (
        frozenset(
            {
                "видеонаблюд",  # камера видеонаблюдения
                "наблюд",
                "веб",  # веб-камера (exact stem: «веб» 3 letters)
                "web",
                "видео",  # видеокамера, цифровое видео
                "цифров",
                "ip",  # IP-камера (exact stem)
                "охранн",  # охранная камера
                "фото",  # фотоаппарат, фотокамера
            }
        ),
        frozenset({"22.11.1"}),  # «Шины, покрышки и камеры резиновые»
    ),
}

# Stems of the anchor words, precomputed for inflection-proof lookup.
# Keyed by the *stemmed* anchor: the light stemmer trims endings
# ("камера" -> "камер", "огнетушитель" -> "огнетушител"), and the
# query stems must match the anchor on the same scale.
_VETO_ANCHOR_STEMS: dict[str, str] = {}
for _anchor in (*PAIR_VETO, *PAIR_VETO_CONTEXT):
    _VETO_ANCHOR_STEMS[stem(_anchor)] = _anchor


def _stem_hit(anchor: str, query_stems: set[str]) -> bool:
    """True when a query stem is the anchor or contains it.

    Containment catches compounds the tokenizer never splits
    ("видеокамера" carries "камер", "огнетушитель порошковый" is two words).
    Short anchors (< 4 chars) must match exactly: "ip" inside "пипетка"
    is a coincidence, not a camera.
    """
    if len(anchor) < 4:
        return anchor in query_stems
    return any(anchor in stem_value for stem_value in query_stems)


def vetoed_codes(query: str) -> set[str]:
    """OKPD2 prefixes banned for this query by the pair-level rules."""
    query_stems = {stem(token) for token in tokens(query)}
    banned: set[str] = set()
    for anchor_stem, anchor in _VETO_ANCHOR_STEMS.items():
        if not _stem_hit(anchor_stem, query_stems):
            continue
        codes = PAIR_VETO.get(anchor)
        if codes:
            banned.update(codes)
        gated = PAIR_VETO_CONTEXT.get(anchor)
        if gated:
            context, codes = gated
            if any(
                context_stem in query_stems
                or (
                    len(context_stem) >= 4
                    and any(context_stem in stem_value for stem_value in query_stems)
                )
                for context_stem in context
            ):
                banned.update(codes)
    return banned


def code_is_vetoed(query: str, code: str) -> bool:
    """True when the (query, candidate code) pair is banned by PAIR_VETO."""
    if not code:
        return False
    return any(
        code == prefix or code.startswith(prefix + ".")
        for prefix in vetoed_codes(query)
    )

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
    "механизированным": "28.24.1",
    "сменных": "25.73.40",
    "пайки": "27.90.31.110",
    "варочные": "27.51.28",
    "офисов": "31.01.1",
    "металлическая": "31.09.11",
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
        require = BRIDGE_REQUIRE.get(token) or BRIDGE_REQUIRE.get(stem(token))
        if require and not query_stems & require:
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
