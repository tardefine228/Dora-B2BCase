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

Edit this file and rebuild the model (``python scripts/build_model.py``) so the
reranker learns the bridged queries too.
"""

from __future__ import annotations

from ml.features import stems

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
    "люстра": "светильники",
    "торшер": "светильники",
    # Ткани (13.2)
    "ткань": "ткани",
    "ситец": "ткани",
    "сатин": "ткани",
    "бязь": "ткани",
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
}

# Reverse index: catalog word -> source words that map to it.
_BY_TARGET: dict[str, list[str]] = {}
for _source, _target in BRIDGES.items():
    _BY_TARGET.setdefault(_target, []).append(_source)


def expand_query(query: str) -> str:
    """Append category words for recognized product words, keeping the query text."""
    query_stems = set(stems(query))
    additions = [
        target
        for source, target in BRIDGES.items()
        if source in query_stems or stems(source)[0] in query_stems
    ]
    if not additions:
        return query
    return f"{query} {' '.join(sorted(set(additions)))}"


def sources_for_target_word(target_word: str) -> list[str]:
    """Product words that bridge into a catalog word (used by training)."""
    return _BY_TARGET.get(target_word, [])
