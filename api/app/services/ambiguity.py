"""Refuse an arbitrary top-1 when a query has several real interpretations.

The guard is deliberately a *post-decision* layer shared by catalog and Qdrant
backends. A high pair probability is evidence of similarity, not evidence that
one of several product types may be selected. Curated generic-name rules provide
complete, useful options for common procurement nouns; a data-driven fallback
covers other bare product words. Exact catalog titles with several OKPD2 codes
are always questions as well.

Only options issued here can later be resolved. The resolver recomputes the
question and checks the catalog hash, so a client cannot turn an arbitrary code
into a specialist-confirmed answer.
"""

from __future__ import annotations

import csv
import hashlib
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from app.core.okpd2 import OKPD2_RE
from app.schemas import (
    AttentionReason,
    Clarification,
    ClarificationOption,
    ClassifyResponse,
    OriginalDecision,
    Resolution,
)
from app.services.source_lookup import clause_url_for
from ml.features import best_match_weight, stem, stems, tokens
from ml.lexicon import CANONICAL_CODES, bridged_targets
from ml.okpd2 import is_code_ancestor, unique_detailed_code
from ml.query import split_purchase_query

# Observed procurement aliases belong to the decision layer, not to the trained
# feature space. Adding them to ml.lexicon changes every training pair and
# requires model-quality admission; these aliases only retrieve existing rows.
_NON_PRODUCT_HEADS = {stem("медицинский"), stem("комплектующие")}

_OBSERVED_BRIDGES = {
    stem("смеситель"): ("смесительная", "28.14.11.160"),
    stem("вентиль"): ("арматура", "28.14.11"),
    stem("ивл"): ("вентиляции", "32.50.21.121"),
}


def observed_bridged_targets(query: str) -> dict[str, str]:
    """Global reviewed bridges plus narrow application-level observed aliases."""
    targets = dict(bridged_targets(query))
    query_stems = set(stems(query))
    for source, (target, _code) in _OBSERVED_BRIDGES.items():
        if source in query_stems:
            targets[source] = target
    return targets


def observed_review_codes(query: str) -> set[str]:
    """Explicit one-to-one aliases eligible for a cautious review rescue."""
    query_stems = set(stems(query))
    return {
        code
        for source, (_target, code) in _OBSERVED_BRIDGES.items()
        if source in query_stems
    }


def observed_canonical_codes(query: str) -> set[str]:
    """Canonical codes for aliases active in this query."""
    targets = observed_bridged_targets(query)
    codes = {CANONICAL_CODES[target] for target in targets.values() if target in CANONICAL_CODES}
    query_stems = set(stems(query))
    codes.update(
        code
        for source, (_target, code) in _OBSERVED_BRIDGES.items()
        if source in query_stems
    )
    return codes


class AmbiguityResolutionError(ValueError):
    """Base class for safe, user-actionable resolution failures."""


class AmbiguityConflict(AmbiguityResolutionError):
    """The question, option or catalog revision is no longer current."""


@dataclass(frozen=True)
class GenericRule:
    question: str
    record_ids: tuple[str, ...]
    discriminators: str


def _rule(question: str, records: str, discriminators: str) -> GenericRule:
    return GenericRule(question, tuple(records.split()), discriminators)


# Record ids, rather than codes alone, matter when two positions share an OKPD2
# code but have different legal titles (for example general and medical lights).
# These are options, never answers. Specific modifiers disable the generic rule
# and let the ordinary measured classifier handle the now-disambiguated query.
GENERIC_RULES: dict[str, GenericRule] = {}


def _register(word: str, rule: GenericRule) -> None:
    GENERIC_RULES[stems(word)[0]] = rule


_register(
    "автомобиль",
    _rule(
        "Какой именно автомобиль закупается?",
        "1-92-01 1-94-01 1-99-01 1-101-01 1-77-01 1-96-01",
        "легковой грузовой пожарный скорая медицинский самосвал бездорожье гольф снег",
    ),
)
_register(
    "мебель",
    _rule(
        "Какого назначения и типа мебель?",
        "2-139-01 2-140-01 1-138-01 2-141-01 2-142-01 3-238-01",
        "кухонный офисный металлический деревянный пластмассовый медицинский "
        "спальня столовая гостиная",
    ),
)
_register(
    "светильник",
    _rule(
        "Какой тип светильника нужен?",
        "2-329-01 2-333-01 1-32-01 2-334-01",
        "настольный прикроватный напольный прожектор медицинский части узконаправленный",
    ),
)
_register(
    "кран",
    _rule(
        "Какой кран имеется в виду?",
        "1-40-01 1-41-01 1-42-01 1-43-01 1-44-01 1-95-01 1-108-01 3-128-01",
        "мостовой козловой полукозловой портальный гусеничный башенный стрелковый "
        "автокран манипулятор шаровой конусный цилиндрический запорный плавучий",
    ),
)
_register(
    "компьютер",
    _rule(
        "Какой тип компьютера закупается?",
        "2-198-01 2-195-01",
        "настольный портативный ноутбук планшетный томограф кассовый банкомат",
    ),
)
_register(
    "счетчик",
    _rule(
        "Что измеряет счётчик?",
        "1-28-01 2-278-01 2-279-01",
        "газ газовый жидкость вода электроэнергия электрический энергия трехфазный",
    ),
)
_register(
    "фильтр",
    _rule(
        "Для какой среды или назначения нужен фильтр?",
        "2-126-01 3-154-01 2-418-01",
        "жидкостный жидкость воздушный воздух дыхательный медицинский сетевой",
    ),
)
_register(
    "насос",
    _rule(
        "Какой тип насоса закупается?",
        "2-117-01 2-115-01 2-116-01 1-34-01",
        "центробежный возвратно-поступательный роторный объемный вакуумный воздушный "
        "компрессор части комплектующие",
    ),
)
_register(
    "клапан",
    _rule(
        "Какой тип клапана закупается?",
        "3-120-01 3-126-01 3-121-01 3-122-01 3-123-01 3-125-01",
        "редукционный запорный регулирующий обратный предохранительный отключающий",
    ),
)
_register(
    "вентиль",
    _rule(
        "Какой тип вентиля закупается?",
        "3-120-01 3-126-01 3-121-01 3-122-01 3-123-01 3-125-01",
        "редукционный запорный регулирующий обратный предохранительный отключающий",
    ),
)
_register(
    "аккумулятор",
    _rule(
        "Какой аккумулятор закупается: стартерный, прочий свинцовый или литий-ионный?",
        "1-31-01 2-190-01 3-100-01",
        "стартерный запуск свинцовый литий ионный литиевый",
    ),
)
_register(
    "шкаф",
    _rule(
        "Какого назначения и материала нужен шкаф?",
        "1-56-01 2-139-01 1-138-01 2-141-01",
        "холодильный кухонный офисный металлический деревянный медицинский вытяжной",
    ),
)
_register(
    "перчатки",
    _rule(
        "Какие перчатки закупаются: тканевые, хирургические или технические резиновые?",
        "1-3-01 3-16-01 3-17-01 3-18-01 3-19-01",
        "тканевый рабочий хирургический резиновый технический латексный стерильный",
    ),
)
_register(
    "подгузники",
    _rule(
        "Какие подгузники закупаются: детские или другое абсорбирующее бельё?",
        "2-2-01 2-362-01",
        "детский взрослый xs сверхмалый",
    ),
)
_register(
    "контейнер",
    _rule(
        "Какого назначения нужен контейнер?",
        "1-113-01 1-114-01 2-74-01 2-378-01",
        "общий универсальный специализированный биопроб полимерный упаковочный кровь",
    ),
)
_register(
    "стол",
    _rule(
        "Какого назначения нужен стол?",
        "2-139-01 2-141-01 2-320-01",
        "письменный офисный деревянный неонатальный медицинский",
    ),
)

_register(
    "генератор",
    _rule(
        "Какой генератор закупается: постоянного или переменного тока, "
        "генераторная установка или генератор сигналов?",
        "2-106-01 2-107-01 2-108-01 2-345-01",
        "постоянный переменный синхронный дизель дизельный дгу "
        "установка генераторная сигнал сигнальный",
    ),
)
_register(
    "состав",
    _rule(
        "Что закупается: подвижной состав или химический состав для обработки?",
        "1-131-01 2-59-01",
        "подвижной железнодорожный травление флюс химический катализатор",
    ),
)
_register(
    "плата",
    _rule(
        "Какие платы закупаются: печатные или звуковые/видео-/сетевые?",
        "2-193-01 2-194-01 3-35-01 3-36-01",
        "печатный смонтированный звуковой видео сетевой платформа",
    ),
)
_register(
    "труба",
    _rule(
        "Из какого материала и для какого назначения нужны трубы?",
        "2-66-01 2-71-01 2-89-01 3-252-01",
        "резиновый вулканизированный пластмассовый алюминиевый музыкальный",
    ),
)
_register(
    "камера",
    _rule(
        "Какая камера нужна: телевизионная, видео- или холодильная?",
        "2-211-01 1-57-01",
        "телевизионный видео холодильный морозильный медицинский",
    ),
)
_register(
    "тетрадь",
    _rule(
        "Какие тетради закупаются: школьные или различного назначения?",
        "2-6-01 2-7-01",
        "школьный ученический различный назначение",
    ),
)
_register(
    "принадлежности",
    _rule(
        "Для какого оборудования или назначения нужны принадлежности?",
        "2-8-01 2-79-01 2-205-01 1-66-01 3-71-01",
        "канцелярский школьный бумага картон пластмассовый вычислительный "
        "компьютерный станок звуковой видео",
    ),
)
_register(
    "телефон",
    _rule(
        "Какой телефонный аппарат нужен: сотовый или другой?",
        "2-212-01 2-213-01",
        "сотовый смартфон беспроводный проводный прочий",
    ),
)
_register(
    "трактор",
    _rule(
        "Уточните тип трактора; в перечне отдельно указаны гусеничные тракторы.",
        "1-84-01",
        "гусеничный колесный",
    ),
)
_register(
    "посуда",
    _rule(
        "Уточните материал и назначение посуды; в перечне отдельно указана пластмассовая.",
        "2-77-01",
        "пластмассовый столовый кухонный стеклянный металлический керамический",
    ),
)
_register(
    "телевизор",
    _rule(
        "Какой телевизор нужен: с записью, без записи или с LCD/плазменной панелью?",
        "2-235-01 2-236-01 3-64-01",
        "запись воспроизведение жидкокристаллический lcd плазменный",
    ),
)
_register(
    "кабель",
    _rule(
        "Какой кабель нужен: волоконно-оптический или электрический?",
        "2-326-01 3-101-01 3-102-01 2-113-01",
        "волоконно оптический индивидуальный оболочка электрический электронный проводной",
    ),
)
_register(
    "лампа",
    _rule(
        "Какая лампа нужна: электронная, осветительная, фотовспышка или щелевая?",
        "2-191-01 2-328-01 2-332-01 2-386-01",
        "электронный вакуумный газонаполненный накаливание светодиодный дуговой "
        "вспышка фотографический щелевой офтальмологический",
    ),
)
_register(
    "монитор",
    _rule(
        "Монитор нужен как периферия компьютера или как самостоятельное устройство?",
        "2-201-01 2-240-01",
        "периферийный компьютерный проектор самостоятельный",
    ),
)
_register(
    "микрофон",
    _rule(
        "Нужен отдельный микрофон или приёмная радиотелефонная аппаратура?",
        "3-68-01 2-241-01",
        "подставка радиотелефонный радиотелеграфный приемный громкоговоритель",
    ),
)
_register(
    "стеллаж",
    _rule(
        "Какой стеллаж нужен: общий металлический или медицинский?",
        "1-138-01 1-140-01",
        "металлический медицинский палата пациент эндоскоп",
    ),
)
_register(
    "кресло",
    _rule(
        "Какое кресло нужно: офисное или с санитарным оснащением?",
        "2-139-01 2-394-01",
        "офисный санитарный медицинский стул",
    ),
)
_register(
    "кондиционер",
    _rule(
        "Уточните тип кондиционера; в перечне отдельно указаны промышленные.",
        "1-55-01",
        "промышленный бытовой",
    ),
)
_register(
    "система",
    _rule(
        "Какого назначения нужна система: охранная, информационная, измерительная или медицинская?",
        "2-221-01 2-228-01 2-262-01 2-302-01 2-255-01 2-258-01",
        "тревожный сигнализация охранный информационный измерительный медицинский "
        "томография дозиметрический радиометрический спектрометрический комбинированный",
    ),
)
_register(
    "ванна",
    _rule(
        "Что требуется: ультразвуковая ванна, гидромассажное оборудование или поручень?",
        "2-359-01 2-359-02 2-389-01 2-391-01",
        "ультразвуковой гидромассажный поручень душевой",
    ),
)
_register(
    "установка",
    _rule(
        "Какого назначения нужна установка: генераторная, термообработка, "
        "аддитивная или измерительная?",
        "2-108-01 3-158-01 1-89-01 1-90-01 2-255-01",
        "генераторный электрический температурный аддитивный экструзионный "
        "струйный дозиметрический измерительный",
    ),
)

# A phrase-level family is needed when the generic head is «аппарат», but its
# modifier still leaves several real medical device types. More specific
# modifiers disable the question and let the ordinary classifier answer.
QUERY_FAMILY_RULES: list[tuple[frozenset[str], GenericRule]] = []


def _register_family(required: str, rule: GenericRule) -> None:
    QUERY_FAMILY_RULES.append((frozenset(stems(required)), rule))


@dataclass(frozen=True)
class CuratedProductRule:
    required: frozenset[str]
    record_id: str
    excluded: frozenset[str] = frozenset()


def _product_rule(required: str, record_id: str, excluded: str = "") -> CuratedProductRule:
    return CuratedProductRule(frozenset(stems(required)), record_id, frozenset(stems(excluded)))


# Unambiguous everyday product phrases whose official catalog titles use a
# broader description. These are application-layer decisions, not new training
# features: the shipped model and its feature space remain untouched.
_CURATED_PRODUCT_RULES = (
    _product_rule("нож кухонный", "2-102-01", "рыба масло"),
    _product_rule("станок токарный", "1-64-01"),
    _product_rule("набор инструмент ручной", "1-24-01", "электрический механизированный"),
    _product_rule("свитер трикотажный", "1-5-01"),
    _product_rule("беговая дорожка", "2-169-01"),
    _product_rule("пульт дистанционный управление", "2-251-01"),
    _product_rule("жесткий диск", "2-203-01"),
    _product_rule("весы промышленный", "2-127-01"),
    # Reviewed procurement formulations from real exports. Specific product
    # modifiers win over generic-head questions; unrelated lookalikes stay absent.
    _product_rule("самосвал", "1-77-01"),
    _product_rule("плата сетевая", "3-36-01"),
    _product_rule("плата звуковая", "3-36-01"),
    _product_rule("аккумулятор свинцовый стационарный", "3-99-01", "запуск стартерный"),
    _product_rule("аккумулятор varta blue dynamic", "3-98-01"),
    _product_rule("аккумулятор bosch s4", "3-98-01"),
    _product_rule("аккумулятор delta dtm", "3-99-01"),
    _product_rule("аккумулятор csb gp1272", "3-99-01"),
    _product_rule("аккумулятор makita bl1850b", "3-100-01"),
    _product_rule("видеокарта", "3-36-01"),
    _product_rule("видеоплата", "3-36-01"),
    _product_rule("шприц медицинский", "2-385-01"),
    _product_rule("плата печатная смонтированная", "2-193-01", "сетевая видео звуковая"),
    _product_rule("компрессор винтовой", "3-118-01"),
    _product_rule("датчик температуры", "2-265-01"),
    _product_rule("сервер", "2-198-01"),
    _product_rule("источник бесперебойное питание", "2-205-01"),
    _product_rule("электродвигатель асинхронный", "2-107-01"),
    _product_rule("задвижка", "3-127-01"),
    _product_rule("кран балка", "1-40-01"),
    _product_rule("кондиционер прецизионный", "1-55-01"),
    _product_rule("теплообменник", "2-124-01"),
    _product_rule("станок фрезерный", "1-64-01"),
    _product_rule("станок сверлильный", "1-65-01"),
    _product_rule("станок шлифовальный", "1-65-01"),
    _product_rule("станок ленточнопильный", "1-65-01"),
    _product_rule("экскаватор одноковшовый", "3-172-01"),
    _product_rule("каток дорожный", "3-170-01"),
    _product_rule("аппарат сварочный", "1-33-01", "неэлектрический газовый"),
    _product_rule("вагон метрополитен моторный", "1-130-01"),
    _product_rule("вагон грузовой", "3-226-01"),
    _product_rule("полувагон", "3-226-01"),
    _product_rule("аппарат ультразвуковой диагностический", "2-314-01"),
    _product_rule("аппарат узи", "2-314-01"),
    _product_rule("электрокардиограф", "2-305-01"),
    _product_rule("труба полиэтиленовая", "2-71-01"),
    _product_rule("труба пнд", "2-71-01"),
    _product_rule("камера холодильная", "1-57-01"),
    _product_rule("модуль памяти", "2-205-01"),
    _product_rule("терминал сбора данных", "2-195-01"),
    _product_rule("программный продукт", "1-146-01"),
    _product_rule("клей пва", "1-12-01"),
    _product_rule("плита eurospan", "1-8-01"),
    _product_rule("плита древесный стружечный", "1-8-01"),
    _product_rule("плита kronospan mdf", "1-9-01"),
    _product_rule("плита мдф", "1-9-01"),
    _product_rule("посудомоечная машина промышленная", "1-60-01"),
    _product_rule("посудомоечная машина abat", "1-60-01"),
)
_CURATED_ABSENT_PHRASES = tuple(
    frozenset(stems(phrase))
    for phrase in (
        "весы кухонные",
        "весы бытовые",
        "паста теплопроводная",
        "термопаста",
        "шкаф распределительный низковольтный",
        "комплект постельное белье",
        "труба стальная",
        "алюминий первичный",
        "холодильник бытовой",
        "холодильник bosch kgv36xw21r",
        "морозильник бытовой",
        "морозильник atlant",
        "посудомоечная машина бытовая",
        "посудомоечная машина bosch",
        "ротор электродвигатель перфоратор",
    )
)


_register_family(
    "ультразвуковой аппарат",
    _rule(
        "Какой ультразвуковой аппарат закупается: для сканирования, терапии или хирургии?",
        "2-314-01 2-318-01 2-380-01",
        "сканирование диагностический терапия хирургический",
    ),
)
_register_family(
    "компрессор",
    _rule(
        "Какой компрессор закупается: общий, передвижной, турбокомпрессор, поршневой или прочий?",
        "1-34-01 3-115-01 3-116-01 3-117-01 3-118-01",
        "передвижной колесный шасси турбокомпрессор поршневой объемный прочий",
    ),
)
_register_family(
    "посудомоечная машина",
    _rule(
        "Уточните тип посудомоечной машины; в перечне отдельно указаны промышленные.",
        "1-60-01",
        "промышленный бытовой",
    ),
)
_register_family(
    "ИВЛ",
    _rule(
        "Выберите позицию аппарата искусственной вентиляции лёгких.",
        "1-139-01 3-275-01",
        "контур расходный комплектующие",
    ),
)
_register_family(
    "прибор учёт",
    _rule(
        "Что измеряет прибор учёта: электроэнергию, газ или жидкость?",
        "1-28-01 2-278-01 2-279-01",
        "электроэнергия электрический энергия газ газовый жидкость вода",
    ),
)


def _norm(text: str | None) -> str:
    return " ".join((text or "").casefold().split())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse_review_flag(raw: object) -> bool | None:
    if raw is None or isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if not text:
        return None
    return text in {"true", "1", "yes", "on"}


def load_catalog_rows(path: Path) -> tuple[list[dict], str]:
    """Load and merge the same ``(name, code)`` positions as CatalogClassifier."""
    by_key: dict[tuple[str, str], dict] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for raw in csv.DictReader(handle, delimiter=";"):
            name = (raw.get("name") or "").strip()
            code = (raw.get("okpd2") or "").strip()
            if not name:
                continue
            appendix = (raw.get("appendix") or "").strip()
            position = (raw.get("position") or "").strip()
            clause = (
                f"Приложение {appendix}, позиция {position}"
                if appendix and position
                else f"Приложение {appendix}"
                if appendix
                else None
            )
            review = _parse_review_flag(raw.get("review_required"))
            key = (name, code)
            if key in by_key:
                row = by_key[key]
                if clause and clause not in (row["clause_ref"] or ""):
                    row["clause_ref"] = f"{row['clause_ref']}; {clause}"
                if review:
                    row["catalog_review_required"] = True
                continue
            by_key[key] = {
                "name": name,
                "okpd2": code,
                "record_id": (raw.get("record_id") or "").strip(),
                "clause_ref": clause,
                "code_role": (raw.get("code_role") or "").strip() or None,
                "catalog_review_required": review,
            }
    return list(by_key.values()), _sha256(path)


def _has_product_anchor(query: str, product_stem: str, title: str | None) -> bool:
    """Whether a single-noun candidate has a literal or curated lexical anchor."""
    # pymorphy maps «всех/всем» to the same stem as «весы». These function
    # words are not evidence that an antenna or a vessel is a weighing device.
    title_stems = {
        stem(token)
        for token in tokens(title or "")
        if token not in {"всех", "всем", "всего"}
    }
    if best_match_weight(product_stem, title_stems) > 0:
        return True
    targets = " ".join(observed_bridged_targets(query).values())
    return any(best_match_weight(target, title_stems) > 0 for target in stems(targets))


class AmbiguityGuard:
    def __init__(self, rows: Iterable[dict], catalog_sha256: str) -> None:
        self.catalog_sha256 = catalog_sha256
        self._by_record: dict[str, dict] = {}
        self._by_title: dict[str, list[dict]] = {}
        for source in rows:
            row = dict(source)
            record_id = (row.get("record_id") or "").strip()
            if not record_id:
                digest = hashlib.sha256(
                    f"{row.get('okpd2', '')}|{_norm(row.get('name'))}".encode()
                ).hexdigest()[:20]
                record_id = f"generated-{digest}"
                row["record_id"] = record_id
            # The first row is the stable option id. The classifier has already
            # merged repeated appendices into this row and accumulated clause_ref.
            self._by_record.setdefault(record_id, row)
            self._by_title.setdefault(_norm(row.get("name")), []).append(row)

    def _procurement_rows(self, row: dict) -> list[dict]:
        """Include applicable parent positions without importing explicit exclusions."""
        code = (row.get("okpd2") or "").strip()
        matches = []
        for candidate in self._by_record.values():
            candidate_code = (candidate.get("okpd2") or "").strip()
            if candidate_code == code:
                # One OKPD2 code can occur in several unrelated medical rows.
                # Equal code alone never merges their legal product descriptions.
                if _norm(candidate.get("name")) == _norm(row.get("name")):
                    matches.append(candidate)
                continue
            if not is_code_ancestor(candidate_code, code):
                continue
            title = (candidate.get("name") or "").casefold()
            if code in title and ("исключ" in title or "кроме" in title):
                continue
            if candidate.get("code_role") == "Исключение":
                continue
            # A decree row for one narrow product is not a blanket rule for the
            # whole parent OKPD2 branch. Require a literal product-description
            # anchor before inheriting its clause (excavator, compressor, etc.).
            stop = set(stems(
                "и в во для по на из от или не с со за исключение кроме прочие прочий "
                "соответствующий соответствующие включая включенный"
            ))
            selected_stems = set(stems(row.get("name") or "")) - stop
            candidate_stems = set(stems(candidate.get("name") or "")) - stop
            if not selected_stems & candidate_stems:
                continue
            matches.append(candidate)
        return matches or [row]

    def enrich_procurement(self, answer: ClassifyResponse) -> ClassifyResponse:
        """Preserve all legal clauses that apply to the selected detailed code."""
        if answer.status not in {"ok", "review"} or not answer.code:
            return answer
        row = self._by_record.get(next(
            (record_id for record_id, candidate in self._by_record.items()
             if candidate.get("okpd2") == answer.code
             and _norm(candidate.get("name")) == _norm(answer.title)),
            "",
        ))
        if row is None:
            row = next(
                (candidate for candidate in self._by_record.values()
                 if candidate.get("okpd2") == answer.code),
                None,
            )
        if row is None:
            return answer
        rows = self._procurement_rows(row)
        clauses = []
        for candidate in rows:
            for clause in (candidate.get("clause_ref") or "").split("; "):
                if clause and clause not in clauses:
                    clauses.append(clause)
        roles = {candidate.get("code_role") for candidate in rows}
        review_flags = {candidate.get("catalog_review_required") for candidate in rows}
        bad_roles = [role for role in roles if role != "Включён"]
        code_role = bad_roles[0] if bad_roles else "Включён"
        review_required = True if True in review_flags else (
            False if review_flags == {False} else None
        )
        reasons = list(answer.attention_reasons)
        if bad_roles and not any(reason.code == "code_exception" for reason in reasons):
            reasons.append(AttentionReason(
                code="code_exception",
                message="Родительская или выбранная позиция требует проверки роли кода.",
            ))
        if review_required and not any(
            reason.code == "catalog_review_required" for reason in reasons
        ):
            reasons.append(AttentionReason(
                code="catalog_review_required",
                message="Строку каталога нужно сверить с первоисточником.",
            ))
        return answer.model_copy(update={
            "clause_ref": "; ".join(clauses) or answer.clause_ref,
            "code_role": code_role,
            "catalog_review_required": review_required,
            "status": "review" if (bad_roles or review_required) else answer.status,
            "attention_reasons": reasons,
        })

    def _hierarchical_answer(
        self, answer: ClassifyResponse, rows: Iterable[dict]
    ) -> ClassifyResponse | None:
        """Collapse one official title only when it has a single safe code leaf."""
        if (
            len(set(stems(answer.query))) < 2
            and set(stems(answer.query)) != set(stems(answer.title))
            and not any(
                reason.code == "curated_product_match"
                for reason in answer.attention_reasons
            )
        ):
            return None
        group = [row for row in rows if row.get("okpd2")]
        if len({_norm(row.get("name")) for row in group}) != 1:
            return None
        detailed = unique_detailed_code(row["okpd2"] for row in group)
        if detailed is None:
            return None
        if any(
            row.get("code_role") != "Включён"
            or row.get("catalog_review_required") is not False
            for row in group
        ):
            return None
        selected = next(row for row in group if row["okpd2"] == detailed)
        return self._answer_from_row(
            answer,
            selected,
            "direct_title_match",
            "Выбран единственный детальный код одной официальной позиции; "
            "условия применимых приложений сохранены.",
        )

    @staticmethod
    def _original(answer: ClassifyResponse) -> OriginalDecision:
        # Guarded responses are never passed here a second time; the cast to the
        # original three-state contract is therefore intentional.
        return OriginalDecision(
            code=answer.code,
            title=answer.title,
            status=answer.status,  # type: ignore[arg-type]
            confidence=answer.confidence,
            confidence_raw=answer.confidence_raw,
            alternatives=answer.alternatives,
        )

    @staticmethod
    def _score_map(answer: ClassifyResponse) -> dict[tuple[str, str], float]:
        scores: dict[tuple[str, str], float] = {}
        if answer.code and answer.title:
            scores[(answer.code, _norm(answer.title))] = answer.confidence_raw or 0.0
        for option in answer.alternatives:
            if option.code and option.title:
                scores[(option.code, _norm(option.title))] = option.confidence
        return scores

    def _option(
        self, row: dict, scores: dict[tuple[str, str], float]
    ) -> ClarificationOption | None:
        code = (row.get("okpd2") or "").strip()
        title = (row.get("name") or "").strip()
        if not code or not title:
            return None
        return ClarificationOption(
            option_id=row["record_id"],
            code=code,
            title=title,
            confidence=round(scores.get((code, _norm(title)), 0.0), 4),
            clause_ref=row.get("clause_ref"),
            clause_url=clause_url_for(row.get("record_id")),
            code_role=row.get("code_role"),
            catalog_review_required=row.get("catalog_review_required"),
        )

    def _clarify(
        self,
        answer: ClassifyResponse,
        question: str,
        rows: Iterable[dict],
        reason: str,
    ) -> ClassifyResponse:
        scores = self._score_map(answer)
        options = []
        seen: set[tuple[str, str]] = set()
        # A curated rule or a top-3 fallback names an interpretation, not the
        # only acceptable code for it. Expand every offered title to all its
        # real catalog rows so resolving a type never hides a sibling code.
        expanded_rows = []
        for row in rows:
            expanded_rows.extend(self._by_title.get(_norm(row.get("name")), [row]))
        grouped: dict[str, list[dict]] = {}
        for row in expanded_rows:
            grouped.setdefault(_norm(row.get("name")), []).append(row)
        collapsed_rows = []
        for group in grouped.values():
            detailed = unique_detailed_code(row["okpd2"] for row in group if row.get("okpd2"))
            safe = all(
                row.get("code_role") == "Включён"
                and row.get("catalog_review_required") is False
                for row in group
            )
            if detailed is not None and safe:
                collapsed_rows.append(next(row for row in group if row["okpd2"] == detailed))
            else:
                collapsed_rows.extend(group)
        for row in collapsed_rows:
            option = self._option(row, scores)
            if option is None:
                continue
            if option.code == answer.code and _norm(option.title) == _norm(answer.title):
                # Qdrant payload is fresher than the CSV fallback. Preserve its
                # catalog facts on the matching option instead of silently
                # replacing them while the answer becomes a clarification.
                if answer.code_role is not None:
                    option.code_role = answer.code_role
                if answer.catalog_review_required is not None:
                    option.catalog_review_required = answer.catalog_review_required
            key = (option.code, _norm(option.title))
            if key in seen:
                continue
            seen.add(key)
            options.append(option)
        if len(options) < 2:
            return answer
        message = (
            "Одно название встречается в перечне с разными кодами ОКПД2."
            if reason == "multiple_codes"
            else "Общее название допускает несколько разных позиций перечня."
        )
        return ClassifyResponse(
            query=answer.query,
            code=None,
            title=None,
            confidence=0.0,
            confidence_raw=0.0,
            status="needs_clarification",
            alternatives=[],
            clarification=Clarification(
                question=question,
                options=options,
                catalog_sha256=self.catalog_sha256,
                original=self._original(answer),
            ),
            attention_reasons=[AttentionReason(code=reason, message=message)],  # type: ignore[arg-type]
            catalog_sha256=self.catalog_sha256,
            item_id=answer.item_id,
            latency_ms=answer.latency_ms,
        )

    def exact_title_answer(self, query: str) -> ClassifyResponse | None:
        """Return a deterministic answer only for a complete catalog-title match.

        Stem-sequence equality handles ordinary Russian inflection («Клапан
        запорный» / «Клапаны запорные») without turning a partial lexical match
        into an answer. Multiple codes stay provisional: ``apply`` immediately
        replaces the first row with a clarification containing all of them.
        """
        rows = self._by_title.get(_norm(query), [])
        if not rows:
            query_stems = stems(query)
            if not query_stems:
                return None
            rows = [
                row
                for title, title_rows in self._by_title.items()
                if stems(title) == query_stems
                for row in title_rows
            ]
            if len({_norm(row.get("name")) for row in rows}) != 1:
                return None
        row = next((row for row in rows if row.get("okpd2")), None)
        if row is None:
            return None
        return ClassifyResponse(
            query=query,
            code=row["okpd2"],
            title=row["name"],
            confidence=1.0,
            confidence_raw=1.0,
            status="ok",
            clause_ref=row.get("clause_ref"),
            clause_url=clause_url_for(row.get("record_id")),
            code_role=row.get("code_role"),
            catalog_review_required=row.get("catalog_review_required"),
            catalog_sha256=self.catalog_sha256,
            latency_ms=0.0,
        )

    def _review_single_option(
        self, answer: ClassifyResponse, question: str, row: dict
    ) -> ClassifyResponse:
        """Do not autopass an underspecified query with one catalogued subtype."""
        reasons = [
            reason for reason in answer.attention_reasons if reason.code != "ambiguous_name"
        ]
        reasons.insert(0, AttentionReason(code="ambiguous_name", message=question))
        return answer.model_copy(
            update={
                "code": row["okpd2"],
                "title": row["name"],
                "status": "review",
                "clause_ref": row.get("clause_ref"),
                "clause_url": clause_url_for(row.get("record_id")),
                "code_role": row.get("code_role"),
                "catalog_review_required": row.get("catalog_review_required"),
                "catalog_sha256": self.catalog_sha256,
                "alternatives": [],
                "clarification": None,
                "attention_reasons": reasons,
            }
        )

    def _answer_from_row(
        self, answer: ClassifyResponse, row: dict, reason: str, message: str
    ) -> ClassifyResponse:
        """Keep calibrated confidence but replace an unsafe statistical choice."""
        updated = answer.model_copy(
            update={
                "code": row["okpd2"],
                "title": row["name"],
                "status": "ok",
                "clause_ref": row.get("clause_ref"),
                "clause_url": clause_url_for(row.get("record_id")),
                "code_role": row.get("code_role"),
                "catalog_review_required": row.get("catalog_review_required"),
                "catalog_sha256": self.catalog_sha256,
                "alternatives": answer.alternatives,
                "attention_reasons": [AttentionReason(code=reason, message=message)],
            }
        )
        return self.enrich_procurement(updated)

    def _curated_product_answer(
        self, query: str, answer: ClassifyResponse
    ) -> ClassifyResponse | None:
        query_stems = set(stems(query))
        for rule in _CURATED_PRODUCT_RULES:
            if not rule.required <= query_stems or query_stems & rule.excluded:
                continue
            row = self._by_record.get(rule.record_id)
            if row is None or not row.get("okpd2"):
                return None
            updated = self._answer_from_row(
                answer,
                row,
                "curated_product_match",
                "Ответ определён по проверенному правилу товарной позиции.",
            )
            same_title = self._by_title.get(_norm(updated.title), [])
            return self._hierarchical_answer(updated, same_title) or updated
        return None

    def _specific_title_answer(
        self, query: str, answer: ClassifyResponse
    ) -> ClassifyResponse | None:
        """Resolve a specified official-title prefix, or expose competing titles.

        Statistical retrieval can choose «музыкальные трубы» for «трубы
        алюминиевые» even though the discriminator occurs in exactly one catalog
        title. A unique stem-subset match is deterministic. Prefix matches are
        stronger: when several titles remain («мебель металлическая»), the user
        must choose instead of letting the backend select different codes.
        """
        query_stems = stems(query)
        if len(query_stems) < 2:
            return None
        exact_rows = self._by_title.get(_norm(query), [])
        exact_codes = {row["okpd2"] for row in exact_rows if row.get("okpd2")}
        if _norm(answer.title) == _norm(query) and len(exact_codes) == 1:
            return answer
        query_set = set(query_stems)
        prefixed: list[dict] = []
        subset: list[dict] = []
        for title, title_rows in self._by_title.items():
            title_stems = stems(title)
            if not title_stems or title_stems[0] != query_stems[0]:
                continue
            if title_stems[: len(query_stems)] == query_stems:
                prefixed.extend(title_rows)
            elif query_set <= set(title_stems):
                subset.extend(title_rows)
        rows = prefixed
        if not rows and len({_norm(row["name"]) for row in subset}) == 1:
            rows = subset
        if not rows:
            return None
        titles = {_norm(row["name"]) for row in rows}
        codes = {row["okpd2"] for row in rows if row.get("okpd2")}
        if len(titles) > 1 or len(codes) > 1:
            hierarchical = self._hierarchical_answer(answer, rows)
            if hierarchical is not None:
                return hierarchical
            guarded = self._clarify(
                answer,
                "Уточните официальную позицию и код закупаемого товара.",
                rows,
                "multiple_codes" if len(titles) == 1 else "ambiguous_name",
            )
            return guarded if guarded.status == "needs_clarification" else None
        row = next((row for row in rows if row.get("okpd2")), None)
        if row is None:
            return None
        return self._answer_from_row(
            answer,
            row,
            "direct_title_match",
            "Ответ определён по конкретному официальному названию перечня.",
        )

    def apply(self, query: str, answer: ClassifyResponse) -> ClassifyResponse:
        if answer.status == "needs_clarification" or answer.decision_source == "specialist":
            return answer
        # An explicit OKPD2 code is itself the disambiguator.
        if OKPD2_RE.fullmatch(query.strip().replace(" ", "")):
            return answer
        literal_rows = self._by_title.get(_norm(query), [])
        literal_codes = {row["okpd2"] for row in literal_rows if row.get("okpd2")}
        if len(literal_codes) > 1:
            hierarchical = self._hierarchical_answer(answer, literal_rows)
            if hierarchical is not None:
                return hierarchical
        if (
            len(literal_codes) == 1
            and _norm(answer.title) == _norm(query)
            and answer.code in literal_codes
        ):
            # A complete official single-code title is itself specific enough
            # («Трубы» as a musical instrument). Generic wording and all
            # multi-code titles still go through the guards below.
            return answer

        head, _ = split_purchase_query(query)
        head_stems = set(stems(head))
        if len(head_stems) == 1 and next(iter(head_stems)) in _NON_PRODUCT_HEADS:
            # A standalone adjective names no purchasable product. Offering
            # refrigerators for «медицинские» is fluent but meaningless.
            return answer.model_copy(
                update={
                    "code": None,
                    "title": None,
                    "status": "not_found",
                    "confidence": 0.0,
                    "confidence_raw": 0.0,
                    "alternatives": [],
                    "clarification": None,
                }
            )
        query_stems = set(stems(query))
        if any(required <= query_stems for required in _CURATED_ABSENT_PHRASES):
            # The catalog lists industrial weighing equipment only. A household
            # modifier must not inherit that code (or a morphological collision
            # with kitchen furniture).
            return answer.model_copy(
                update={
                    "code": None,
                    "title": None,
                    "status": "not_found",
                    "confidence": 0.0,
                    "confidence_raw": 0.0,
                    "alternatives": [],
                    "clarification": None,
                    "clause_ref": None,
                    "clause_url": None,
                    "code_role": None,
                    "catalog_review_required": None,
                    "attention_reasons": [],
                }
            )
        curated = self._curated_product_answer(query, answer)
        if curated is not None:
            return curated

        # A brand or model does not disambiguate the product type. Apply the
        # generic rule to the leading product noun even when specifications follow.
        product_head = next(iter(stems(head)), "")
        rule = GENERIC_RULES.get(product_head)
        answer_title_codes = {
            row["okpd2"]
            for row in self._by_title.get(_norm(answer.title), [])
            if row.get("okpd2")
        }
        expanded_generic = len(head_stems) > 1 and (
            answer.status != "ok" or len(answer_title_codes) > 1
        )
        if (
            not literal_rows
            and rule is not None
            and (len(head_stems) == 1 or expanded_generic)
            and not (query_stems & set(stems(rule.discriminators)))
        ):
            rows = [
                self._by_record[record_id]
                for record_id in rule.record_ids
                if record_id in self._by_record
            ]
            guarded = self._clarify(answer, rule.question, rows, "ambiguous_name")
            if guarded.status == "needs_clarification":
                return guarded
            if len(rows) == 1:
                return self._review_single_option(answer, rule.question, rows[0])

        for required_stems, family_rule in QUERY_FAMILY_RULES:
            if not required_stems <= query_stems:
                continue
            if query_stems & set(stems(family_rule.discriminators)):
                continue
            rows = [
                self._by_record[record_id]
                for record_id in family_rule.record_ids
                if record_id in self._by_record
            ]
            guarded = self._clarify(
                answer, family_rule.question, rows, "ambiguous_name"
            )
            if guarded.status == "needs_clarification":
                return guarded
            if len(rows) == 1:
                return self._review_single_option(answer, family_rule.question, rows[0])

        specific = self._specific_title_answer(query, answer)
        if specific is not None:
            return specific

        # A title can have several codes even when the model is highly
        # confident. Looking only at its top-3 alternatives would hide this.
        # Check both the literal request and the selected title: procurement
        # queries often use a short product name («Шприц») or a different word
        # order («Самосвал» → «Автомобили-самосвалы…»).
        multiple_code_question = (
            "Одно название указано в перечне с разными кодами. "
            "Какой код и пункт требуются?"
        )
        rows = self._by_title.get(_norm(head), [])
        codes = {(row.get("okpd2") or "").strip() for row in rows if row.get("okpd2")}
        if len(codes) >= 2:
            hierarchical = self._hierarchical_answer(answer, rows)
            if hierarchical is not None:
                return hierarchical
            guarded = self._clarify(
                answer, multiple_code_question, rows, "multiple_codes"
            )
            if guarded.status == "needs_clarification":
                return guarded

        # A specific short query can identify a long multi-code title even if
        # retrieval chose another family («Светильники медицинские» → household
        # lights). Match all its content stems against the catalog, not only the
        # already selected answer. Phrase-level families above take precedence
        # so a broad ultrasound query is not narrowed to surgical devices.
        if len(head_stems) >= 2:
            matching_rows = []
            for title, title_rows in self._by_title.items():
                title_codes = {
                    (row.get("okpd2") or "").strip()
                    for row in title_rows
                    if row.get("okpd2")
                }
                if (
                    len(title_codes) >= 2
                    and head_stems <= set(stems(title))
                    and next(iter(stems(title)), None) == next(iter(stems(head)), None)
                ):
                    matching_rows.extend(title_rows)
            if matching_rows:
                hierarchical = self._hierarchical_answer(answer, matching_rows)
                if hierarchical is not None:
                    return hierarchical
                guarded = self._clarify(
                    answer, multiple_code_question, matching_rows, "multiple_codes"
                )
                if guarded.status == "needs_clarification":
                    return guarded

        selected_rows = self._by_title.get(_norm(answer.title), [])
        selected_codes = {
            (row.get("okpd2") or "").strip() for row in selected_rows if row.get("okpd2")
        }
        if len(selected_codes) >= 2:
            # Do not turn an unrelated weak top-1 into a question. Requiring
            # the primary product stem from the query head in the official
            # title covers inflection, word order and extra specifications
            # without treating any multi-code retrieval result as a match.
            answer_title_stems = set(stems(answer.title or ""))
            primary_product_stem = next(iter(stems(head)), None)
            if primary_product_stem in answer_title_stems:
                hierarchical = self._hierarchical_answer(answer, selected_rows)
                if hierarchical is not None:
                    return hierarchical
                guarded = self._clarify(
                    answer, multiple_code_question, selected_rows, "multiple_codes"
                )
                if guarded.status == "needs_clarification":
                    return guarded

            # Even an unrelated candidate with several legal codes must never
            # look automatically accepted. Keep a weak candidate reviewable,
            # but expose the catalog conflict to the common report service.
            reasons = list(answer.attention_reasons)
            if not any(reason.code == "multiple_codes" for reason in reasons):
                reasons.append(
                    AttentionReason(
                        code="multiple_codes",
                        message="Одно название встречается в перечне с разными кодами ОКПД2.",
                    )
                )
            return answer.model_copy(
                update={
                    "status": "review" if answer.status == "ok" else answer.status,
                    "attention_reasons": reasons,
                }
            )

        # Generic fallback for unregistered product nouns. It deliberately works
        # only on a single content stem and literal candidate anchors: dense
        # false friends without the product word cannot create a question.
        # The labeling doctrine explicitly accepts canonical parent categories
        # («Обувь» → 15.2, «Одежда» → 14.1). Curated rules above intentionally
        # override this legacy canon for the new ambiguity registry.
        if len(head_stems) != 1:
            return answer
        product_stem = next(iter(head_stems))
        if any(
            stems(word) == [product_stem] and answer.code == code
            for word, code in CANONICAL_CODES.items()
        ):
            return answer
        anchored = []
        if answer.code and answer.title and next(iter(stems(answer.title)), None) == product_stem:
            anchored.append(
                {
                    "okpd2": answer.code,
                    "name": answer.title,
                    "record_id": "",
                    "clause_ref": answer.clause_ref,
                    "code_role": answer.code_role,
                    "catalog_review_required": answer.catalog_review_required,
                }
            )
        for alternative in answer.alternatives:
            if (
                alternative.code
                and alternative.title
                and next(iter(stems(alternative.title)), None) == product_stem
            ):
                if alternative.confidence >= 0.08:
                    anchored.append(
                        {
                            "okpd2": alternative.code,
                            "name": alternative.title,
                            "record_id": "",
                            "clause_ref": alternative.clause_ref,
                        }
                    )
        if len({(row["okpd2"], _norm(row["name"])) for row in anchored}) < 2:
            if answer.status in {"ok", "review"} and not _has_product_anchor(
                query, product_stem, answer.title
            ):
                # A single noun with no literal or curated relation to the
                # candidate must not receive a plausible-looking review/ok
                # («Весы» → antennas). Multi-word semantic queries continue
                # through the measured dense/lexical classifier unchanged.
                alternatives = [
                    alternative
                    for alternative in answer.alternatives
                    if _has_product_anchor(query, product_stem, alternative.title)
                ]
                return ClassifyResponse(
                    query=answer.query,
                    code=None,
                    title=None,
                    confidence=0.0,
                    confidence_raw=0.0,
                    status="not_found",
                    alternatives=alternatives,
                    catalog_sha256=self.catalog_sha256,
                    item_id=answer.item_id,
                    latency_ms=answer.latency_ms,
                )
            return answer
        # Resolve generated options back to real catalog rows by title+code; if
        # they are not in the active catalog they must not become resolvable.
        rows = []
        for candidate in anchored:
            match = next(
                (
                    row
                    for row in self._by_title.get(_norm(candidate["name"]), [])
                    if row.get("okpd2") == candidate["okpd2"]
                ),
                None,
            )
            if match is not None:
                rows.append(match)
        return self._clarify(
            answer,
            "Уточните тип товара: одно общее слово покрывает несколько позиций перечня.",
            rows,
            "ambiguous_name",
        )

    def resolve_answer(
        self, answer: ClassifyResponse, option_id: str, expected_catalog_sha256: str
    ) -> ClassifyResponse:
        if expected_catalog_sha256 != self.catalog_sha256:
            raise AmbiguityConflict(
                "Каталог изменился. Запустите классификацию повторно и выберите новое уточнение."
            )
        if answer.status != "needs_clarification" or answer.clarification is None:
            raise AmbiguityConflict("Для этого запроса сейчас нет активного вопроса уточнения.")
        if answer.clarification.catalog_sha256 != self.catalog_sha256:
            raise AmbiguityConflict(
                "Вопрос построен на другой версии каталога. Повторите проверку."
            )

        # A single-query response is client input on the resolve call. Never
        # trust its options or audit metadata: rebuild the pending decision from
        # the original model answer and take the option from that fresh question.
        # Batch answers follow the same path, so both transports have identical
        # authorization rules rather than relying on a hidden server-side job.
        original = answer.clarification.original
        original_answer = ClassifyResponse(
            query=answer.query,
            code=original.code,
            title=original.title,
            confidence=original.confidence,
            confidence_raw=original.confidence_raw,
            status=original.status,
            alternatives=original.alternatives,
            item_id=answer.item_id,
            latency_ms=answer.latency_ms,
        )
        current = self.apply(answer.query, original_answer)
        if current.status != "needs_clarification" or current.clarification is None:
            raise AmbiguityConflict("Для этого запроса сейчас нет активного вопроса уточнения.")
        option = next(
            (item for item in current.clarification.options if item.option_id == option_id),
            None,
        )
        if option is None:
            raise AmbiguityConflict("Выбранный вариант не относится к этому запросу.")
        row = self._by_record.get(option_id)
        if (
            row is None
            or row.get("okpd2") != option.code
            or _norm(row.get("name")) != _norm(option.title)
        ):
            raise AmbiguityConflict("Вариант не найден в активном каталоге. Повторите проверку.")
        blocking = option.code_role != "Включён" or option.catalog_review_required is not False
        reasons = []
        if option.code_role != "Включён":
            reasons.append(
                AttentionReason(
                    code="code_exception",
                    message=(
                        "Выбранная позиция является исключением или имеет "
                        "неподтверждённую роль кода."
                    ),
                )
            )
        if option.catalog_review_required is not False:
            reasons.append(
                AttentionReason(
                    code="catalog_review_required",
                    message="Строку каталога нужно дополнительно сверить с первоисточником.",
                )
            )
        resolved = ClassifyResponse(
            query=answer.query,
            code=option.code,
            title=option.title,
            confidence=option.confidence,
            confidence_raw=option.confidence,
            status="review" if blocking else "ok",
            clause_ref=option.clause_ref,
            clause_url=option.clause_url,
            code_role=option.code_role,
            catalog_review_required=option.catalog_review_required,
            decision_source="specialist",
            resolution=Resolution(
                option_id=option_id,
                resolved_at=datetime.now(UTC).isoformat(),
                original=current.clarification.original,
            ),
            attention_reasons=reasons,
            catalog_sha256=self.catalog_sha256,
            item_id=answer.item_id,
            latency_ms=answer.latency_ms,
        )
        return self.enrich_procurement(resolved)
