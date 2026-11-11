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
from ml.query import split_purchase_query


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
        "холодильный офисный металлический деревянный медицинский вытяжной",
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

# A phrase-level family is needed when the generic head is «аппарат», but its
# modifier still leaves several real medical device types. More specific
# modifiers disable the question and let the ordinary classifier answer.
QUERY_FAMILY_RULES: list[tuple[frozenset[str], GenericRule]] = []


def _register_family(required: str, rule: GenericRule) -> None:
    QUERY_FAMILY_RULES.append((frozenset(stems(required)), rule))


_register_family(
    "ультразвуковой аппарат",
    _rule(
        "Какой ультразвуковой аппарат закупается: для сканирования, терапии или хирургии?",
        "2-314-01 2-318-01 2-380-01",
        "сканирование диагностический терапия хирургический",
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
    targets = " ".join(bridged_targets(query).values())
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
        for row in expanded_rows:
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

    def apply(self, query: str, answer: ClassifyResponse) -> ClassifyResponse:
        if answer.status == "needs_clarification" or answer.decision_source == "specialist":
            return answer
        # An explicit OKPD2 code is itself the disambiguator.
        if OKPD2_RE.fullmatch(query.strip().replace(" ", "")):
            return answer

        head, _ = split_purchase_query(query)
        head_stems = set(stems(head))
        query_stems = set(stems(query))
        rule = GENERIC_RULES.get(next(iter(head_stems), "")) if len(head_stems) == 1 else None
        if rule is not None and not (query_stems & set(stems(rule.discriminators))):
            rows = [
                self._by_record[record_id]
                for record_id in rule.record_ids
                if record_id in self._by_record
            ]
            guarded = self._clarify(answer, rule.question, rows, "ambiguous_name")
            if guarded.status == "needs_clarification":
                return guarded

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
        return ClassifyResponse(
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
