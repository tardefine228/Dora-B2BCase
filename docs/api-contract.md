# Контракт API v1.0

Формат результата классификации, согласованный с фронтендом (`frontend/src`)
и закреплённый тестами `api/tests/test_contract.py`. **Правила версионирования:**
в рамках v1.x поля только добавляются (опциональные с дефолтом); переименование
или удаление поля — только в v2 и с одновременным обновлением фронтенда.
Поле `contract_version` присутствует в каждом ответе `/classify`.

## Классификация

### `POST /classify`

Запрос:

```json
{ "name": "Ноутбук Lenovo IdeaPad" }
```

Ответ `ClassifyResponse`:

```json
{
  "query": "Ноутбук Lenovo IdeaPad",
  "code": "26.20.11",
  "title": "Компьютеры портативные массой не более 10 кг, такие как ноутбуки, ...",
  "confidence": 0.88,
  "confidence_raw": 0.46,
  "status": "review",
  "code_role": "Включён",
  "catalog_review_required": false,
  "clause_ref": "Приложение 2, позиция 195; Приложение 3, позиция 37",
  "clause_url": "/source/2-195-01",
  "alternatives": [
    { "code": "26.20.17", "title": "Мониторы и проекторы ...", "confidence": 0.07, "clause_ref": "Приложение 2, позиция 174", "clause_url": "/source/2-201-01" }
  ],
  "latency_ms": 1.2,
  "contract_version": "1.0"
}
```

| Поле | Тип | Семантика |
| --- | --- | --- |
| `query` | string | исходный запрос (с пробелами, как прислал клиент) |
| `code` | string\|null | вероятный код ОКПД2; `null` при `not_found` |
| `title` | string\|null | наименование позиции перечня; `null` при `not_found` |
| `confidence` | float 0..1 | уверенность модели в **top-1**, откалиброванная по реальному трафику (что показывать) |
| `confidence_raw` | float 0..1 | сырая уверенность до калибровки — **по ней принимается `status`**; при `not_found` — `0.0` |
| `status` | enum | `ok` — уверенная подсказка; `review` — проверить специалисту; `not_found` — надёжного совпадения нет |
| `code_role` | string\|null | роль кода в каталоге найденной позиции (`"Включён"`, `"Не указан"`, `"Исключение"`); `null` — каталог не говорит (нет кода / mock-режим) |
| `catalog_review_required` | bool\|null | признак «требуется подтверждение» из строки каталога найденной позиции; `null` — каталог не говорит |
| `clause_ref` | string\|null | где в перечне: «Приложение N, позиция M» (для дублей — перечисление) |
| `clause_url` | string\|null | ссылка на источник: `GET /source/{record_id}` — фрагмент `data/source_text.txt` со страницами позиции; `null`, если позиция неизвестна (mock-режим) |
| `verifier` | string\|null | кто подтвердил код, если ответ пришёл от проверяющей ступени: `"llm-top3"` — модель языка выбрала одну из трёх предложенных позиций; `null` — ответ дан классификатором (поле добавлено в 1.0 как необязательное, эксп. 32) |
| `verifier_note` | string\|null | короткая пометка ступени: «код подтверждён по top-3» или «нет совпадений»; не юридическое заключение |
| `alternatives` | array ≤3 | альтернативные позиции; `confidence` — оценка соответствия пары запрос/позиция; каждая несёт свои `clause_ref`/`clause_url` и флаги каталога |
| `latency_ms` | float ≥0 | время обработки на сервере |
| `contract_version` | string | версия контракта (`"1.0"`) |

**Инварианты** (проверяются тестами):

1. `status = "not_found"` ⇒ `code = null`, `title = null`, `confidence = confidence_raw = 0.0`.
2. `0.0 ≤ confidence ≤ 1.0`; уверенность — подсказка, не юридический вердикт.
3. Прямая передача кода ОКПД2 в `name` ⇒ `confidence = confidence_raw = 1.0`, `status = "ok"`.
4. `status` определяется только `confidence_raw`; `confidence` — монотонная (Платт) калибровка
   сырой шкалы под реальные запросы, поэтому статусы от неё не зависят
   (`python scripts/build_model.py` фитит карту на калибровочной площадке,
   метрики — `eval/experiments.md`, эксп. 10).
5. `status in {ok, review}` ⇒ `code` и `title` непустые (BE-9). Инвариант
   зашит валидатором `ClassifyResponse` (`api/app/schemas.py`) и покрыт
   `api/tests/test_contract_invariants.py`; нарушение — дефект сервиса, а не
   допустимый ответ.

   Одно исключение из инварианта 4, измеренное и узкое (эксп. 34): если модель
   посчитала совпадение ниже порога, но позиция несёт **головное слово строки**
   в своём названии («Осциллограф цифровой запоминающий» → «Осциллоскопы и
   осциллографы…») и оценка пары ≥ 0.08, ответ отдаётся как `review`, а не
   `not_found`. Причина: строка закупки почти всегда несёт спецификаторы,
   которых в перечне нет, и каждое такое слово снижает оценку пары вдвое-втрое.
   Правило не выдаёт `ok`, требует код и не срабатывает, если головное слово —
   модификатор области («Сейф офисный» ≠ «Машины офисные прочие»); пустых
   ответов на товары вне перечня оно не добавляет (`api/tests/test_head_rescue.py`,
   `api/tests/test_calibration.py`). Правило вынесено в общий помощник
   `_head_rescued_answer` и применяется обоими бэкендами — каталожным и Qdrant:
   иначе прод с векторным поиском отвечал бы «нет в перечне» на строках, которые
   каталожный режим уже показывает как `review` (`api/tests/test_qdrant_dedup.py`).

6. `verifier` заполнен ⇒ `status = "review"` и `code` — один из кодов, которые
   были предложены ступени. Проверяющая ступень (LLM) **не может** вернуть `ok`
   и **не может** назвать код, которого нет в шорт-листе: такой ответ
   отбрасывается, и результат остаётся прежним (`api/tests/test_llm_verifier.py`).

### Флаги каталога (`code_role`, `catalog_review_required`)

Каталог несёт два факта о позиции перечня: роль кода (`code_role` из колонки
`code_role` — «Включён» / «Не указан» / «Исключение») и признак
`review_required` («позицию нужно подтвердить»). Сервис отдаёт их **как есть,
информационно**: они не меняют `status`, `confidence` и порядок альтернатив.
Что делать с позициями-«Исключениями» (DA-6) и строками `review_required`
(DA-7/PO-1) — продуктовое решение каталога; до него API не фильтрует и не
демотирует такие позиции по флагам. Единственное исключение: если top-1
найден **без кода** (в строке каталога код не указан), ответ демотируется до
`not_found` по инварианту 5, а позиция без кода предлагается первой
альтернативой с `code: null` — специалист видит подсказку и решает сам.
`null` в обоих флагах означает «каталог не говорит» (пустая ячейка CSV
экспорта — то же самое).

### `POST /classify/batch` → `GET /classify/batch/{job_id}`

```json
{ "items": ["до 100 названий, ≤500 символов каждое"] }
```

`202 Accepted` → `{job_id, status:"queued", total, status_url, contract_version:"1.0"}`.
Тот же `contract_version` есть в оболочке ответа `GET status_url` и у принятого
`/classify/file`; он идентифицирует версию batch-формата. Опрос `status_url`
каждые ~300–500 мс:

```json
{ "job_id": "…", "status": "queued|running|completed|failed",
  "total": 2, "completed": 1, "progress_percent": 50.0,
  "results": [ ClassifyResponse, … ], "error": null,
  "contract_version": "1.0" }
```

`results` заполняются по мере обработки (порядок соответствует `items`).

### `GET /classify/batch/{job_id}/events` — прогресс по SSE

`text/event-stream` с событиями:

```text
event: progress
data: {"job_id": "...", "status": "running", "completed": 2, "total": 5, "progress_percent": 40.0}

event: done
data: {"job_id": "...", "status": "completed", "total": 5, "completed": 5,
       "progress_percent": 100.0, "results": [ ClassifyResponse, ... ], "error": null,
       "contract_version": "1.0"}
```

Поток закрывается после `done`; раз в 15 с простоя приходит комментарий
`: keep-alive`. Заголовки: `Cache-Control: no-cache`, `X-Accel-Buffering: no`
(прокси не буферизует). Неизвестный `job_id` → 404 до начала стрима. Источник
истины — хранилище задач, поэтому поток продолжается после рестарта API, если
`JOB_STORE=sqlite`.

### `GET /classify/batch/{job_id}.csv` — экспорт результата

`Content-Disposition: attachment; filename="batch-{job_id}.csv"`,
`Content-Type: text/csv; charset=utf-8`, разделитель `;`, CRLF и BOM — файл
открывается в Excel на русской локали двойным щелчком. Колонки:

```text
query;code;title;status;confidence;confidence_raw;clause_ref;clause_url;
code_role;catalog_review_required;
alt1_code;alt1_title;alt2_code;alt2_title;alt3_code;alt3_title
```

Кавычки внутри значений удваиваются, поля с `;`/кавычками/переводами строк
экранируются. `confidence` и `confidence_raw` записываются с четырьмя знаками
после запятой; если `confidence_raw` отсутствует в сохранённом результате,
ячейка остаётся пустой. `code_role` пуст, когда каталог не говорит роль кода;
`catalog_review_required` — пустая ячейка (`null`), `True` или `False`.
Для задачи в статусе `queued`/`running` — 409, для неизвестного `job_id` — 404;
у `failed` отдаются частичные результаты.

### `GET /source/{record_id}`

Источник позиции перечня. В каталоге `pp_link` — это цитата, а не URL, поэтому
сервис отдаёт страницы из `data/source_text.txt`, записанные в строке каталога
(`pdf_page_start`/`pdf_page_end`):

```json
{
  "record_id": "2-339-01",
  "name": "Электрочайники",
  "okpd2": "27.51.24.110",
  "clause_ref": "Приложение 2, позиция 339",
  "pages": { "start": 64, "end": 64 },
  "text": "…339. Электрочайники 27.51.24.110…",
  "truncated": false,
  "source_file": "source_text.txt",
  "clause_url": "/source/2-339-01"
}
```

`text` обрезается до 20 000 символов (`truncated: true`). Ответ кэшируется
(`Cache-Control: public, max-age=3600`).

## Административные (заголовок `X-Admin-API-Key`)

| Метод | Путь | Назначение |
| --- | --- | --- |
| POST | `/admin/catalog` | загрузка/проверка каталога (`file`, `dry_run`, `auto_reindex`) |
| GET | `/admin/catalog` | факты об установленном каталоге (строки, sha256, время) |
| POST | `/admin/reindex` | переиндексация Qdrant (фоновая задача, 202 + `status_url`) |
| GET | `/admin/reindex/{id}` | статус переиндексации |

Подробности загрузки каталога — `docs/catalog-update.md`.

## Ошибки

| Код | Когда | Формат `detail` |
| --- | --- | --- |
| 403 | нет/неверный `X-Admin-API-Key` | строка |
| 404 | неизвестная задача | строка |
| 409 | переиндексация уже идёт | строка |
| 413 | тело больше `MAX_REQUEST_BODY_BYTES` | строка |
| 422 | валидация запроса/файла | список ошибок pydantic или строка |
| 404 | неизвестный `record_id` в `/source/{id}` | строка |
| 429 | переполнение очереди фоновых задач или rate limit (`Retry-After`) | строка |
| 503 | ML-режим выключен/не собран, классификатор не инициализирован, недоступен `source_text.txt` | строка |

Все ответы несут заголовки `X-Request-ID` (прокидывается из запроса) и
`X-Process-Time-Ms`. CORS настраивается `CORS_ALLOW_ORIGINS`.

### Rate limit

Публичные эндпоинты ограничены token bucket'ом на клиента. Клиент — адрес сокета;
`X-Forwarded-For` учитывается только для прокси из `TRUSTED_PROXIES`, и берётся
правый непроверенный хоп (тот, что дописал ваш прокси): `RATE_LIMIT_RPS`/`RATE_LIMIT_BURST`
для обычных запросов, `RATE_LIMIT_FILE_RPS`/`RATE_LIMIT_FILE_BURST` — для
`POST /classify/file` (считается в файлах: одна загрузка — до 100 позиций)
(по умолчанию 10/20), у `/admin/*` — отдельный строгий лимит
(`RATE_LIMIT_ADMIN_RPS`/`_BURST`, по умолчанию 1/2). `/health*`, `/metrics` и
статика `/ui` не лимитируются; лимит выключается `RATE_LIMIT_ENABLED=false`.

```http
HTTP/1.1 429 Too Many Requests
Retry-After: 1
X-RateLimit-Limit: 20
X-RateLimit-Remaining: 0

{"detail": "Слишком много запросов: превышен лимит 10 зап/с (burst 20). Повторите позже."}
```

Успешные ответы несут `X-RateLimit-Limit` и `X-RateLimit-Remaining`; отказы
считаются в `GET /metrics` (`dora_http_rate_limited_total{path}`).

## Наблюдаемость (`/health`, `/metrics`)

Системные эндпоинты не лимитируются и не требуют ключей.

### `GET /health` — liveness

Дешёвая проверка «процесс жив»: `{status: "ok", service, classifier_backend,
version}`. Не грузит модели и не требует Qdrant — подходит для liveness-проба
оркестратора.

### `GET /health/ready` — readiness

200, когда сконфигурированный классификатор инициализирован; иначе 503 с
`{status: "not_ready", ...}`. Ответ 200:

```json
{ "status": "ready", "service": "dora-api", "classifier_backend": "catalog",
  "version": "…", "vector_search": { "enabled": true, "embedder": "…", "dim": 512,
  "store": "qdrant", "positions": 1126 },
  "vector_health": { "vector_search_enabled": true, "embedder_up": true,
  "store_up": true, "degraded": false, "embedder": "…", "dim": 512,
  "store": "qdrant", "positions": 1126 },
  "job_store": "memory", "queue_backend": "inprocess" }
```

`vector_health` (BE-6) — живой статус эмбеддера и векторного хранилища:
TTL-кэш (пинг Qdrant не чаще раза на интервал, таймаут 2 с), информационный —
деградация хранилища не делает сервис неготовым (поиск уходит в память), но
`degraded: true` обязан быть виден оператору. `store_up: false` означает, что
запрошенное хранилище (Qdrant) не обслуживает поиск, даже если зеркало упало
в память и запросы продолжают работать.

### `GET /metrics` — метрики

По умолчанию отдаётся **текстовый формат Prometheus** (`text/plain;
version=0.0.4; charset=utf-8`) — скрейпится `prometheus`/`victoriametrics` без
экспортёров. Клиенты с `Accept: application/json` получают легаси-снимок
(in-process счётчики + `vector_health`) — формат для отладки, не контракт.

Счётчики и гистограмма (labels в фигурных скобках):

| Метрика | Тип | Labels | Смысл |
| --- | --- | --- | --- |
| `dora_http_requests_total` | counter | `path`, `method`, `status` | HTTP-ответы по путям/методам/кодам |
| `dora_http_request_duration_seconds` | histogram | `path` | латентность ответов (корзины 5 мс … 30 с) |
| `dora_http_rate_limited_total` | counter | `path` | отказы 429 по путям |
| `dora_classifications_total` | counter | `mode` (`single`/`batch`), `backend`, `status` | ответы классификации по статусам |
| `dora_batch_jobs_total` | counter | `event` (`accepted`/`completed`/`failed`) | жизненный цикл batch-задач |

Гейджи (обновляются на каждый скрейп):

| Метрика | Labels | Смысл |
| --- | --- | --- |
| `dora_classifier_up` | — | 1, если классификатор инициализирован |
| `dora_embedder_up` | — | 1, если эмбеддер загружен |
| `dora_embedder_info` | `tag` | 1 + тег активного эмбеддера |
| `dora_embedder_dim` | — | размерность вектора |
| `dora_vector_search_enabled` | — | 1, если векторный слой фактически работает |
| `dora_vector_store_up` | — | 1, если запрошенное хранилище обслуживает поиск |
| `dora_vector_store_info` | `store` | 1 + фактическое хранилище (`qdrant`/`memory`) |
| `dora_vector_positions` | — | число проиндексированных позиций |
| `dora_build_info` | `version`, `classifier_backend`, `job_store`, `queue_backend` | 1 — информация о сборке/конфигурации |

Ограничение: счётчики живут **в процессе API**. При `QUEUE_BACKEND=celery`
инференс исполняется в процессе воркера (`celery -A app.worker.celery_app`),
который HTTP-эндпоинтов не отдаёт: `dora_classifications_total` и
`dora_batch_jobs_total{event="accepted"}` в API отражают приём задач, а
`completed`/`failed` по факту обработки воркером в `/metrics` API не видны.
Прогресс и итоги таких задач — через `GET /classify/batch/{job_id}` (общее
хранилище задач). Сквозные счётчики воркера — отдельная задача (см. OPS-6
аудита: Prometheus/Grafana/алерты).
## Обратная связь по результатам (добавлено для фронтенда)

`POST /feedback` — опциональный вердикт специалиста; HTTP 201 означает, что
вердикт **сохранён**. В Docker данные записываются в PostgreSQL
`classification_feedback`, при локальном запуске с SQLite — в
`classification_feedback.sqlite` рядом с файлом заданий.

```json
{
  "query": "Электрочайники",
  "predicted_code": "27.51.24.110",
  "verdict": "incorrect",
  "corrected_code": "27.51.24"
}
```

`verdict`: `correct` | `incorrect` | `absent`. Для `incorrect` нужен
`corrected_code`; сервер отклоняет его отсутствие кодом 422. Ответ:
`{"id":"uuid", "saved":true}`. Публичного эндпоинта просмотра всех отзывов
**нет**: названия товаров могут содержать внутренние данные закупок.
Локальная история самого пользователя включается добровольно на клиенте.
