# Контракт API v1.0

Формат результата классификации, согласованный с фронтендом (`frontend/src/api.ts`)
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
| `query` | string | исходное название после удаления внешних пробелов; смысловая очистка его в ответе не подменяет |
| `code` | string\|null | вероятный код ОКПД2; `null` при `not_found` |
| `title` | string\|null | наименование позиции перечня; `null` при `not_found` |
| `confidence` | float 0..1 | отображаемая оценка; её смысл задаётся `confidence_kind` |
| `confidence_kind` | enum | `calibrated` — калиброванная модель для этого каталога; `exact` — точное совпадение кода; `uncalibrated` — скоринг, который нельзя выдавать за вероятность |
| `confidence_raw` | float 0..1 | сырая уверенность до калибровки — **по ней принимается `status`**; при `not_found` — `0.0` |
| `status` | enum | `ok` — уверенная подсказка; `review` — проверить специалисту; `not_found` — надёжного совпадения нет |
| `clause_ref` | string\|null | где в перечне: «Приложение N, позиция M» (для дублей — перечисление) |
| `clause_url` | string\|null | ссылка на источник: `GET /source/{record_id}` — фрагмент `data/source_text.txt` со страницами позиции; `null`, если позиция неизвестна (mock-режим) |
| `alternatives` | array ≤3 | альтернативные позиции; `confidence` — оценка соответствия пары запрос/позиция; каждая несёт свои `clause_ref`/`clause_url` |
| `latency_ms` | float ≥0 | время обработки на сервере |
| `contract_version` | string | версия контракта (`"1.0"`) |

**Инварианты** (проверяются тестами):

1. `status = "not_found"` ⇒ `code = null`, `title = null`, `confidence = confidence_raw = 0.0`.
2. `0.0 ≤ confidence ≤ 1.0`; уверенность — подсказка, не юридический вердикт.
3. Точное совпадение переданного кода с включённой строкой каталога ⇒ `confidence = confidence_raw = 1.0`, `confidence_kind = "exact"`, `status = "ok"`. Отсутствующий или исключённый код не ищется по нечёткому сходству.
4. В актуальной модели `status` определяется сырой оценкой до округления; `confidence` — её монотонное преобразование. Если модель не соответствует каталогу, результат получает `confidence_kind = "uncalibrated"`, а `ok` по текстовому запросу заменяется на `review`.
5. Калибровка обучена на отдельной синтетической площадке с подготовленной лексикой, а не на собранном промышленном трафике. Применимость к новым данным проверяется отдельно.

### `POST /classify/batch` → `GET /classify/batch/{job_id}`

```json
{ "items": ["до 100 названий, ≤500 символов каждое"] }
```

`202 Accepted` → `{job_id, status:"queued", total, status_url}`. Опрос
`status_url` каждые ~300–500 мс:

```json
{ "job_id": "…", "status": "queued|running|completed|failed",
  "total": 2, "completed": 1, "progress_percent": 50.0,
  "results": [ ClassifyResponse, … ], "error": null }
```

`results` заполняются по мере обработки (порядок соответствует `items`).

### `GET /classify/batch/{job_id}/events` — прогресс по SSE

`text/event-stream` с событиями:

```text
event: progress
data: {"job_id": "...", "status": "running", "completed": 2, "total": 5, "progress_percent": 40.0}

event: done
data: {"job_id": "...", "status": "completed", "total": 5, "completed": 5,
       "progress_percent": 100.0, "results": [ ClassifyResponse, ... ], "error": null}
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
query;code;title;status;confidence;clause_ref;clause_url;
alt1_code;alt1_title;alt2_code;alt2_title;alt3_code;alt3_title
```

Кавычки внутри значений удваиваются, поля с `;`/кавычками/переводами строк
экранируются. Для задачи в статусе `queued`/`running` — 409, для неизвестного
`job_id` — 404; у `failed` отдаются частичные результаты.

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

`text` обрезается до 20 000 символов (`truncated: true`). Ответ не кэшируется
(`Cache-Control: no-store`), чтобы обновление каталога не оставляло старый фрагмент в браузере.
Дополнительные поля `edition` и `source_url` передают сведения оператора. Параметр
`?format=html` возвращает читаемую HTML-страницу с экранированным текстом.
Если страницы отсутствуют или каталог и источник не соответствуют манифесту,
возвращается 503; начало чужого документа в качестве замены больше не используется.

## Административные (заголовок `X-Admin-API-Key`)

| Метод | Путь | Назначение |
| --- | --- | --- |
| POST | `/admin/catalog` | загрузка/проверка каталога (`file`, `dry_run`, `auto_reindex`) |
| GET | `/admin/catalog` | факты об установленном каталоге (строки, sha256, время) |
| POST | `/admin/reindex` | переиндексация Qdrant (фоновая задача, 202 + `status_url`) |
| POST | `/admin/models/reload` | перечитать локальные веса после обучения, вернуть readiness |
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

Публичные эндпоинты ограничены token bucket'ом на клиента (первый адрес
`X-Forwarded-For`, иначе адрес сокета): `RATE_LIMIT_RPS`/`RATE_LIMIT_BURST`
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
считаются в `GET /metrics` (`rate_limited_total`, `rate_limited_by_path`).
