# Контракт API v1.0

Формат результата классификации, согласованный с фронтендом (`frontend/app.js`)
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
  "confidence": 0.38,
  "status": "review",
  "clause_ref": "Приложение 2, позиция 195; Приложение 3, позиция 37",
  "clause_url": null,
  "alternatives": [
    { "code": "26.20.17", "title": "Мониторы и проекторы ...", "confidence": 0.07, "clause_ref": "Приложение 2, позиция 174" }
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
| `confidence` | float 0..1 | откалиброванная уверенность модели в **top-1** для этого запроса |
| `status` | enum | `ok` — уверенная подсказка; `review` — проверить специалисту; `not_found` — надёжного совпадения нет |
| `clause_ref` | string\|null | где в перечне: «Приложение N, позиция M» (для дублей — перечисление) |
| `clause_url` | string\|null | зарезервировано для ссылки на источник (сейчас `null`) |
| `alternatives` | array ≤3 | альтернативные позиции; `confidence` — оценка соответствия пары запрос/позиция |
| `latency_ms` | float ≥0 | время обработки на сервере |
| `contract_version` | string | версия контракта (`"1.0"`) |

**Инварианты** (проверяются тестами):

1. `status = "not_found"` ⇒ `code = null`, `title = null`.
2. `0.0 ≤ confidence ≤ 1.0`; уверенность — подсказка, не юридический вердикт.
3. Прямая передача кода ОКПД2 в `name` ⇒ `confidence = 1.0`, `status = "ok"`.

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
| 429 | переполнение очереди фоновых задач (`Retry-After`) | строка |
| 503 | ML-режим выключен/не собран, классификатор не инициализирован | строка |

Все ответы несут заголовки `X-Request-ID` (прокидывается из запроса) и
`X-Process-Time-Ms`. CORS настраивается `CORS_ALLOW_ORIGINS`.
