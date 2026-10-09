# Dora B2BCase — Backend API

Бэкенд на **FastAPI** для классификации товаров по **Постановлению
Правительства РФ № 1875** (национальный режим в закупках): по названию товара
сервис находит вероятную позицию перечня и код ОКПД2, показывает уверенность и
альтернативы. Результат — **подсказка для специалиста**, не юридическая проверка.

## Быстрый старт (локально)

```bash
python -m pip install -r api/requirements-dev.txt
python scripts/build_model.py          # лёгкие ML-артефакты в models/ (секунды)
PYTHONPATH=api uvicorn app.main:app --reload --port 8000
```

- Демо-UI:  http://localhost:8000/ui/   (корень `/` ведёт туда же)
  — экспорт батча в CSV (кнопка «Скачать CSV», те же колонки, что у
  `/classify/batch/{id}.csv`), прогресс через SSE с автоматическим фолбэком на
  опрос, ссылки «источник» на `GET /source/{record_id}`
- API:      http://localhost:8000
- Swagger:  http://localhost:8000/docs
- Health:   http://localhost:8000/health

## Запуск через Docker (рекомендуемый)

```bash
docker compose up --build
```

- UI/API:  http://localhost:8000/ui/
- Qdrant:  http://localhost:6333 (данные в volume `qdrant_data`, каталог — в `catalog_data`)

Проверка стека на целевой машине: `bash scripts/docker_smoke.sh --build`
(чек-лист: `eval/docker-target-checklist.md` — в песочнице разработки Docker
отсутствовал, образы проверяются этим скриптом на месте).

## Режимы классификации (`CLASSIFIER_BACKEND`)

| Значение | Что делает | Когда использовать |
| --- | --- | --- |
| `catalog` | реальный каталог `data/data.csv`: лексическая выдача + обученный реранкер + откалиброванные пороги (**по умолчанию**) | демо, работа без весов ML |
| `qdrant` | семантический поиск в Qdrant (`ml/`), тот же реранкер и пороги | когда в `models/` лежит модель (например `bge-m3`) |
| `mock` | детерминированная заглушка `MOCK-001` | разработка фронтенда |

**Выбор режима для демо:** каталоговый гибрид работает офлайн и даёт
top-1 83.7% (top-3 89.2%) на размеченном наборе из 214 строк (`eval/eval_report.md`); семантический режим
закрывает абстрактные категории («костюм» → «Одежда, кроме одежды из меха») и
включается без изменений API. Реранкер, пороги `ok`/`review` и формат ответа
общие для обоих режимов — контракт v1.0 (`docs/api-contract.md`).

## Ограничение частоты запросов

Ответы 429 отдают `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`;
клиент определяется по первому адресу `X-Forwarded-For` (за прокси) или по
сокету. `/health*`, `/metrics` и статика `/ui` не лимитируются, у `/admin/*`
свой строгий лимит. Счётчик отказов — в `GET /metrics`
(`rate_limited_total`, `rate_limited_by_path`).

```bash
# 21-й запрос подряд с одного адреса в ту же секунду (burst 20) → 429
for i in $(seq 1 25); do
  curl -s -o /dev/null -w "%{http_code} " -X POST localhost:8000/classify \
    -H 'Content-Type: application/json' -H "X-Forwarded-For: 203.0.113.7" \
    -d '{"name": "Ноутбук Lenovo"}'
done; echo
```

Для нагрузочных прогонов (`scripts/load_test.py`, `eval/api-load-test.md`)
лимит отключают: `RATE_LIMIT_ENABLED=false`. Скрипт сам подсказывает это,
если получил 429.

## ML-модель и переиндексация

```bash
# лёгкие артефакты (tfidf-эмбеддер + калиброванный реранкер) — без интернета
python scripts/build_model.py

# полный семантический режим: положите SentenceTransformers-модель в models/
INSTALL_ML=true CLASSIFIER_BACKEND=qdrant MODEL_NAME=./models/bge-m3 \
  ADMIN_API_KEY=<секрет> docker compose up --build
curl -X POST localhost:8000/admin/reindex -H "X-Admin-API-Key: <секрет>"
```

`/admin/reindex` переиндексирует каталог в Qdrant фоном и атомарно переключает
алиас (поиск не прерывается, повторная переиндексация не оставляет мусор).
Пороговое ПО ML устанавливается только при `INSTALL_ML=true`; без модели
работает `EMBEDDER_BACKEND=tfidf` (офлайн-фолбэк, проверен сквозными тестами
`api/tests/test_ml_e2e.py`).

## Переменные окружения (основные)

| Переменная | По умолчанию | Что делает |
| --- | --- | --- |
| `CLASSIFIER_BACKEND` | `catalog` | `catalog` \| `qdrant` \| `mock` (см. таблицу выше) |
| `ML_ENABLED` | `false` | включает тяжёлый ML-стек (Qdrant-режим) |
| `DATA_DIR` | `data` | каталог `data.csv`/`data.json` и `source_text.txt` |
| `CATALOG_PATH` | — | явный путь к `data.csv` (перебивает `DATA_DIR`) |
| `MODELS_DIR` | `models` | артефакты модели (`reranker.json`, `tfidf_embedder.json`) |
| `JOB_STORE` | `sqlite` | хранилище фоновых задач: `sqlite` (переживает рестарт) или `memory` |
| `JOB_STORE_PATH` | `data/jobs.sqlite` | файл sqlite-хранилища (в Docker — volume `jobs_data`) |
| `JOB_TTL_HOURS` | `24` | сколько хранить завершённые задачи до очистки |
| `ADMIN_API_KEY` | — | ключ `X-Admin-API-Key`; без него админ-эндпоинты отключены |
| `MAX_REQUEST_BODY_BYTES` | `1000000` | ограничение размера тела запроса (413) |
| `RATE_LIMIT_ENABLED` | `true` | включает per-client rate limit (token bucket) |
| `RATE_LIMIT_RPS` / `RATE_LIMIT_BURST` | `10` / `20` | лимит публичных эндпоинтов (на клиента) |
| `RATE_LIMIT_ADMIN_RPS` / `..._BURST` | `1` / `2` | отдельный строгий лимит на `/admin/*` |

## Обновление каталога перечня

Регламент — `docs/catalog-update.md`. Коротко: `POST /admin/catalog`
(`dry_run=true` — проверка без установки) принимает CSV/JSON, валидирует
строки и атомарно заменяет `data.csv` + `data.json`; затем
`POST /admin/reindex` (или `auto_reindex=true`). Для проверки файла без API:
`python scripts/validate_catalog.py new_catalog.csv`.

## Качество и оценка

- Размеченный набор: `eval/labeled_examples.csv` (214 запросов: 166 с кодом, 48
  «вне перечня» — товар проверен по `data/data.csv`, 17 спорных помечены
  `disputed=1`).
- Прогон: `PYTHONPATH=api python eval/run_eval.py [--sweep]` → `eval/eval_report.md`
  (top-1/top-3, доля верного `not_found`, регрессия на точных названиях
  каталога, обе шкалы уверенности `confidence_raw`/`confidence` и ECE, свип
  порога `review`, разбор ошибок).
- Синтетический бенчмарк без разметки: `PYTHONPATH=api python eval/synthetic_eval.py`
  → `eval/synthetic_report.md` (информационный, CI-гейт его не проверяет).
- CI-гейт качества: job `quality` в `.github/workflows/lint.yml` пересобирает
  модель и падает, если метрики ниже порогов `--min-top1/--min-top3/--min-silence/--min-exact`.
- Эксперименты и выводы: `eval/experiments.md`.
- Нагрузочный замер: `eval/api-load-test.md`, `scripts/load_test.py`.
- План следующих работ (мерж, часть D на целевой машине, известное
  train/serve-расхождение): `docs/next-steps.md`.

## Эндпоинты (кратко)

| Метод | Путь | Назначение |
| --- | --- | --- |
| POST | `/classify` | один товар → `ClassifyResponse` (контракт v1.0) |
| POST | `/classify/batch` | до 100 товаров → фоновая задача с прогрессом |
| GET | `/classify/batch/{job_id}` | статус/результат задачи |
| GET | `/classify/batch/{job_id}/events` | SSE-прогресс (`progress`, `done`) |
| GET | `/classify/batch/{job_id}.csv` | результат батча в CSV (`;`, UTF-8 BOM) |
| GET | `/source/{record_id}` | фрагмент текста ПП № 1875 — источник для `clause_url` |
| GET | `/health`, `/health/ready`, `/metrics` | liveness/readiness/метрики |
| GET/POST | `/admin/catalog` | сведения о каталоге / загрузка нового (ключ `X-Admin-API-Key`) |
| POST | `/admin/reindex` | переиндексация Qdrant (ключ `X-Admin-API-Key`) |

Полный контракт результата — `docs/api-contract.md`; он закреплён
`api/tests/test_contract.py` и используется фронтендом `frontend/`.

### Пример ответа и ссылка на источник

```bash
curl -s localhost:8000/classify -H 'Content-Type: application/json' \
  -d '{"name": "Электрочайники"}'
```

```json
{
  "query": "Электрочайники",
  "code": "27.51.24.110",
  "title": "Электрочайники",
  "confidence": 0.9976,
  "confidence_raw": 0.8799,
  "status": "ok",
  "clause_ref": "Приложение 2, позиция 339",
  "clause_url": "/source/2-339-01",
  "alternatives": [ { "code": "27.32", "title": "…", "confidence": 0.0165, "clause_ref": "…", "clause_url": "/source/2-113-01" } ],
  "latency_ms": 3.1,
  "contract_version": "1.0"
}
```

`pp_link` в каталоге — это цитата («ПП № 1875, приложение 2, позиция 339, стр. 64»), а
не URL, поэтому источник отдаёт сам сервис: `clause_url` ведёт на `GET /source/{record_id}`,
который возвращает страницы `pdf_page_start..pdf_page_end` из `data/source_text.txt`:

```bash
curl -s localhost:8000/source/2-339-01 | python -m json.tool | head -20
```

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

## Тесты

```bash
python -m pip install -r api/requirements-dev.txt
PYTHONPATH=api python -m pytest api/tests -q
```

## Структура

```
api/
  app/
    main.py            # FastAPI: роуты, admin-каталог, lifespan, статика /ui
    schemas.py         # контракт v1.0 (Pydantic)
    job_store.py       # in-memory фоновые задачи
    core/              # config, logging, correlation id, metrics
    middleware/        # observe, ограничение размера тела, rate limit
    services/          # classifier (catalog/mock/qdrant), catalog_io, source_lookup
  tests/               # API, классификатор, контракт, ML E2E, compose
frontend/              # демо-UI (статика, без сборки)
ml/                    # эмбеддеры (tfidf/sentence-transformers), индексация,
                       # поиск, реранкер, лексические мосты, обучение
models/                # артефакты (генерируются: scripts/build_model.py)
data/                  # каталог ПП № 1875 (data.csv + data.json)
eval/                  # размеченный набор, метрики, эксперименты, чек-листы
docs/                  # api-contract.md, catalog-update.md
scripts/               # build_model, validate_catalog, load_test, docker_smoke
```

> Фоновые задачи хранятся в store, выбранном `JOB_STORE` (`sqlite` по умолчанию,
> файл `JOB_STORE_PATH`): после перезапуска API незавершённые задачи помечаются
> `failed` с `error="interrupted by restart"`, а завершённые переживают рестарт.
> Для многопроцессного деплоя всё ещё нужен общий store (Redis/DB).
