# Dora B2BCase — Backend API

Бэкенд на **FastAPI** для классификации товаров по **Постановлению Правительства РФ № 1875**
(национальный режим в закупках). Задание первого спринта: один эндпоинт для одного товара,
пакетный — через фоновые задания с прогрессом, docker-compose (API + Qdrant), логи и тайминг.

## Быстрый старт (локально)

```bash
python -m pip install -r api/requirements-dev.txt
PYTHONPATH=api uvicorn app.main:app --reload --port 8000
```

- API:        http://localhost:8000
- Swagger:    http://localhost:8000/docs
- Health:     http://localhost:8000/health
- Метрики:    http://localhost:8000/metrics

## Запуск через Docker (рекомендуемый)

```bash
docker compose up --build
```

- API:     http://localhost:8000
- Qdrant:  http://localhost:6333  (данные сохраняются в volume `qdrant_data`)
- Swagger: http://localhost:8000/docs

## Эндпоинты

### `POST /classify` — один товар

```bash
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"name":"Ноутбук Lenovo IdeaPad"}'
```

Ответ:

```json
{
  "query": "Ноутбук Lenovo IdeaPad",
  "code": "26.20.11",
  "title": "Компьютеры портативные массой не более 10 кг, такие как ноутбуки, ...",
  "confidence": 0.5,
  "status": "review",
  "clause_ref": "Приложение 2",
  "clause_url": null,
  "alternatives": [],
  "latency_ms": 1.2
}
```

### `POST /classify/batch` — до 100 товаров (фоновое задание)

Возвращает `job_id` и `status_url`. Фронтенд опрашивает `GET` по `status_url`,
получая статус, прогресс (в %), число обработанных и готовые результаты.

```bash
curl -X POST http://localhost:8000/classify/batch \
  -H 'Content-Type: application/json' \
  -d '{"items":["Ноутбук Lenovo","Мышь компьютерная"]}'
```

```json
{ "job_id": "3f1...", "status": "queued", "total": 2,
  "status_url": "/classify/batch/3f1..." }
```

### `GET /classify/batch/{job_id}` — статус, прогресс, результат

```json
{
  "job_id": "3f1...", "status": "completed", "total": 2, "completed": 2,
  "progress_percent": 100.0, "results": [ ... ], "error": null
}
```

## Реальный ответ (`/classify`)

По умолчанию (`CLASSIFIER_BACKEND=catalog`) товар классифицируется по **реальному**
каталогу `data/data.csv` (позиции ПП № 1875): прямое совпадение ОКПД2-кода или
эвристика по пересечению токенов названия с наименованиями позиций. Это рабочий
ответ на демонстрационных данных — общий критерий готовности спринта выполнен.

Для перехода на ML-семантический поиск ML-разработчик поднимает индекс в Qdrant и
переключает бэкенд: `CLASSIFIER_BACKEND=qdrant` (адаптер в
`api/app/services/classifier.py::QdrantClassifier` вызывает `ml.search`).

Режимы (`CLASSIFIER_BACKEND`):

| Значение  | Что делает                                              |
|-----------|---------------------------------------------------------|
| `catalog` | реальная классификация по `data/data.csv` (по умолчанию)|
| `mock`    | детерминированная заглушка `MOCK-001` (для фронтенда)   |
| `qdrant` | семантический поиск через `ml.search` (нужен индекс)    |

## Логи и метрики

- Каждый запрос логируется с корреляционным `rid` (`X-Request-ID`), методом,
  путём, статусом и временем обработки. Логи настраиваются через `LOG_LEVEL`.
- Каждый ответ содержит заголовки `X-Request-ID` и `X-Process-Time-Ms`.
- `GET /metrics` отдаёт агрегированную статистику (число запросов, распределение
  по статусам, среднее время, запросы по путям).

## Переменные окружения

См. `.env.example`. Ключевые: `LOG_LEVEL`, `CLASSIFIER_BACKEND`, `DATA_DIR`,
`CATALOG_PATH`, `QDRANT_URL`, `API_PORT`.

## Тесты

```bash
python -m pip install -r api/requirements-dev.txt
PYTHONPATH=api python -m pytest api/tests -q
```

## Структура

```
api/
  app/
    main.py            # FastAPI-приложение, роуты, lifespan
    schemas.py         # Pydantic-модели запросов/ответов
    job_store.py       # in-memory хранилище фоновых заданий
    core/              # config, logging, context (correlation id), metrics
    middleware/        # observe: correlation id + тайминг + логи
    services/          # classifier: catalog / mock / qdrant
  tests/               # тесты API и классификатора
  Dockerfile
docker-compose.yml     # API + Qdrant + healthcheck + volume
data/                  # каталог ПП № 1875 (data.csv используется классификатором)
ml/                    # ML-пакет (эмбеддинги, индексация, поиск в Qdrant)
```

> Задания хранятся в памяти API. При перезапуске незавершённые задачи пропадают —
> для текущего прототипа это допустимо (для прод-режима потребуется общий стор,
> например Redis).
