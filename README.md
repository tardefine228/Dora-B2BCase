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
top-1 ≈ 73% на размеченном наборе (`eval/eval_report.md`); семантический режим
закрывает абстрактные категории («костюм» → «Одежда, кроме одежды из меха») и
включается без изменений API. Реранкер, пороги `ok`/`review` и формат ответа
общие для обоих режимов — контракт v1.0 (`docs/api-contract.md`).

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

## Обновление каталога перечня

Регламент — `docs/catalog-update.md`. Коротко: `POST /admin/catalog`
(`dry_run=true` — проверка без установки) принимает CSV/JSON, валидирует
строки и атомарно заменяет `data.csv` + `data.json`; затем
`POST /admin/reindex` (или `auto_reindex=true`). Для проверки файла без API:
`python scripts/validate_catalog.py new_catalog.csv`.

## Качество и оценка

- Размеченный набор: `eval/labeled_examples.csv` (89 запросов, включая
  «нет в перечне» и спорные позиции).
- Прогон: `PYTHONPATH=api python eval/run_eval.py [--sweep]` → `eval/eval_report.md`
  (top-1/top-3, доля верного `not_found`, регрессия на точных названиях
  каталога, калибровка уверенности ECE, свип порога `review`, разбор ошибок).
- CI-гейт качества: job `quality` в `.github/workflows/lint.yml` пересобирает
  модель и падает, если метрики ниже порогов `--min-top1/--min-top3/--min-silence/--min-exact`.
- Эксперименты и выводы: `eval/experiments.md`.
- Нагрузочный замер: `eval/api-load-test.md`, `scripts/load_test.py`.

## Эндпоинты (кратко)

| Метод | Путь | Назначение |
| --- | --- | --- |
| POST | `/classify` | один товар → `ClassifyResponse` (контракт v1.0) |
| POST | `/classify/batch` | до 100 товаров → фоновая задача с прогрессом |
| GET | `/classify/batch/{job_id}` | статус/результат задачи |
| GET | `/health`, `/health/ready`, `/metrics` | liveness/readiness/метрики |
| GET/POST | `/admin/catalog` | сведения о каталоге / загрузка нового (ключ `X-Admin-API-Key`) |
| POST | `/admin/reindex` | переиндексация Qdrant (ключ `X-Admin-API-Key`) |

Полный контракт результата — `docs/api-contract.md`; он закреплён
`api/tests/test_contract.py` и используется фронтендом `frontend/`.

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
    middleware/        # observe + ограничение размера тела
    services/          # classifier (catalog/mock/qdrant), catalog_io
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

> Фоновые задачи хранятся в памяти API — при перезапуске незавершённые
> пропадают. Для прод-режима нужен общий стор (Redis/DB).
