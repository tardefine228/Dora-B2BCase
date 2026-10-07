# Dora B2BCase — Backend API

Бэкенд на **FastAPI** для классификации товаров по Постановлению Правительства РФ № 1875.
API описывает запросы и ответы через Pydantic; интерактивная OpenAPI-документация доступна
в `/docs` и `/redoc`.

> Результат классификации — подсказка для проверки специалистом, а не юридическое заключение.

## Что работает сейчас

- `POST /classify` — классификация одного товара.
- `POST /classify/batch` — фоновая обработка до 100 товаров; статус и результаты доступны
  через `GET /classify/batch/{job_id}`.
- `/health` (liveness), `/health/ready` (готовность классификатора) и `/metrics`.
- Docker Compose поднимает API и Qdrant с постоянным именованным volume.
- По умолчанию используется автономный облегчённый классификатор по `data/data.csv`.
- ML-классификатор, кэш ответов, ограничение параллельного инференса и защищённая задача
  переиндексации в Qdrant подключаются отдельно.

## Быстрый запуск без Docker

Требуется Python 3.10+ (в CI используется Python 3.12).

```bash
python -m venv .venv
# Linux/macOS
. .venv/bin/activate
# Windows PowerShell: .venv\\Scripts\\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r api/requirements-dev.txt
PYTHONPATH=api python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Каталоговый режим не требует Qdrant, скачивания модели или доступа к интернету после установки
зависимостей. Проверить сервис:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/health/ready
curl -X POST http://localhost:8000/classify \
  -H 'Content-Type: application/json' \
  -d '{"name":"Ноутбук Lenovo IdeaPad"}'
```

При запуске из Windows вместо `PYTHONPATH=api` используйте PowerShell:

```powershell
$env:PYTHONPATH = "api"
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Запуск через Docker Compose

Docker Engine и Compose v2 должны быть установлены. Compose читает переменные из `.env`:

```bash
cp .env.example .env
# При необходимости измените порты, лимиты памяти и другие настройки в .env
docker compose up --build -d
docker compose ps
curl http://localhost:8000/health/ready
```

Полезные адреса:

- API: `http://localhost:8000`
- Swagger: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
- Qdrant HTTP: `http://localhost:6333` (порт привязан к `127.0.0.1` по умолчанию)

Данные Qdrant сохраняются в named volume `qdrant_data` между пересозданиями контейнеров.
Обычный `docker compose down` volume не удаляет; `docker compose down -v` удалит его и все
данные индекса.

Compose по умолчанию ограничивает API до 1 GiB RAM / 2 CPU, Qdrant — до 768 MiB / 1 CPU.
Значения можно изменить через `API_MEMORY_LIMIT`, `API_CPUS`, `QDRANT_MEMORY_LIMIT` и
`QDRANT_CPUS`. Для ML-модели может потребоваться больше памяти.

Логи контейнера:

```bash
docker compose logs -f api
```

## API-контракт

### `POST /classify`

Запрос:

```json
{"name":"Ноутбук Lenovo"}
```

Ответ:

```json
{
  "query": "Ноутбук Lenovo IdeaPad",
  "code": "26.20.11",
  "title": "Компьютеры портативные массой не более 10 кг, такие как ноутбуки, ...",
  "confidence": 0.5,
  "status": "ok",
  "clause_ref": "Приложение 2",
  "clause_url": null,
  "alternatives": [],
  "latency_ms": 1.2
}
```

`status` имеет значения `ok`, `review` или `not_found`; код, заголовок и ссылка могут быть
`null`, если совпадение не найдено. Эвристическая уверенность не является вероятностью
юридической корректности.

### `POST /classify/batch`

В теле передаётся `{"items":["Ноутбук Lenovo","Мышь компьютерная"]}`. Допускается от 1
до 100 элементов, длина каждого названия — от 1 до 500 символов. Пустые строки обрезаются и
отклоняются. Тело запроса ограничено 1 MiB (`MAX_REQUEST_BODY_BYTES`). Запрос возвращает HTTP
`202` и `job_id`; клиент опрашивает `status_url`:

```json
{
  "job_id": "<uuid>",
  "status": "queued",
  "total": 2,
  "status_url": "/classify/batch/<uuid>"
}
```

Статусы задачи: `queued`, `running`, `completed`, `failed`. Пока задача выполняется, ответ
содержит число обработанных элементов, процент и готовые результаты.

### Ошибки и ограничения

- `422 Unprocessable Entity` — некорректный JSON, неизвестные поля или нарушение правил
  Pydantic-валидации (в том числе пустое название, более 500 символов, больше 100 товаров).
  Это стандартный FastAPI-код для синтаксически корректного, но невалидного запроса.
- `404` — неизвестный или истёкший `job_id`.
- `429` — достигнут лимит параллельных фоновых задач; повторите позже (`Retry-After`).
- `503` — классификатор или административная возможность не готова.
- `500` — неожиданная ошибка; детали исключения не выдаются клиенту, но сохраняются в логах.

Фоновые задачи и кэш хранятся в памяти процесса: перезапуск API удаляет незавершённые
задачи и результаты кэша. Для этого прототипа включён один worker. Для нескольких реплик,
гарантированной доставки и переживания рестартов подключите Redis/очередь задач и общее
хранилище статусов.

## Классификаторы

Значение задаётся переменной `CLASSIFIER_BACKEND`:

| Значение | Поведение |
| --- | --- |
| `catalog` | По умолчанию. Быстрое автономное сопоставление с `data/data.csv`; нет семантического поиска. |
| `mock` | Детерминированный ответ `MOCK-001` для фронтенда и интеграционных тестов. |
| `qdrant` | Семантический поиск с SentenceTransformers и Qdrant; требует дополнительных зависимостей, локальной модели и индекса. |

Каталоговый режим не выдаёт фиктивный ответ при отсутствии файла: готовность станет `503`,
а причина попадёт в лог. Исходные строки каталога хранятся в Git; пользовательские модели и
векторные данные в Git не добавляются.

### Необязательный ML/Qdrant-режим и работа офлайн

В репозитории **нет весов модели**: их размер и лицензия зависят от выбранной модели. Поэтому
образ по умолчанию не скачивает модель и остаётся сравнительно небольшим; runtime не должен
загружать веса автоматически, если включён `MODEL_LOCAL_ONLY=true`.

1. Поместите полностью подготовленную модель SentenceTransformers в `./models/bge-m3`.
   Каталог `models/` смонтирован в API-контейнер как `/app/models` и исключён из Git.
2. В `.env` укажите:

   ```dotenv
   INSTALL_ML=true
   CLASSIFIER_BACKEND=qdrant
   MODEL_NAME=./models/bge-m3
   MODEL_LOCAL_ONLY=true
   MODEL_DEVICE=cpu
   ADMIN_API_KEY=<длинный-случайный-секрет>
   ```

3. Пересоберите API и запустите сервисы: `docker compose up --build -d`.
4. Сначала проверьте `docker compose logs -f api`. Для первичной индексации используйте
   защищённый endpoint ниже; пока коллекция не создана, `/health/ready` вернёт `503`, но
   liveness `/health` и административный endpoint останутся доступны.

`INSTALL_ML=true` устанавливает тяжёлые ML-зависимости во время сборки, включая CPU-only
PyTorch, чтобы не добавлять CUDA-библиотеки в образ. Это **не** копирует веса модели в
Docker-образ: их нужно подготовить локально и сохранить в `models/`. Первый
`docker compose up --build` всё ещё требует доступа к реестрам для скачивания базовых образов
и Python-пакетов. После успешной сборки, при локальной модели и сохранённом Qdrant volume,
работа классификатора не требует интернета. Для абсолютного офлайн-развёртывания заранее
сохраните Docker-образы и пакетные артефакты на машине жюри.

### `POST /admin/reindex`

Включён только когда заданы `ADMIN_API_KEY` и `INSTALL_ML=true` (последний параметр также
включает runtime-флаг `ML_ENABLED`). Endpoint требует заголовок
`X-Admin-API-Key: <секрет>` и запускает фоновую задачу:

```bash
curl -X POST http://localhost:8000/admin/reindex \
  -H "X-Admin-API-Key: $ADMIN_API_KEY"
```

В ответе возвращаются `job_id` и `status_url`; статус можно получить через
`GET /admin/reindex/{job_id}` с тем же заголовком. Переиндексация строит отдельную staging-
коллекцию и переключает alias после успешной загрузки, чтобы не удалять текущий индекс до
готовности нового. Путь JSON-каталога по умолчанию — `/app/data/data.json` в контейнере;
для локального запуска — `data/data.json`.

## Логи, метрики, кэш и производительность

- Логи идут в stdout/stderr (так их собирает Docker); запись в контейнерный файл не требуется.
  В каждой записи есть request ID, HTTP-статус, время и размер запроса, если он указан клиентом.
  Для batch отдельно логируется число товаров.
- Запросы возвращают `X-Request-ID` и `X-Process-Time-Ms`. Клиент может передать свой безопасный
  `X-Request-ID` длиной до 128 символов.
- `GET /metrics` показывает число запросов, статусы, среднюю задержку и счётчики по шаблону
  маршрута (без бесконечного роста от случайных `job_id`). Метрики локальны для процесса.
- LRU-кэш ответов ограничен `CLASSIFICATION_CACHE_SIZE` (по умолчанию 512 записей); число
  одновременных инференсов ограничено `MAX_CONCURRENT_INFERENCES` (по умолчанию 1).
- Batch классифицируется блоками; ML-адаптер кодирует запросы пачкой, а не загружает модель
  заново на каждый товар. Старт приложения загружает настроенную модель один раз.
- `/health` — liveness, `/health/ready` — готовность выбранного классификатора. Docker
  healthcheck использует readiness.

## Тестирование и нагрузочный прогон

```bash
python -m pip install -r api/requirements-dev.txt
PYTHONPATH=api python -m pytest api/tests -q
ruff check .
ruff format --check .
```

Для локальной оценки batch API (по умолчанию 10 одновременных запросов по 100 товаров):

```bash
python scripts/load_test.py --base-url http://127.0.0.1:8000
```

Скрипт измеряет задержку принятия запросов, полное время, число обработанных товаров и
метрики API. Чтобы дополнительно получить пиковый RSS API-процесса на Linux, передайте его
PID: `python scripts/load_test.py --pid <PID>`. Фактические RAM и latency нужно измерить на
целевой машине: числа зависят от CPU, модели, Docker и состава каталога. Один локальный
smoke-замер (без ML и Docker) записан в [`eval/api-load-test.md`](eval/api-load-test.md).

## Переменные окружения

Полный шаблон — `.env.example`. Важные настройки: `API_PORT`, `LOG_LEVEL`,
`CLASSIFIER_BACKEND`, `INSTALL_ML`, `DATA_DIR`, `CATALOG_PATH`, `QDRANT_URL`,
`QDRANT_COLLECTION`, `MODEL_NAME`, `MODEL_LOCAL_ONLY`, `ADMIN_API_KEY`,
`MAX_REQUEST_BODY_BYTES`, `MAX_CONCURRENT_INFERENCES`, `CLASSIFICATION_CACHE_SIZE`, лимиты памяти и CPU.

## Структура проекта

```text
api/app/
  main.py                 # FastAPI routes, lifespan, фоновые задачи
  schemas.py              # Pydantic / OpenAPI контракты
  job_store.py            # bounded in-memory job store
  core/                   # settings, logging, request metrics
  middleware/             # correlation ID, timing, safe error response
  services/classifier.py  # catalog / mock / Qdrant adapters, cache
api/tests/                # API and classifier tests
api/Dockerfile            # small default API image, optional ML build arg
ml/                       # embedding, catalog indexing and Qdrant search
scripts/load_test.py      # async batch load smoke test
data/                     # checked-in PP-1875 catalog
docker-compose.yml        # API + Qdrant + persistent qdrant_data volume
```
