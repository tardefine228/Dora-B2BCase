# Dora B2BCase — Backend API

Бэкенд на **FastAPI** для классификации товаров по **Постановлению
Правительства РФ № 1875** (национальный режим в закупках): по названию товара
сервис находит вероятную позицию перечня и код ОКПД2, показывает уверенность и
альтернативы. Результат — **подсказка для специалиста**, не юридическая проверка.

## Стек

| Компонент | Технология | Роль |
| --- | --- | --- |
| API | FastAPI | `/classify`, пакетная обработка, админ-эндпоинты, UI на `/ui/` |
| Интерфейс | React + TypeScript + Vite | одиночный товар, загрузка списка, прогресс, экспорт |
| БД | **PostgreSQL** | фоновые задачи и результаты батчей, журнал классификаций (`classification_log`) |
| Очередь | **Celery + Redis** | тяжёлые пакеты обрабатывает отдельный воркер, API не блокируется |
| Векторный поиск | **Qdrant** + эмбеддинги (multilingual-e5 → векторы spaCy → TF-IDF) | кандидаты «по смыслу» + признаки реранкера |
| Реранкер | логистическая модель с калибровкой | порядок кандидатов, уверенность, статусы `ok`/`review`/`not_found` |

## Запуск через Docker (рекомендуемый)

```bash
docker compose up --build
```

Поднимаются `api`, `worker` (Celery), `postgres`, `redis`, `qdrant`. Первая сборка долгая
(PyTorch + модель эмбеддингов + обучение реранкера под неё, ~10–15 минут). Повторный
`docker compose up --build` идёт из кэша: слой с загрузками стоит **до** копирования кода
и опирается только на файлы требований, поэтому правки кода/UI не вызывают повторных
загрузок, а pip- и Hugging Face-кэши лежат в BuildKit cache-mount'ах. Обычный
`docker compose up` (без `--build`) не скачивает ничего и стартует за секунды;
векторный кэш каталога и коллекции Qdrant переживают рестарт в volumes.

- UI/API:  http://localhost:8000/ui/  ·  Swagger: http://localhost:8000/docs
- Что реально работает: `curl localhost:8000/health/ready` → блок `vector_search`
  (`embedder`, `dim`, `store` = `qdrant`/`memory`, `reranker` — какой артефакт скорит,
  `scoring`), `job_store`, `queue_backend`.
- Без трансформера (быстрая сборка, ~2 мин): `INSTALL_TRANSFORMER=false docker compose up --build` —
  векторный слой работает на офлайн-векторах spaCy.
- Если скачивание модели при сборке не удалось, сборка **не падает**: сервис откатывается на
  spaCy, а `/health/ready` покажет `embedder: spacy-ru_core_news_md`.
- Чтобы трансформер запёкся, сборке нужны исходящие HTTPS на: `huggingface.co` (API и веса),
  `*.hf.co` / `cdn-lfs.hf.co` (CDN весов), `download.pytorch.org` (torch CPU), `pypi.org` и
  `files.pythonhosted.org` (пакеты). Антивирусный веб-фильтр или корпоративный прокси любят
  резать именно их: сборка теперь проверяет доступность Hugging Face **до** тяжёлых скачиваний
  и печатает в лог, какой именно шаг упал. Успех виден так: в логе сборки есть строка
  `Model intfloat/multilingual-e5-small saved to …`, без `WARNING: transformer step failed`,
  а `/health/ready` → `vector_search.embedder` = `st-embedder`, `dim` = 384 (у spaCy —
  `spacy-ru_core_news_md`, `dim` = 300).
- Сборка успешна, но `/health/ready` всё равно показывает `spacy-ru_core_news_md`? Почти
  всегда виноват устаревший `MODEL_NAME` в вашем `.env` (например, `./models/bge-m3` из
  старых ревизий `.env.example`): переменная из `.env` перекрывает путь `/opt/models/embedder`,
  запечённый в образе, каталог не находится — и рантайм молча откатывается на spaCy.
  Диагностика: `docker compose exec api ls /opt/models/embedder` (модель в образе есть?)
  и `docker compose exec api printenv MODEL_NAME` (что реально видит сервис?). Лечение —
  убрать строку `MODEL_NAME` из `.env` и `docker compose up -d`; пересборка не нужна.
  Первый старт после переключения займёт минуту-другую: каталог переэмбедится в новую
  коллекцию Qdrant (имя коллекции включает тег эмбеддера, старая удаляется автоматически).

Проверка стека на целевой машине: `bash scripts/docker_smoke.sh --build`.

## Быстрый старт (локально, без Docker)

```bash
python -m pip install -r api/requirements-dev.txt
PYTHONPATH=api uvicorn app.main:app --reload --port 8000 --no-proxy-headers
```

По умолчанию локально: задачи в SQLite (`JOB_STORE=sqlite`), пакеты в процессе API
(`QUEUE_BACKEND=inprocess`), векторы в памяти (`VECTOR_STORE=memory`). Полный стек —
переменными окружения (см. таблицу ниже) и сервисами из `docker-compose.yml`.

## Векторный поиск

Классификатор в режиме `catalog` — **гибридный**: шорт-лист = лексический матчер ∪ ближайшие
по смыслу позиции из векторного индекса (`ml/dense.py`). Векторы названий позиций строятся
выбранным эмбеддером (`EMBEDDER_BACKEND=auto`: sentence-transformers → spaCy → TF-IDF),
публикуются в Qdrant (коллекция адресуется содержимым каталога: новый каталог = новая
коллекция, старая удаляется) и кэшируются на диске. Ближайших соседей отдаёт Qdrant;
при его недоступности поиск автоматически идёт по матрице в памяти.

Косинус «запрос ↔ кандидат» — признак реранкера. Шкалы косинусов у разных эмбеддеров разные,
поэтому реранкер с векторными признаками обучается **под конкретный эмбеддер**
(`reranker.<тег-эмбеддера>.json`), а API берёт тот, что соответствует загруженному эмбеддеру —
в **обоих** режимах: `catalog` и `qdrant` (в семантическом косинус попадает в `dense_cosine`/
`dense_gap`, иначе признак был бы нулевым и артефакт не влиял бы на решение).
`SEMANTIC_RERANKER=lexical` возвращает семантический режим на общий `reranker.json`
(откат без пересборки образа). Какой артефакт реально скорит, видно в
`GET /health/ready` → `vector_search.reranker`; при векторных признаках пороги `QDRANT_*_FLOOR`
надо перемерить под эмбеддер.

**Предохранитель (эксп. 19).** Векторные признаки учатся на синтетических запросах и
могут ухудшить реальные ответы. Поэтому `scripts/build_model.py --dense --gate`
сравнивает такой реранкер с лексическим **на одинаковом уровне тишины** (свип порога
`review`, подгонка порога кандидата к тишине базы) и оставляет его только при выигрыше
≥ +1 п.п. top-1. Артефакт при отказе **не удаляется** — решение пишется в реестр
`dense_gate.json`, и загрузчик его игнорирует. Замер 2026-10-20 на доступном офлайн
эмбеддере (TF-IDF): dense проигрывает на равной тишине (85.5% против 86.7%) →
отклонён; e5-small замеряется той же командой на целевой машине.

Что измерено (размеченный набор, 166 товаров из перечня + 108 вне перечня):

| Вариант | top-1 с кодом | тишина (верный `not_found`) |
| --- | ---: | ---: |
| лексика + леммы + парные запреты + поля длинных имён (**текущий артефакт**) | 86.7% | 80.6% |
| + кандидаты из векторов spaCy, лексический реранкер | = лексике | = лексике |
| + векторы spaCy как признаки реранкера | 81.3% (до лемм.) | 88.9% |
| + векторы tfidf как признаки реранкера (гейт 2026-10-20) | 85.5% | 80.6% (равная) — **отклонён** |

Статические векторы spaCy слишком грубые (500 тыс. слов делят 20 тыс. векторов), поэтому
как признак они не помогают, и предохранитель их отклоняет. Выигрыша от **трансформера**
(multilingual-e5) в среде разработки измерить было нельзя (Hugging Face недоступен) — его
эффект покажет шаг `--gate` при сборке образа и `eval/run_eval.py` на вашей машине.

## Режимы классификации (`CLASSIFIER_BACKEND`)

| Значение | Что делает | Когда использовать |
| --- | --- | --- |
| `catalog` | реальный каталог `data/data.csv`: гибридная выдача (лексика + векторный поиск) + обученный реранкер + откалиброванные пороги (**по умолчанию**) | основной режим |
| `qdrant` | чисто семантический поиск в Qdrant (`ml/`), тот же реранкер, свои пороги | эксперименты со сравнением режимов |
| `mock` | детерминированная заглушка `MOCK-001` | разработка фронтенда |

**Выбор режима для демо:** каталоговый гибрид работает офлайн и даёт
top-1 86.7% (top-3 92.8%) по товарам из перечня на размеченном наборе из 274 строк (`eval/eval_report.md`).
Семантический режим измерен на том же наборе и настроен отдельно (эксп. 15–16,
`eval/backend_comparison.md`): на своей рабочей точке (`review = 0.25`) он почти не
выдумывает подсказок там, где товара нет в перечне — 97.2% против 88.0% у лексики, — но
отвечает реже (82.5% против 91.6% покрытия) и хуже по top-1, и стоит вчетверо дороже
(p50 ~171 мс против ~39 мс). Поэтому **по умолчанию остаётся `catalog`**, а `qdrant`
включается флагом как второй режим с другим компромиссом. Реранкер, формат ответа и
шкала уверенности общие; пороги решения — свои у каждого режима
(`QDRANT_OK_FLOOR`/`QDRANT_REVIEW_FLOOR`; для каталога — `CATALOG_*_FLOOR`), контракт v1.0 (`docs/api-contract.md`).

**Актуальные цифры каталога (2026-10-20, пересборка сессии ML):** top-1 86.7% /
top-3 92.8% / верный `not_found` 80.6% / top-1 всего 84.3% / точные названия 100% /
ECE 0.079 — см. `eval/experiments.md`, эксп. 17–22. Пять уверенных ложных друзей
(шредер, камера видеонаблюдения, конструктор, веб-камера, огнетушитель) замолчаны
парными запретами (`ml/lexicon.py`, `PAIR_VETO`).

## Ограничение частоты запросов

Ответы 429 отдают `Retry-After`, `X-RateLimit-Limit`, `X-RateLimit-Remaining`.
`/health*`, `/metrics` и статика `/ui` не лимитируются. Свои бакеты у
`/admin/*` (строгий лимит на изменяющие операции) и у `POST /classify/file`:
один файл — это до 100 позиций, поэтому загрузки считаются **по файлам**
(`RATE_LIMIT_FILE_RPS=1`, `RATE_LIMIT_FILE_BURST=3`) и не тратят общий лимит
запросов. Счётчик отказов — в `GET /metrics` (`rate_limited_total`,
`rate_limited_by_path`).

**Кто считается клиентом.** По умолчанию — адрес сокета, а `X-Forwarded-For`
**игнорируется**: иначе любой клиент подставлял бы случайный адрес в заголовок
и получал свежий бакет на каждый запрос. Заголовок читается только если запрос
пришёл от прокси из `TRUSTED_PROXIES` (IP или CIDR через запятую), и берётся
**правый непроверенный хоп** — тот, который дописал ваш прокси; всё, что клиент
дописал слева, не учитывается. Прокси за прокси обходятся справа налево, пока
не встретится адрес не из списка.

```bash
# Укажите фактический IP/узкую CIDR-сеть прокси (пример: 10.0.0.1).
# 21-й запрос с одного адреса → 429.
TRUSTED_PROXIES=10.0.0.1 docker compose up -d
for i in $(seq 1 25); do
  curl -s -o /dev/null -w "%{http_code} " -X POST localhost:8000/classify \
    -H 'Content-Type: application/json' -H "X-Forwarded-For: 203.0.113.7" \
    -d '{"name": "Ноутбук Lenovo"}'
done; echo

# Без прокси (сервис доступен напрямую) адрес берётся из сокета, а подделанный
# заголовок ничего не меняет — те же 429 на том же клиенте.
```

`TRUSTED_PROXIES` обязателен при работе за прокси: без него все клиенты
попадут в один бакет (лимит станет общим на весь сервис). У `uvicorn` в образе
и в примерах запуска стоит `--no-proxy-headers`, чтобы он не переписывал адрес
клиента сам: решение о доверии заголовку принимает только приложение.
Опечатка в `TRUSTED_PROXIES` роняет старт, а не игнорируется молча.

Для нагрузочных прогонов (`scripts/load_test.py`, `eval/api-load-test.md`)
лимит отключают: `RATE_LIMIT_ENABLED=false`. Скрипт сам подсказывает это,
если получил 429.

## ML-модель и переиндексация

```bash
# Готовые артефакты (лексический реранкер + tfidf-эмбеддер) лежат в ml/artifacts/ и входят в
# образ. Пересборка (без интернета, ~2–3 мин на 1 CPU), затем скопировать в ml/artifacts/:
python scripts/build_model.py
cp models/reranker.json models/tfidf_embedder.json ml/artifacts/

# реранкер с векторными признаками под активный эмбеддер (+ проверка предохранителем)
python scripts/build_model.py --dense --gate --models-dir models
```

Лемматизация (`pymorphy3`) встроена в `ml/features.stem` — основы в артефакте построены по
леммам, поэтому `pymorphy3` обязателен в рантайме (он в `api/requirements.txt`). После смены
каталога/стеммера артефакт нужно пересобрать.

Каталог обновляется **без передеплоя**: `POST /admin/catalog` кладёт новый `data.csv` на общий
том, а API и воркер сами пересобирают классификатор и векторный индекс при следующем запросе
(проверяется сигнатура файла). `POST /admin/reindex` нужен только режиму `qdrant`.

## Переменные окружения (основные)

| Переменная | По умолчанию | Что делает |
| --- | --- | --- |
| `CLASSIFIER_BACKEND` | `catalog` | `catalog` \| `qdrant` \| `mock` (см. таблицу выше) |
| `CATALOG_OK_FLOOR` / `CATALOG_REVIEW_FLOOR` | пороги из артефакта | decision floors для `catalog`; пусто — значения обученной модели, иначе `0..1` и `OK > REVIEW`; применяются при старте без переобучения |
| `VECTOR_SEARCH` | `true` | векторный слой в режиме `catalog` (при ошибке эмбеддера — лексика, видно в `/health/ready`) |
| `VECTOR_STORE` | `memory` (в Docker `qdrant`) | где хранятся и ищутся векторы; при недоступности Qdrant — память |
| `SEMANTIC_RERANKER` | `auto` | реранкер режима `qdrant`: `auto` — `reranker.<тег-эмбеддера>.json`, иначе `reranker.json`; `lexical` — всегда общий (откат) |
| `EMBEDDER_BACKEND` | `auto` | `sentence-transformers` \| `spacy` \| `tfidf`; явный выбор не откатывается молча |
| `MODEL_NAME` / `MODEL_LOCAL_ONLY` | `/opt/models/embedder` / `true` | путь к модели эмбеддингов (в образе скачана при сборке) |
| `ML_ENABLED` | `false` | включает чисто семантический режим `qdrant` |
| `DATA_DIR` | `data` | каталог `data.csv`/`data.json` и `source_text.txt` |
| `CATALOG_PATH` | — | явный путь к `data.csv` (перебивает `DATA_DIR`) |
| `MODELS_DIR` | `models` | артефакты модели (`reranker.json`, `tfidf_embedder.json`) |
| `QDRANT_OK_FLOOR` / `QDRANT_REVIEW_FLOOR` | `0.77` / `0.25` | пороги решения для семантического режима; пусто — взять из артефакта (эксп. 16) |
| `JOB_STORE` | `sqlite` (в Docker `postgres`) | хранилище задач: `postgres` \| `sqlite` \| `memory` |
| `DATABASE_URL` | — | строка подключения PostgreSQL (обязательна при `JOB_STORE=postgres`) |
| `QUEUE_BACKEND` / `REDIS_URL` | `inprocess` (в Docker `celery`) / `redis://redis:6379/0` | где выполняются пакеты: в процессе API или в Celery-воркере |
| `AUDIT_LOG` | `true` при `postgres` | журнал ответов в таблицу `classification_log` |
| `JOB_STORE_PATH` | `data/jobs.sqlite` | файл sqlite-хранилища (локальный режим) |
| `JOB_TTL_HOURS` | `24` | сколько хранить завершённые задачи до очистки |
| `ADMIN_API_KEY` | — | ключ `X-Admin-API-Key`; без него админ-эндпоинты отключены |
| `MAX_REQUEST_BODY_BYTES` | `1000000` | ограничение размера тела запроса (413) |
| `TRUSTED_PROXIES` | — | адреса/CIDR прокси, которым доверяем `X-Forwarded-For`; пусто — заголовок игнорируется |
| `RATE_LIMIT_ENABLED` | `true` | включает per-client rate limit (token bucket) |
| `RATE_LIMIT_FILE_RPS` / `RATE_LIMIT_FILE_BURST` | `1` / `3` | отдельный бакет `POST /classify/file`, считается в файлах |
| `RATE_LIMIT_RPS` / `RATE_LIMIT_BURST` | `10` / `20` | лимит публичных эндпоинтов (на клиента) |
| `RATE_LIMIT_ADMIN_RPS` / `..._BURST` | `1` / `2` | отдельный строгий лимит на `/admin/*` |

## Обновление каталога перечня

Регламент — `docs/catalog-update.md`. Коротко: `POST /admin/catalog`
(`dry_run=true` — проверка без установки) принимает CSV/JSON, валидирует
строки и атомарно заменяет `data.csv` + `data.json`; затем
`POST /admin/reindex` (или `auto_reindex=true`). Для проверки файла без API:
`python scripts/validate_catalog.py new_catalog.csv`.

## Качество и оценка

- Размеченный набор: `eval/labeled_examples.csv` (274 запроса: 166 с кодом, 108
  «вне перечня» — товар проверен по `data/data.csv`, 17 спорных помечены
  `disputed=1`).
- Прогон: `PYTHONPATH=api python eval/run_eval.py [--sweep]` → `eval/eval_report.md`
  (top-1/top-3, доля верного `not_found`, регрессия на точных названиях
  каталога, обе шкалы уверенности `confidence_raw`/`confidence` и ECE, свип
  порога `review`, разбор ошибок).
- Синтетический бенчмарк без разметки: `PYTHONPATH=api python eval/synthetic_eval.py`
- Абляция расхождений обучения и API (шортлист, fallback): `PYTHONPATH=api python eval/serving_ablation.py`
  → `eval/synthetic_report.md` (информационный, CI-гейт его не проверяет).
- Сравнение режимов `catalog` и `qdrant` на одном наборе:
  `PYTHONPATH=api python eval/backend_comparison.py --markdown eval/backend_comparison.md`
  → `eval/backend_comparison.csv` + отчёт; семантический прогон со свипом —
  `eval/run_eval.py --backend qdrant --output eval/eval_report_qdrant.md`.
  Итог эксп. 15: лексика 83.7% top-1 против 79.5% у семантики при ×4 латентности,
  поэтому режим по умолчанию — `catalog` (`eval/experiments.md`).
- Пороги под семантическую шкалу (эксп. 16): `PYTHONPATH=api python eval/threshold_sweep.py
  --backend qdrant --output eval/threshold_sweep_qdrant.csv` — снимает сырые уверенности
  за один проход и считает сетку порогов мгновенно. Выбранная точка `review = 0.25`
  поднимает точность честного `not_found` с 89.8% до 97.2% (ложных подсказок 13 → 3 из 108)
  ценой 15 верных подсказок; применяется через `QDRANT_REVIEW_FLOOR`/`QDRANT_OK_FLOOR`.
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
    job_store.py       # хранилища задач: memory / sqlite
    job_store_postgres.py  # PostgreSQL: задачи, прогресс, журнал классификаций
    dispatch.py        # отправка пакета: фоновая задача FastAPI или Celery
    tasks.py           # тело пакетной задачи (общее для API и воркера)
    worker.py          # Celery-приложение (брокер Redis)
    core/              # config, logging, correlation id, metrics
    middleware/        # observe, ограничение размера тела, rate limit
    services/          # classifier (catalog/mock/qdrant), catalog_io, source_lookup
  tests/               # API, классификатор, контракт, очередь, PostgreSQL, векторы, compose
frontend/              # демо-UI (React + Vite; собирается в frontend/dist и отдаётся на /ui/)
ml/                    # эмбеддеры, плотный индекс + Qdrant (dense.py), реранкер, лексика, обучение
  artifacts/           # готовые артефакты модели (входят в образ)
data/                  # каталог ПП № 1875 (data.csv + data.json + source_text.txt)
eval/                  # размеченный набор, метрики, эксперименты, чек-листы
docs/                  # api-contract.md, catalog-update.md
scripts/               # build_model, download_model, parse_pp1875, validate_catalog, load_test, docker_smoke
```

> Задачи пакетной обработки лежат в PostgreSQL (в Docker), поэтому переживают перезапуск и
> API, и воркера. Задачи, которые никто не трогал дольше `JOB_STALE_SECONDS` (воркер умер),
> помечаются `failed`, а не висят вечно; недоступная очередь даёт `503`.
