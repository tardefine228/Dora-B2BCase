# Наблюдаемость: логи, метрики, алерты, дашборд

Дата: 2026-10-26. Закрывает OPS-6 и рантайм-часть OPS-3. Все артефакты в репозитории,
внешних SaaS-настроек не требуется.

## 1. Логи: `LOG_FORMAT=text|json`

Формат выбирается переменной окружения (`api/app/core/log_config.py`):

| `LOG_FORMAT` | Когда | Вид строки |
| --- | --- | --- |
| `text` | локальный запуск uvicorn, значение по умолчанию вне Docker | `2026-10-26T12:00:00+0000 INFO [dora.api] rid=abc …` |
| `json` | docker compose (задан по умолчанию в `docker-compose.yml`) | один JSON-объект на строку |

JSON-запись содержит фиксированные поля:

```json
{
  "ts": "2026-10-26T12:00:00.123456+00:00",
  "level": "INFO",
  "logger": "dora.api",
  "request_id": "8f1c…",
  "message": "POST /classify status=200 latency_ms=12.40 content_length=84"
}
```

- `request_id` проставляется middleware (`X-Request-ID`, валидируется; свой не подменить) и
  виден в каждой строке запроса, включая фоновые задачи воркера.
- Исключения попадают в поле `exc_info` (полный трейсбек), пользовательские `extra=` — в `extra`.
- Тот же формат получают логгеры uvicorn: разнобоя «JSON приложения / текст сервера» нет.
- Доступ-лог uvicorn отключён (`WARNING`): доступ пишет `ObserveMiddleware` одной
  структурированной строкой со статусом, маршрутом-шаблоном и латентностью.
- Локально вернуть привычный текст: `LOG_FORMAT=text PYTHONPATH=api uvicorn app.main:app`.

## 2. Метрики: `GET /metrics`

Prometheus text exposition по умолчанию; `Accept: application/json` отдаёт прежний
JSON-снимок для совместимости. Источник истины по именам и лейблам —
`api/app/core/metrics.py`. Основные ряды:

| Метрика | Тип | Назначение |
| --- | --- | --- |
| `dora_http_requests_total{path,method,status}` | counter | трафик по шаблону маршрута и коду ответа |
| `dora_http_request_duration_seconds{path}` | histogram | латентность, бакеты от 5 мс до 30 с |
| `dora_http_rate_limited_total{path}` | counter | отказы 429 |
| `dora_classifications_total{mode,backend,status}` | counter | ответы классификатора (`ok`/`review`/`not_found`) |
| `dora_batch_jobs_total{event}` | counter | жизненный цикл пакетных задач |
| `dora_classifier_up`, `dora_embedder_up`, `dora_vector_store_up` | gauge | готовность компонентов |
| `dora_embedder_info{tag}`, `dora_vector_store_info{store}` | gauge | кто реально обслуживает запросы |
| `dora_build_info{version,classifier_backend,job_store,queue_backend}` | gauge | версия и конфигурация процесса |

Счётчики живут в процессе: инференс пакетов идёт в воркере, и его счётчики видны в его
логах, а не в `/metrics` API (события смены статуса задач при этом есть и в API).

## 3. Стек мониторинга (опциональный профиль compose)

```bash
docker compose --profile monitoring up --build
# Prometheus: http://127.0.0.1:9090  (пароль не нужен, только loopback)
# Grafana:    http://127.0.0.1:3000  (admin / $GRAFANA_ADMIN_PASSWORD, по умолчанию dora)
```

- Профиль `monitoring` не запускается обычным `docker compose up` — базовый стек не меняет
  ни объём памяти, ни число контейнеров.
- Prometheus (`deploy/monitoring/prometheus/prometheus.yml`) скрейпит `api:8000/metrics`
  раз в 15 с; правила — `rules/dora-alerts.yml`.
- Grafana сама провижинит datasource (`uid: dora-prometheus`) и дашборд
  `Dora B2B — сервис и классификация` из read-only монтирований; правки через UI запрещены
  (`allowUiUpdates: false`) — панели живут в git.
- Порты опубликованы только на `127.0.0.1`; наружу — через обратный прокси с TLS.
- Лимиты и ретенция настраиваются в `.env` (`PROMETHEUS_*`, `GRAFANA_*`,
  `PROMETHEUS_RETENTION=15d` по умолчанию).

## 4. Алерты

Файл `deploy/monitoring/prometheus/rules/dora-alerts.yml`; тест
`api/tests/test_observability_assets.py` падает, если правило ссылается на метрику,
которой нет в коде.

| Алерт | Severity | Условие | Что проверять |
| --- | --- | --- | --- |
| `DoraClassifierDown` | critical | `dora_classifier_up == 0` 2 мин | артефакт реранкера, `ALLOW_DEGRADE`, логи старта; `/health/ready` = 503 |
| `DoraEmbedderDegraded` | warning | векторы включены, но `dora_embedder_up == 0` 5 мин | тег в `dora_embedder_info`, каталог модели `MODEL_NAME` |
| `DoraVectorStoreDown` | warning | `dora_vector_store_up == 0` 5 мин | контейнер qdrant, том `qdrant_data` |
| `DoraHighErrorRate` | warning | доля 5xx > 5 % при трафике > 0.1 rps, 5 мин | JSON-логи по `request_id` |
| `DoraHighLatencyP95` | warning | p95 > 1 с, 5 мин | CPU/память, семафор `MAX_CONCURRENT_INFERENCES` |
| `DoraBatchJobsFailing` | warning | есть `event="failed"` за 15 мин | журнал воркера (`dora.worker`) |
| `DoraRateLimitSaturated` | info | доля 429 > 25 %, 10 мин | `TRUSTED_PROXIES`, лимиты `RATE_LIMIT_*` |

Prometheus по умолчанию не шлёт уведомления сам: подключите Alertmanager в вашей сети
(или правило в корпоративный мониторинг) — файл правил совместим со стандартным
`rule_files` без изменений.

## 5. Связка с политикой деградации (OPS-3)

- Сборка образа: сбой ML-шага роняет build, если build-arg `ALLOW_DEGRADE=false`
  (значение по умолчанию, `api/Dockerfile`).
- Рантайм: та же переменная в окружении контейнера. При `false` отсутствие
  калиброванного артефакта реранкера поднимает `TrainedArtifactUnavailable`,
  `/health/ready` отвечает 503, `dora_classifier_up` = 0 и срабатывает
  `DoraClassifierDown`. Тихой подмены на эвристический скоринг с другими порогами нет.
- `EMBEDDER_BACKEND=auto` при откате эмбеддера пишет явный warning, а метрика
  `dora_embedder_up` и алерт `DoraEmbedderDegraded` делают откат видимым. Явный выбор
  бэкенда (`sentence-transformers`/`spacy`/`tfidf`) не деградирует молча вообще —
  цепочка фолбэков обрезается до выбранного.

## 6. Воспроизведение и тесты

```bash
# Форматтеры и политика деградации
PYTHONPATH=api pytest api/tests/test_logging.py api/tests/test_degradation_guard.py -q
# Контракт мониторинга (метрики, алерты, дашборд, профиль compose)
PYTHONPATH=api pytest api/tests/test_observability_assets.py -q
# Экспозиция метрик в API
PYTHONPATH=api pytest api/tests/test_metrics_prometheus.py -q
```
