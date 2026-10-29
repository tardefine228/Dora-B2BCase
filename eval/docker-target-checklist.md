# Docker: проверка целевой машины — 26.10.2026

Проверено в `D:\b2b project\Dora-B2BCase`, Windows / Docker Desktop Linux engine.
Факты: [ops-evidence.json](data-review-2026-10-26/ops-evidence.json).

По умолчанию работают FastAPI, Celery worker, PostgreSQL, Redis и Qdrant.
`CLASSIFIER_BACKEND=catalog`, `VECTOR_SEARCH=true`: используется настоящий
multilingual-e5-small. `ML_ENABLED=false` не означает отсутствие векторного поиска.

| № | Проверка | Статус | Факт |
|---|---|---|---|
| 1 | Конфигурация Compose | PASS | docker compose config -q, exit 0 |
| 2 | Сборка образа | PREVIOUS PASS | Образ 7f38f7150e92, успешная сборка в submission-check-2026-10-26.md; в этом прогоне менялись данные, новая сборка не запускалась. |
| 3 | Запуск стека | PASS | Все 5 сервисов работают. |
| 4 | Готовность API | PASS | healthy; /health/ready = ready после перезапуска. |
| 5 | Готовность Qdrant | PASS | REST collections доступен, store_up=true. |
| 6 | /health | PASS | status=ok. |
| 7 | Классификация | PASS | Электрочайники → 27.51.24.110. |
| 8 | Сохранность Qdrant | PASS | Имена коллекций и points_count до/после перезапуска совпали. |
| 9 | Каталог после перезапуска | PASS | 1126 строк; SHA-256 совпадает с установленным каталогом. |
| 10 | Фактическая модель и реранкер | PASS | multilingual-e5-small, 384; vector-features, degraded=false. |
| 11 | Версии Qdrant | PASS | Клиент 1.14.3, сервер 1.14.1. |
| 12 | POST /admin/reindex | N/A | Для экспериментального ML_ENABLED=true. В текущем catalog режиме индекс обновляется автоматически; новая коллекция построена. |
| 13 | Размеры образов | PASS | API/worker 1.18 GB; Qdrant 78 MB (Docker compose images). |
| 14 | Ресурсы | PASS | API около 951 MiB/2 GiB, worker 970 MiB/2 GiB, Qdrant 70 MiB/768 MiB; замер при простое, не пиковое потребление. |
| 15 | Время с активной моделью | PASS | См. acceptance-after-pdf.json, 100 товаров 8.78 с. |
| 16 | Миграции | PASS | alembic_version=0002_feedback_comment. |
| 17 | Колонка comment | PASS | Подтверждена запросом information_schema. |
| 18 | Сроки хранения | PASS | dry_run: audit=90 дней, feedback=0, skipped=null; записи не удалялись. |
| 19 | Резервная копия и откат | PASS | catalog-backup.csv; откат восстановил прежний SHA-256, затем возвращён исправленный каталог. |
| 20 | Выгрузка вердиктов | PASS (разделённая проверка) | Живой export доступен; 12 интеграционных тестов сохранения, комментария и экспорта пройдены на отдельной тестовой БД. Рабочая БД тестовыми вердиктами не засорялась. |

## Повторный запуск

```powershell
Set-Location "D:\b2b project\Dora-B2BCase"
docker compose up -d --build
docker compose ps
python scripts/check_acceptance.py eval/labeled_examples.csv --output acceptance-report.json
```

## Шаги вручную (шаблон повторной проверки)

| # | Проверка | Команда | Ожидание | Факт (заполняется) |
| --- | --- | --- | --- | --- |
| 1 | Конфигурация compose | `docker compose config -q` | без ошибок | |
| 2 | Сборка образа API | `docker compose build` | сборка проходит; время/размер записать | |
| 3 | Запуск стека | `docker compose up -d` | оба сервиса подняты | |
| 4 | Healthcheck API | `docker compose ps` | `api` → `healthy` ≤ 90 с | |
| 5 | Qdrant readiness | `curl -fsS localhost:6333/healthz` | HTTP 200; у официального образа нет curl внутри контейнера | |
| 6 | Живой ответ | `curl -s localhost:8000/health` | `{"status":"ok"…}` | |
| 7 | Классификация | `curl -s -X POST localhost:8000/classify -H 'Content-Type: application/json' -d '{"name":"Электрочайники"}'` | код `27.51.24.110` | |
| 8 | Перезапуск Qdrant | `docker compose restart qdrant` → `curl localhost:6333/healthz` | данные volume `qdrant_data` сохранены (коллекция доступна) | |
| 9 | Перезапуск API с загруженным каталогом | `docker compose restart api` → `GET /admin/catalog` | `rows` не сброшены (volume `catalog_data`) | |
| 10 | Векторный реранкер | в логе сборки: `gate: артефакт …` / в логе старта: строка про реранкер | либо `gate: принят` и `reranker.tfidf.json` скорит, либо `gate: ОТКЛОНЁН` с причиной в `/app/ml/artifacts/dense-gate.json`; молчаливого «нет реранкера» без вердикта быть не должно | |
| 11 | Версии Qdrant | `docker compose logs api \| grep -i qdrant` | нет `Qdrant client version … is incompatible`; клиент закреплён под сервер (`qdrant-client>=1.14,<1.15` ↔ `v1.14.1`) | |
| 10 | Переиндексация (нужен `ML_ENABLED=true`) | `POST /admin/reindex` с `X-Admin-API-Key` | job `completed`, `indexed` ≈ числу строк каталога | |
| 11 | Размеры образов | `docker images` | записать `api` (и `qdrant`) | |
| 12 | RAM/CPU | `docker stats --no-stream` | API ≤ `API_MEMORY_LIMIT`, Qdrant ≤ `QDRANT_MEMORY_LIMIT` | |
| 13 | Ресурсы ML-режима | запуск с `ML_ENABLED=true` + модель в `./models` | RSS API после загрузки модели; время ответа `/classify` | |
| 14 | Миграции схемы применились сами | `docker compose logs api \| grep -i миграц` или `docker compose exec api python -c "from app.migrations_runner import current_revision; from app.core.config import settings; print(current_revision(settings.database_url))"` | ревизия `0002_feedback_comment`, ручных `alembic`-команд не требуется; на уже существующем volume таблицы не пересозданы | |
| 15 | Колонка `comment` в журнале вердиктов | `docker compose exec postgres psql -U dora -d dora -c "\d classification_feedback"` | есть `comment` — значит миграция 0002 применилась на живой базе | |
| 16 | Срок хранения журналов | `curl -X POST "localhost:8000/admin/maintenance/purge?dry_run=true" -H "X-Admin-API-Key: …"` | `audit_retention_days: 90`, `feedback_retention_days: 0`, `skipped: null`; удаление — тем же запросом без `dry_run` | |
| 17 | Резервная копия каталога и откат | `GET /admin/catalog/export` (файл сохраняется), затем `POST /admin/catalog/rollback` | `restored.version_id` непустой, `catalog.sha256` совпадает с записанным до загрузки | |
| 18 | Выгрузка вердиктов | `POST /feedback` (вердикт), затем `GET /admin/feedback/export` | файл `feedback-*.csv`, в нём строка вердикта, `comment` перенесён | |

## Режимы запуска

```bash
# каталоговый (по умолчанию) — без ML, маленький образ
docker compose up --build

# семантический — модель в ./models (например, models/bge-m3), ключ администратора
ML_ENABLED=true MODEL_NAME=./models/bge-m3 ADMIN_API_KEY=<секрет> \
  CLASSIFIER_BACKEND=qdrant docker compose up --build
```

После первого запуска ML-режима выполните `POST /admin/reindex` (curl или
Swagger) и проверьте `GET /admin/reindex/{job_id}` до статуса `completed`.

## Критерии приёмки

- [ ] `docker compose up --build` проходит без ручных правок;
- [ ] healthcheck обоих сервисов `healthy`;
- [ ] `POST /classify` возвращает реальный ответ каталога;
- [ ] данные Qdrant переживают `docker compose restart qdrant`;
- [ ] загруженный через `POST /admin/catalog` каталог переживает перезапуск;
- [ ] размеры образов и потребление RAM записаны в таблицу выше;
- [ ] ML-режим: `/admin/reindex` → `completed`, классификация работает, повторная
      переиндексация не оставляет «висящих» коллекций;
- [ ] миграции применились при старте без ручных команд, `comment` на месте;
- [ ] откат каталога возвращает предыдущую редакцию без перезапуска контейнера;
- [ ] `scripts/pg_backup.sh` создаёт дамп в `backups/`, а
      `scripts/pg_restore_check.sh` заканчивается `OK` (см. `docs/backup-restore.md`);
- [ ] `docker compose --profile monitoring up` поднимает Prometheus (9090) и
      Grafana (3000), дашборд «Dora B2B» и алерты видны (см. `docs/observability.md`);
- [ ] удаление артефакта реранкера при `ALLOW_DEGRADE=false` делает
      `/health/ready` = 503, а не молча меняет ответы.

## Семантический режим: отдельный свип порога

Пороги `ok`/`review` откалиброваны на лексических скорах каталогового режима и
для семантических скоров не годятся — после первого запуска с моделью (bge-m3)
свип нужно переснять и зафиксировать в `eval/experiments.md`:

```bash
# 1) проиндексировать каталог моделью (Docker, ML_ENABLED=true)
CLASSIFIER_BACKEND=qdrant MODEL_NAME=./models/bge-m3 docker compose up --build
curl -X POST localhost:8000/admin/reindex -H "X-Admin-API-Key: $ADMIN_API_KEY"

# 2) переснять свип и метрики уже на семантическом реранкере
PYTHONPATH=api CLASSIFIER_BACKEND=qdrant python eval/run_eval.py --sweep

# 3) сравнить с каталоговым режимом и, если плато то же (0.16–0.18), оставить
#    REVIEW_CONFIDENCE_FLOOR = 0.165, иначе — пересобрать модель с новым floor
```

Отдельно стоит прогнать синтетический бенчмарк на семантическом режиме
(`PYTHONPATH=api CLASSIFIER_BACKEND=qdrant python eval/synthetic_eval.py`) — он
показывает, закрывает ли модель лексически нерешаемые кейсы из
`eval/experiments.md` («Настольная лампа», «Смарт-часы», «Компьютер настольный»).

## Вердикт специалиста по спорным строкам

17 строк помечены в `eval/labeled_examples.csv` как `disputed=1` — это места, где
разметку должен подтвердить человек по нацрежиму. Список с текущими ответами
модели печатает таблица «Ошибки и спорные случаи» в `eval/eval_report.md`;
получить только спорные строки:

```bash
python - <<'HELPER'
import csv
rows = [r for r in csv.DictReader(open('eval/labeled_examples.csv', encoding='utf-8-sig'), delimiter=';') if r['disputed'] == '1']
print(len(rows), 'спорных строк')
for r in rows:
    print(f"- {r['query']}: ожидается {r['expected_okpd2'] or '(нет в перечне)'} — {r['notes']}")
HELPER
```

| Запрос | Ожидалось | Ответ модели | Вердикт специалиста | Комментарий |
| --- | --- | --- | --- | --- |
| заполняется по колонкам из `eval/eval_report.md` | | | | |

Решения переносятся в `eval/labeled_examples.csv`: `disputed` → `0`, при
необходимости правится `expected_okpd2` и заметка. После правок — повторный
прогон `eval/run_eval.py --sweep` (метрики и, возможно, порог `review` меняются).

## Границы этой проверки

Интерфейс: http://localhost:8000/ui/. Ключ администратора берётся из локального `.env`; в отчёты не включён.
При намеренном перезапуске Qdrant был краткий отказ соединения; после восстановления проверена готовность без degraded-режима.
Техническая проверка не подтверждает юридическую актуальность нормативного каталога.
