# Чек-лист проверки Docker на целевой машине

Разработка велась в песочнице **без Docker**, поэтому образы и compose здесь
не запускались. Всё нижеперечисленное проверяется на целевой машине одной
командой и фиксируется в этом файле.

## Быстрая проверка

```bash
bash scripts/docker_smoke.sh --build   # сборка + все проверки ниже
```

## Шаги вручную

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
- [ ] откат каталога возвращает предыдущую редакцию без перезапуска контейнера.

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
