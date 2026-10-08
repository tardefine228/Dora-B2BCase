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
| 5 | Healthcheck Qdrant | `docker compose ps` | `qdrant` → `healthy` | |
| 6 | Живой ответ | `curl -s localhost:8000/health` | `{"status":"ok"…}` | |
| 7 | Классификация | `curl -s -X POST localhost:8000/classify -H 'Content-Type: application/json' -d '{"name":"Электрочайники"}'` | код `27.51.24.110` | |
| 8 | Перезапуск Qdrant | `docker compose restart qdrant` → `curl localhost:6333/healthz` | данные volume `qdrant_data` сохранены (коллекция доступна) | |
| 9 | Перезапуск API с загруженным каталогом | `docker compose restart api` → `GET /admin/catalog` | `rows` не сброшены (volume `catalog_data`) | |
| 10 | Переиндексация (нужен `INSTALL_ML=true`) | `POST /admin/reindex` с `X-Admin-API-Key` | job `completed`, `indexed` ≈ числу строк каталога | |
| 11 | Размеры образов | `docker images` | записать `api` (и `qdrant`) | |
| 12 | RAM/CPU | `docker stats --no-stream` | API ≤ `API_MEMORY_LIMIT`, Qdrant ≤ `QDRANT_MEMORY_LIMIT` | |
| 13 | Ресурсы ML-режима | запуск с `INSTALL_ML=true` + модель в `./models` | RSS API после загрузки модели; время ответа `/classify` | |

## Режимы запуска

```bash
# каталоговый (по умолчанию) — без ML, маленький образ
docker compose up --build

# семантический — модель в ./models (например, models/bge-m3), ключ администратора
INSTALL_ML=true MODEL_NAME=./models/bge-m3 ADMIN_API_KEY=<секрет> \
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
      переиндексация не оставляет «висящих» коллекций.
