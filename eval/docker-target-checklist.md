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

Интерфейс: http://localhost:8000/ui/. Ключ администратора берётся из локального `.env`; в отчёты не включён.
При намеренном перезапуске Qdrant был краткий отказ соединения; после восстановления проверена готовность без degraded-режима.
Техническая проверка не подтверждает юридическую актуальность нормативного каталога.
