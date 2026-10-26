# Резервное копирование и восстановление PostgreSQL (OPS-5)

Дата: 2026-10-26. БД `dora` хранит пакетные задачи (`jobs`), журнал запросов
(`classification_log`) и вердикты специалистов (`classification_feedback`). Каталог
товаров живёт в томе `catalog_data` и в git (`data/data.csv`) и этим процессом не
покрывается; Qdrant и Redis восстановимы: индекс переиндексируется, очередь эфемерна.

## 1. Бэкап

```bash
scripts/pg_backup.sh
# -> backups/dora-postgres-20261026T021500Z.sql.gz
```

- Дамп — plain SQL + gzip, флаги `--no-owner --clean --if-exists`: роли в дамп не
  попадают, восстановление в чистую БД не требует тех же пользователей.
- Запись атомарная: сначала `*.sql.gz.tmp`, затем переименование — прерванный прогон
  не оставит битый «последний» файл.
- Каталог `backups/` в `.gitignore` (это операционные данные, не код). Другой каталог:
  `BACKUP_DIR=/mnt/nfs/dora scripts/pg_backup.sh`.
- Рекомендуемый cron на целевой машине (ежедневно 02:15, хранение 14 суток):

  ```cron
  15 2 * * * cd /opt/dora && scripts/pg_backup.sh && find backups -name '*.sql.gz' -mtime +14 -delete
  ```

## 2. Проверка восстановления

Бэкап без проверки восстановления — гипотеза, а не бэкап. Скрипт поднимает в том же
контейнере postgres временную БД `dora_restore_check`, грузит туда дамп и сверяет с
боевой:

```bash
scripts/pg_restore_check.sh                          # самый свежий дамп
scripts/pg_restore_check.sh backups/dora-postgres-20261026T021500Z.sql.gz
```

Сверяются:

1. ревизия alembic (`alembic_version`) — схема копии обязана совпасть с источником;
2. число строк в `jobs`, `classification_log`, `classification_feedback`;
3. сам факт восстановления (psql с `ON_ERROR_STOP=1` — любая ошибка роняет скрипт).

Временная БД удаляется всегда (`trap cleanup EXIT`), включая сценарий ошибки.
Ненулевой код выхода — сигнал для cron/мониторинга. Рекомендация: гонять проверку
еженедельно и после каждого обновления образа.

## 3. Аварийное восстановление на чистой БД

```bash
# 1. Поднять пустой postgres штатным compose (том либо новый, либо очищенный)
docker compose up -d postgres
# 2. API при старте сам применит миграции; для чистого восстановления из дампа:
gunzip -c backups/dora-postgres-<timestamp>.sql.gz | \
  docker compose exec -T postgres psql -U dora -d dora -v ON_ERROR_STOP=1
# 3. Поднять остальной стек
docker compose up -d
curl -fsS http://127.0.0.1:8000/health/ready
```

Дамп с `--clean --if-exists` можно накатывать и на непустую БД: объекты будут
пересозданы. Вердикты (`classification_feedback`) по политике хранения не удаляются
(`FEEDBACK_RETENTION_DAYS=0`), поэтому попадают в каждый дамп целиком.

## 4. Лимиты памяти

Все лимиты параметризованы и задокументированы в `.env.example`; значения по умолчанию
совпадают с `docker-compose.yml`:

| Сервис | Переменная | Дефолт |
| --- | --- | --- |
| api | `API_MEMORY_LIMIT` | 2g |
| worker | `WORKER_MEMORY_LIMIT` | 2g |
| postgres | `POSTGRES_MEMORY_LIMIT` | 256m |
| redis | `REDIS_MEMORY_LIMIT` | 128m |
| qdrant | `QDRANT_MEMORY_LIMIT` | 768m |
| prometheus (профиль `monitoring`) | `PROMETHEUS_MEMORY_LIMIT` | 512m |
| grafana (профиль `monitoring`) | `GRAFANA_MEMORY_LIMIT` | 256m |

Для BGE-M3 (тяжёлый эмбеддер) api/worker нужны 4g — это отмечено в `.env.example`;
postgres 256m достаточно при текущем профиле нагрузки (журналы, не транзакционная OLTP).

## 5. CI-страховка

Джоба `postgres-restore` в `.github/workflows/lint.yml` на каждом PR поднимает реальный
PostgreSQL 16 (та же мажорная версия, что в compose), применяет миграции, пишет строку,
делает `pg_dump`, восстанавливает в отдельную БД и сверяет ревизию и количества строк.
Регрессия в схеме или процедуре восстановления не доедет до целевой машины.

## 6. Пароль БД

`POSTGRES_PASSWORD` (дефолт `dora`) подставляется compose и в строку `DATABASE_URL`.
Для прода задайте собственный пароль в `.env` — менять нужно только эту переменную.
Файл `.env` в git не коммитится.
