#!/usr/bin/env bash
# OPS-5: prove that a backup actually restores.
#
# Loads the given dump (or the newest file in ./backups) into a throwaway
# database inside the same postgres container, then asserts:
#   1. the alembic revision of the restored copy matches the live database;
#   2. every application table has the same row count on both sides.
# The scratch database is always dropped at the end. Exit code is non-zero on
# any mismatch or restore error (ON_ERROR_STOP), so cron/CI can alert on it.
#
# Usage:
#   scripts/pg_restore_check.sh                         # newest backup
#   scripts/pg_restore_check.sh backups/dora-postgres-20261026T021500Z.sql.gz
set -euo pipefail
cd "$(dirname "$0")/.."

DB_USER="${POSTGRES_USER:-dora}"
DB_NAME="${POSTGRES_DB:-dora}"
SCRATCH="${SCRATCH_DB:-dora_restore_check}"
TABLES=(jobs classification_log classification_feedback)

dump="${1:-}"
if [ -z "$dump" ]; then
  dump="$(ls -t backups/dora-postgres-*.sql.gz 2>/dev/null | head -1 || true)"
fi
[ -n "$dump" ] && [ -f "$dump" ] || {
  echo "Дамп не найден (передайте путь или сначала запустите scripts/pg_backup.sh)" >&2
  exit 1
}

psql_exec() {
  docker compose exec -T postgres psql -U "$DB_USER" -v ON_ERROR_STOP=1 "$@"
}

cleanup() {
  psql_exec -d postgres -q -c "DROP DATABASE IF EXISTS $SCRATCH" >/dev/null
}
trap cleanup EXIT

echo "==> Готовлю чистую БД $SCRATCH"
psql_exec -d postgres -q \
  -c "DROP DATABASE IF EXISTS $SCRATCH" \
  -c "CREATE DATABASE $SCRATCH"

echo "==> Восстанавливаю $dump"
gunzip -c "$dump" | psql_exec -d "$SCRATCH" -q

revision_src="$(psql_exec -d "$DB_NAME" -t -Ac 'SELECT version_num FROM alembic_version')"
revision_dst="$(psql_exec -d "$SCRATCH" -t -Ac 'SELECT version_num FROM alembic_version')"
[ -n "$revision_dst" ] || { echo "В восстановленной БД нет alembic_version" >&2; exit 1; }
[ "$revision_src" = "$revision_dst" ] || {
  echo "Ревизии не совпадают: источник=$revision_src копия=$revision_dst" >&2
  exit 1
}
echo "==> Ревизия совпадает: $revision_dst"

for table in "${TABLES[@]}"; do
  count_src="$(psql_exec -d "$DB_NAME" -t -Ac "SELECT count(*) FROM $table")"
  count_dst="$(psql_exec -d "$SCRATCH" -t -Ac "SELECT count(*) FROM $table")"
  [ "$count_src" = "$count_dst" ] || {
    echo "Число строк в $table differs: источник=$count_src копия=$count_dst" >&2
    exit 1
  }
  echo "==> $table: $count_dst строк"
done

echo "OK: дамп $dump полностью восстановлен и совпал с источником"
