#!/usr/bin/env bash
# OPS-5: logical backup of the PostgreSQL store (jobs, audit log, feedback).
#
# Writes a gzipped plain-SQL dump into ./backups (host directory, outside the
# postgres volume). Plain SQL is chosen over the custom format so the dump stays
# inspectable with zcat and restores into any PostgreSQL 16 instance without a
# matching pg_restore build. --no-owner keeps roles out of the dump, which lets
# scripts/pg_restore_check.sh reload it into a scratch database.
#
# Usage:
#   scripts/pg_backup.sh                 # backups/dora-postgres-<UTC timestamp>.sql.gz
#   BACKUP_DIR=/mnt/backups scripts/pg_backup.sh
#
# Cron example (daily 02:15, keep 14 dumps):
#   15 2 * * * cd /opt/dora && scripts/pg_backup.sh && find backups -name '*.sql.gz' -mtime +14 -delete
set -euo pipefail
cd "$(dirname "$0")/.."

DB_USER="${POSTGRES_USER:-dora}"
DB_NAME="${POSTGRES_DB:-dora}"
BACKUP_DIR="${BACKUP_DIR:-backups}"
mkdir -p "$BACKUP_DIR"

stamp="$(date -u +%Y%m%dT%H%M%SZ)"
out="$BACKUP_DIR/dora-postgres-$stamp.sql.gz"

# Atomic publish: a crashed run must never leave a half-written "latest" dump.
docker compose exec -T postgres \
  pg_dump -U "$DB_USER" -d "$DB_NAME" --no-owner --clean --if-exists \
  | gzip -9 >"$out.tmp"
mv "$out.tmp" "$out"

echo "$out"
