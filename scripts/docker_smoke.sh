#!/usr/bin/env bash
# Проверка Docker-развёртывания на целевой машине (см. eval/docker-target-checklist.md; обязательный минимум — шаги 3b и 5).
# Использование: bash scripts/docker_smoke.sh [--build]
set -euo pipefail

COMPOSE=${COMPOSE:-"docker compose"}
BUILD_ARG=${1:-}

log() { printf '\n=== %s ===\n' "$*"; }

log "1. Конфигурация compose"
$COMPOSE config -q
echo "config OK"

log "2. Сборка и запуск"
if [[ "$BUILD_ARG" == "--build" ]]; then
  $COMPOSE up -d --build
else
  $COMPOSE up -d
fi

log "3. Ожидание API (до 4 мин: загрузка эмбеддера и векторного индекса)"
api_container=$($COMPOSE ps -q api | head -1)
for _ in $(seq 1 120); do
  api_health=$(docker inspect --format='{{.State.Health.Status}}' "$api_container" 2>/dev/null || echo unknown)
  [[ "$api_health" == "healthy" ]] && break
  sleep 2
done
echo "api=$api_health"
[[ "$api_health" == "healthy" ]] || { echo "FAIL: healthcheck"; $COMPOSE logs --tail=40 api; exit 1; }
curl -fsS "http://localhost:${QDRANT_PORT:-6333}/healthz" >/dev/null && echo "qdrant OK"
for service in worker postgres redis; do
  [[ -n "$($COMPOSE ps -q --status running $service)" ]] || { echo "FAIL: $service не запущен"; exit 1; }
done
echo "worker/postgres/redis запущены"

log "3b. Что реально обслуживает запросы (эмбеддер, хранилище, очередь)"
ready=$(curl -fsS "http://localhost:${API_PORT:-8000}/health/ready")
echo "$ready"
grep -q '"enabled":true' <<<"$ready" || { echo "FAIL: векторный поиск выключен"; exit 1; }
grep -q '"job_store":"postgres"' <<<"$ready" || { echo "FAIL: хранилище задач не PostgreSQL"; exit 1; }
grep -q '"queue_backend":"celery"' <<<"$ready" || { echo "FAIL: очередь не Celery"; exit 1; }

log "3c. Схема БД: миграции применились сами (BE-5)"
# Руками `alembic upgrade` на целевой машине никто не запускает: ревизию ставит
# сам API при старте. Пустая строка в ответе = схема не под миграциями.
revision=$($COMPOSE exec -T api python -c \
  "from app.core.config import settings; from app.migrations_runner import current_revision; print(current_revision(settings.database_url) or '')" \
  2>/dev/null | tr -d '\r')
[[ -n "$revision" ]] || { echo "FAIL: в БД нет ревизии схемы — миграции не применились"; exit 1; }
echo "schema revision: $revision"

log "4. Ответ API"
curl -fsS "http://localhost:${API_PORT:-8000}/health" | head -c 300; echo
curl -fsS -X POST "http://localhost:${API_PORT:-8000}/classify" \
  -H 'Content-Type: application/json' \
  -d '{"name":"Электрочайники"}' | head -c 400; echo
ui_html=$(curl -fsS "http://localhost:${API_PORT:-8000}/ui/")
grep -q 'id="root"' <<<"$ui_html" || {
  echo "FAIL: the demo UI was not served from /ui/"
  exit 1
}
# The page must reference the compiled bundle, not the TypeScript sources.
ui_script=$(grep -o 'src="/ui/assets/[^"]*\.js"' <<<"$ui_html" | head -1 | sed 's/^src="//; s/"$//' || true)
[[ -n "$ui_script" ]] || { echo "FAIL: /ui/ does not reference a built bundle (src=/ui/assets/*.js)"; exit 1; }
ui_type=$(curl -fsS -o /dev/null -w '%{content_type}' "http://localhost:${API_PORT:-8000}${ui_script}")
grep -qi 'javascript' <<<"$ui_type" || { echo "FAIL: ${ui_script} served as '${ui_type}'"; exit 1; }
echo "demo UI OK (${ui_script}, ${ui_type})"

log "5. Пакет через Redis -> Celery-воркер -> PostgreSQL"
job=$(curl -fsS -X POST "http://localhost:${API_PORT:-8000}/classify/batch" \
  -H 'Content-Type: application/json' \
  -d '{"items":["Ноутбук Lenovo","Электрочайники","Станок токарный с ЧПУ"]}')
status_url=$(python3 -c 'import json,sys; print(json.loads(sys.argv[1])["status_url"])' "$job")
for _ in $(seq 1 30); do
  state=$(curl -fsS "http://localhost:${API_PORT:-8000}${status_url}" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')
  [[ "$state" == "completed" || "$state" == "failed" ]] && break
  sleep 1
done
echo "batch: $state"
[[ "$state" == "completed" ]] || { $COMPOSE logs --tail=30 worker; exit 1; }
$COMPOSE exec -T postgres psql -U dora -d dora -tAc "select count(*) || ' записей в classification_log' from classification_log"

log "5b. Состояние переживает перезапуск API"
$COMPOSE restart api >/dev/null
for _ in $(seq 1 90); do
  curl -fsS "http://localhost:${API_PORT:-8000}/health" >/dev/null 2>&1 && break
  sleep 2
done
curl -fsS "http://localhost:${API_PORT:-8000}${status_url}" | head -c 120; echo

log "6. Размер образов"
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' | grep -Ei 'api|qdrant|REPOSITORY' || true

log "7. Потребление RAM/CPU"
$COMPOSE ps
docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' || true

log "Готово. Заполните результатами eval/docker-target-checklist.md"
