#!/usr/bin/env bash
# Проверка Docker-развёртывания на целевой машине (см. eval/docker-target-checklist.md).
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

log "3. Ожидание API и Qdrant (до 90 сек)"
# The official Qdrant image is minimal and does not include curl. Check its
# /healthz endpoint from the host rather than via an in-container healthcheck.
for _ in $(seq 1 45); do
  api_health=$(docker inspect --format='{{.State.Health.Status}}' "$($COMPOSE ps -q api)" 2>/dev/null || echo unknown)
  if curl -fsS "http://localhost:${QDRANT_PORT:-6333}/healthz" >/dev/null 2>&1; then
    qdrant_health=healthy
  else
    qdrant_health=starting
  fi
  [[ "$api_health" == "healthy" && "$qdrant_health" == "healthy" ]] && break
  sleep 2
done
echo "api=$api_health qdrant=$qdrant_health"
[[ "$api_health" == "healthy" && "$qdrant_health" == "healthy" ]] || { echo "FAIL: healthcheck"; exit 1; }

log "4. Ответ API"
$COMPOSE exec -T api python -c 'import json, urllib.request; r=json.load(urllib.request.urlopen("http://127.0.0.1:8000/health/ready")); assert r.get("calibrated") and r.get("model_catalog_current"), r; print("Trained model matches catalog")'
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

log "5. Данные Qdrant переживают перезапуск"
$COMPOSE restart qdrant >/dev/null
sleep 5
curl -fsS "http://localhost:${QDRANT_PORT:-6333}/healthz" >/dev/null && echo "qdrant restarted OK"
if [[ -n "${ADMIN_API_KEY:-}" && "${INSTALL_ML:-false}" == "true" ]]; then
  curl -fsS -X POST "http://localhost:${API_PORT:-8000}/admin/reindex" \
    -H "X-Admin-API-Key: ${ADMIN_API_KEY}" | head -c 300; echo
fi

log "6. Размер образов"
docker images --format 'table {{.Repository}}\t{{.Tag}}\t{{.Size}}' | grep -Ei 'api|qdrant|REPOSITORY' || true

log "7. Потребление RAM/CPU"
$COMPOSE ps
docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.CPUPerc}}' || true

log "Готово. Заполните результатами eval/docker-target-checklist.md"
