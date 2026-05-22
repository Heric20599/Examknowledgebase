#!/usr/bin/env sh
# Invoke GET /internal/cron/ping every 15 minutes via crontab.
# Requires: API running (uvicorn), CRON_SECRET in .env and exported for cron.

set -eu

API_BASE_URL="${API_BASE_URL:-http://127.0.0.1:8000}"
API_BASE_URL="${API_BASE_URL%/}"

if [ -z "${CRON_SECRET:-}" ]; then
  echo "CRON_SECRET is not set" >&2
  exit 1
fi

curl -fsS -m 60 \
  -H "X-Cron-Secret: ${CRON_SECRET}" \
  "${API_BASE_URL}/internal/cron/ping"
