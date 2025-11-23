#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${DASHBOARD_CONFIG:-/app/config/sample-pipeline.yaml}"
if [[ ! -f "${CONFIG_PATH}" && -f "/data/config/pipeline.yaml" ]]; then
  CONFIG_PATH="/data/config/pipeline.yaml"
fi

HOST="${DASHBOARD_HOST:-0.0.0.0}"
PORT="${DASHBOARD_PORT:-8080}"

ARGS=(--port "${PORT}" --host "${HOST}")
if [[ -f "${CONFIG_PATH}" ]]; then
  ARGS+=(--config "${CONFIG_PATH}")
else
  echo "[dashboard] configuration not found (${CONFIG_PATH}), dashboard will run without Kafka auto-config."
fi

exec python scripts/run_dashboard.py "${ARGS[@]}" "$@"
