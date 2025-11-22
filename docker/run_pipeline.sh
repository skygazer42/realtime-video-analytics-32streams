#!/usr/bin/env bash
set -euo pipefail

# Allow override via PIPELINE_CONFIG; fall back to common mount points.
CONFIG_PATH="${PIPELINE_CONFIG:-/app/config/sample-pipeline.yaml}"
if [[ ! -f "${CONFIG_PATH}" && -f "/data/config/pipeline.yaml" ]]; then
  CONFIG_PATH="/data/config/pipeline.yaml"
fi

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[pipeline] configuration not found: ${CONFIG_PATH}" >&2
  exit 1
fi

exec python scripts/run_pipeline.py --config "${CONFIG_PATH}" "$@"
