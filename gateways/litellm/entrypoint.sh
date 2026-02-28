#!/bin/sh
set -eu

load_env_file() {
  f="$1"
  [ -f "$f" ] || return 0
  # Export everything defined in the env file for the current process.
  # shellcheck disable=SC1090
  set -a
  . "$f"
  set +a
}

# Secrets are mounted as files by Docker Compose.
load_env_file /run/secrets/litellm_env
load_env_file /run/secrets/providers_env

PORT="${LITELLM_PORT:-4000}"
WORKERS="${LITELLM_NUM_WORKERS:-1}"

exec litellm --config /app/config.yaml --port "$PORT" --num_workers "$WORKERS"
