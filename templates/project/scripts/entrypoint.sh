#!/usr/bin/env bash
set -euo pipefail

export HOME="${HOME:-/tmp/home}"
mkdir -p "$HOME"

mkdir -p "${WORKSPACE:-/workspace}"

BOOTSTRAP_DIR="/usr/local/lib/agent/bootstrap.d"

# Source secrets in-process so subsequent bootstrap scripts can use them.
if [[ -f "${BOOTSTRAP_DIR}/00-load-secrets.sh" ]]; then
  # shellcheck disable=SC1090
  source "${BOOTSTRAP_DIR}/00-load-secrets.sh" || true
fi

# Run the rest of bootstrap steps (idempotent-ish)
if [[ -d "${BOOTSTRAP_DIR}" ]]; then
  for f in "${BOOTSTRAP_DIR}"/*.sh; do
    [[ -e "$f" ]] || continue
    [[ "$(basename "$f")" == "00-load-secrets.sh" ]] && continue
    "$f" || true
  done
fi

exec "$@"
