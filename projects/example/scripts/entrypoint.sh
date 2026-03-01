#!/usr/bin/env bash
set -euo pipefail

export HOME="${HOME:-/tmp/home}"
mkdir -p "$HOME"

mkdir -p "${WORKSPACE:-/workspace}"

# Bootstrap is designed to be idempotent.
if [[ -d /usr/local/lib/agent/bootstrap.d ]]; then
  for f in /usr/local/lib/agent/bootstrap.d/*.sh; do
    [[ -e "$f" ]] || continue
    "$f"
  done
fi

# Default to whatever CMD is
exec "$@"
