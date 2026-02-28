#!/usr/bin/env bash
set -euo pipefail
command -v glab >/dev/null 2>&1 || exit 0

if [[ -n "${GITLAB_TOKEN:-}" && "${GITLAB_TOKEN}" != "glpat_REPLACE_ME" ]]; then
  host="${GITLAB_HOST:-gitlab.com}"
  echo "$GITLAB_TOKEN" | glab auth login --hostname "$host" --stdin >/dev/null 2>&1 || true
fi
