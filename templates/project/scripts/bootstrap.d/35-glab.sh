#!/usr/bin/env bash
set -euo pipefail

command -v glab >/dev/null 2>&1 || exit 0

if [[ -n "${GITLAB_TOKEN:-}" ]]; then
  host="${GITLAB_HOST:-gitlab.com}"
  echo "$GITLAB_TOKEN" | glab auth login --hostname "$host" --stdin || true
fi
