#!/usr/bin/env bash
set -euo pipefail
command -v gh >/dev/null 2>&1 || exit 0

# Non-interactive GitHub CLI auth (optional). If you prefer no persistent auth, leave it.
if [[ -n "${GITHUB_TOKEN:-}" && "${GITHUB_TOKEN}" != "ghp_REPLACE_ME" ]]; then
  echo "$GITHUB_TOKEN" | gh auth login --with-token >/dev/null 2>&1 || true
fi
