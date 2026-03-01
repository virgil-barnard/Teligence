#!/usr/bin/env bash
set -euo pipefail

command -v gh >/dev/null 2>&1 || exit 0

# Non-interactive login if token provided.
# `gh auth login --with-token` accepts a token on stdin.
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  host="${GITHUB_HOST:-github.com}"
  # Store auth in $HOME so `gh` works without env vars later.
  echo "$GITHUB_TOKEN" | gh auth login --hostname "$host" --with-token || true
  # Optionally configure git to use gh credential helper.
  gh auth setup-git --hostname "$host" || true
fi
