#!/usr/bin/env bash
set -euo pipefail
command -v git >/dev/null 2>&1 || exit 0

repo_url="${REPO_URL:-}"
repo_dir="${REPO_DIR:-/workspace/repo}"
repo_branch="${REPO_BRANCH:-}"

[[ -n "$repo_url" ]] || exit 0

if [[ ! -d "$repo_dir/.git" ]]; then
  mkdir -p "$(dirname "$repo_dir")"
  if [[ -n "$repo_branch" ]]; then
    git clone --branch "$repo_branch" "$repo_url" "$repo_dir"
  else
    git clone "$repo_url" "$repo_dir"
  fi
fi
