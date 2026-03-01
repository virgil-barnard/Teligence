#!/usr/bin/env bash
set -euo pipefail

command -v git >/dev/null 2>&1 || exit 0

if [[ -n "${GIT_USER_NAME:-}" ]]; then
  git config --global user.name "$GIT_USER_NAME"
fi

if [[ -n "${GIT_USER_EMAIL:-}" ]]; then
  git config --global user.email "$GIT_USER_EMAIL"
fi

git config --global init.defaultBranch "${GIT_DEFAULT_BRANCH:-main}" || true
