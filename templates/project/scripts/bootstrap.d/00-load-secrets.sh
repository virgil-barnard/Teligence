#!/usr/bin/env bash
set -euo pipefail

load_env_file() {
  local f="$1"
  [[ -f "$f" ]] || return 0
  set -a
  # shellcheck disable=SC1090
  source "$f"
  set +a
}

load_env_file /run/secrets/git_env
load_env_file /run/secrets/github_env
load_env_file /run/secrets/glab_env
load_env_file /run/secrets/llm_env

# Convenience for gh: GH_TOKEN is recognized; map from GITHUB_TOKEN if present.
if [[ -n "${GITHUB_TOKEN:-}" && -z "${GH_TOKEN:-}" ]]; then
  export GH_TOKEN="${GITHUB_TOKEN}"
fi
