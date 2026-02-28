#!/usr/bin/env bash
set -euo pipefail
command -v opencode >/dev/null 2>&1 || exit 0

# OpenCode config is mounted at /run/configs/opencode by compose.yaml
if [[ -f "${OPENCODE_CONFIG:-}" ]]; then
  :
fi

# Provider keys are expected from /run/secrets/llm_env (loaded by 00-load-secrets.sh)
