#!/usr/bin/env bash
set -euo pipefail

# If an OpenCode config file is mounted, point to it.
if [[ -f "${CONFIG_DIR:-/run/configs}/opencode/opencode.jsonc" ]]; then
  export OPENCODE_CONFIG="${CONFIG_DIR:-/run/configs}/opencode/opencode.jsonc"
fi
