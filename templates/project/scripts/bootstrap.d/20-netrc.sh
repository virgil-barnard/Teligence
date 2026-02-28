#!/usr/bin/env bash
set -euo pipefail

netrc="$HOME/.netrc"
touch "$netrc"
chmod 600 "$netrc"

# GitHub
if [[ -n "${GITHUB_TOKEN:-}" && "${GITHUB_TOKEN}" != "ghp_REPLACE_ME" ]]; then
  user="${GITHUB_USER:-x-access-token}"
  cat >>"$netrc" <<EOF
machine github.com
  login ${user}
  password ${GITHUB_TOKEN}
EOF
fi

# GitLab
if [[ -n "${GITLAB_TOKEN:-}" && "${GITLAB_TOKEN}" != "glpat_REPLACE_ME" ]]; then
  host="${GITLAB_HOST:-gitlab.com}"
  cat >>"$netrc" <<EOF
machine ${host}
  login oauth2
  password ${GITLAB_TOKEN}
EOF
fi
