#!/usr/bin/env bash
set -euo pipefail

netrc="$HOME/.netrc"
mkdir -p "$(dirname "$netrc")"
touch "$netrc"
chmod 600 "$netrc"

# GitHub HTTPS git auth
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  host="${GITHUB_HOST:-github.com}"
  cat >>"$netrc" <<EOF
machine ${host}
  login x-access-token
  password ${GITHUB_TOKEN}
EOF
fi

# GitLab HTTPS git auth
if [[ -n "${GITLAB_TOKEN:-}" ]]; then
  host="${GITLAB_HOST:-gitlab.com}"
  cat >>"$netrc" <<EOF
machine ${host}
  login oauth2
  password ${GITLAB_TOKEN}
EOF
fi
