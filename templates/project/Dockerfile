# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]
ARG DEBIAN_FRONTEND=noninteractive

# Pin versions where helpful
ARG GLAB_VERSION=1.46.0

# Core tooling
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git bash jq openssh-client tar unzip tini \
    gnupg \
  && rm -rf /var/lib/apt/lists/*

# ---- GitHub CLI (gh) via official apt repo (Debian/Ubuntu)
RUN set -euo pipefail; \
  if command -v apt-get >/dev/null 2>&1; then \
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg; \
    chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg; \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      > /etc/apt/sources.list.d/github-cli.list; \
    apt-get update; \
    apt-get install -y --no-install-recommends gh; \
    rm -rf /var/lib/apt/lists/*; \
  fi

# ---- Install glab from GitLab release tarball
RUN set -euo pipefail; \
  arch="x86_64"; \
  case "$(uname -m)" in \
    x86_64|amd64) arch="x86_64" ;; \
    aarch64|arm64) arch="arm64" ;; \
  esac; \
  url="https://gitlab.com/gitlab-org/cli/-/releases/v${GLAB_VERSION}/downloads/glab_${GLAB_VERSION}_Linux_${arch}.tar.gz"; \
  curl -fsSL "$url" -o /tmp/glab.tgz; \
  tar -xzf /tmp/glab.tgz -C /tmp; \
  glab_bin="$(find /tmp -maxdepth 3 -type f -name glab | head -n 1)"; \
  install -m 0755 "$glab_bin" /usr/local/bin/glab; \
  rm -rf /tmp/glab.tgz /tmp/glab_* /tmp/glab-*

# ---- Install OpenCode (binary install script)
RUN set -euo pipefail; \
  curl -fsSL https://opencode.ai/install | bash -s -- --no-modify-path; \
  install -m 0755 /root/.opencode/bin/opencode /usr/local/bin/opencode; \
  rm -rf /root/.opencode

# ---- Optional: extra OS packages
COPY apt.packages /tmp/apt.packages
RUN set -euo pipefail; \
  if [[ -s /tmp/apt.packages ]]; then \
    apt-get update; \
    grep -vE '^\s*#' /tmp/apt.packages | sed '/^\s*$/d' | xargs -r apt-get install -y --no-install-recommends; \
    rm -rf /var/lib/apt/lists/*; \
  fi

# ---- Optional: extra pip requirements (only if pip exists)
COPY requirements.extra.txt /tmp/requirements.extra.txt
RUN set -euo pipefail; \
  if command -v pip3 >/dev/null 2>&1 && [[ -s /tmp/requirements.extra.txt ]]; then \
    pip3 install --no-cache-dir -r /tmp/requirements.extra.txt; \
  fi

# Runtime bootstrap scripts
COPY scripts/ /usr/local/lib/agent/
RUN chmod +x /usr/local/lib/agent/entrypoint.sh /usr/local/lib/agent/bootstrap.d/*.sh

ENV CONFIG_DIR=/run/configs
ENV WORKSPACE=/workspace

ENTRYPOINT ["/usr/bin/tini","--","/usr/local/lib/agent/entrypoint.sh"]
CMD ["bash"]
