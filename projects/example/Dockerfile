# syntax=docker/dockerfile:1
ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE}

SHELL ["/bin/bash", "-lc"]
ARG DEBIAN_FRONTEND=noninteractive

# Pin versions (override via build args)
ARG GH_VERSION=2.57.0
ARG GLAB_VERSION=1.46.0
ARG OPENCODE_VERSION=

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl git bash jq openssh-client tar unzip tini \
  && rm -rf /var/lib/apt/lists/*

# ---- Install GitHub CLI (gh) from GitHub releases (single binary)
RUN set -euo pipefail; \
  arch="amd64"; \
  case "$(uname -m)" in \
    x86_64|amd64) arch="amd64" ;; \
    aarch64|arm64) arch="arm64" ;; \
  esac; \
  url="https://github.com/cli/cli/releases/download/v${GH_VERSION}/gh_${GH_VERSION}_linux_${arch}.tar.gz"; \
  curl -fsSL "$url" -o /tmp/gh.tgz; \
  tar -xzf /tmp/gh.tgz -C /tmp; \
  install -m 0755 "/tmp/gh_${GH_VERSION}_linux_${arch}/bin/gh" /usr/local/bin/gh; \
  rm -rf /tmp/gh.tgz "/tmp/gh_${GH_VERSION}_linux_${arch}"

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

# ---- Install OpenCode via official install script
RUN set -euo pipefail; \
  if [[ -n "${OPENCODE_VERSION}" ]]; then \
    curl -fsSL https://opencode.ai/install | bash -s -- --no-modify-path --version "${OPENCODE_VERSION}"; \
  else \
    curl -fsSL https://opencode.ai/install | bash -s -- --no-modify-path; \
  fi; \
  install -m 0755 /root/.opencode/bin/opencode /usr/local/bin/opencode; \
  rm -rf /root/.opencode

# ---- Optional: extra OS packages
COPY apt.packages /tmp/apt.packages
RUN set -euo pipefail; \
  if [[ -s /tmp/apt.packages ]]; then \
    apt-get update; \
    pkgs="$(grep -vE '^\s*#|^\s*$' /tmp/apt.packages || true)"; \
    if [[ -n "$pkgs" ]]; then \
      echo "$pkgs" | xargs -r apt-get install -y --no-install-recommends; \
    fi; \
    rm -rf /var/lib/apt/lists/*; \
  fi

# ---- Optional: extra pip requirements (only if pip exists)
COPY requirements.extra.txt /tmp/requirements.extra.txt
RUN set -euo pipefail; \
  if command -v pip3 >/dev/null 2>&1 && [[ -s /tmp/requirements.extra.txt ]]; then \
    reqs="$(grep -vE '^\s*#|^\s*$' /tmp/requirements.extra.txt || true)"; \
    if [[ -n "$reqs" ]]; then \
      echo "$reqs" > /tmp/reqs.clean; \
      pip3 install --no-cache-dir -r /tmp/reqs.clean; \
    fi; \
  fi

# Runtime bootstrap
COPY scripts/ /usr/local/lib/agent/
RUN chmod +x /usr/local/lib/agent/entrypoint.sh /usr/local/lib/agent/bootstrap.d/*.sh

ENV CONFIG_DIR=/run/configs
ENV WORKSPACE=/workspace
ENV HOME=/tmp/home

ENTRYPOINT ["/usr/bin/tini","--","/usr/local/lib/agent/entrypoint.sh"]
CMD ["bash"]
