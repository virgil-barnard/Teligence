# AgentLab (scaffold)

A lightweight, repeatable way to run “coding agent workstations” per project using Docker Compose.

**Design goals**
- **No secrets in images**: tokens + keys are mounted at runtime as **Compose secrets** (files under `/run/secrets/*`).  
- **Project shareability**: project folder contains shareable config (base image, repo URL, extra deps, entrypoint scripts).
- **Persistent interaction**: project workspace is mounted to `/workspace`.
- **Composable runtime**: optional services (like **Ollama** and **LiteLLM**) are enabled via Compose **profiles**.

---

## Prereqs
- Docker + Docker Compose v2 (`docker compose ...`)

---

## Quick start

### 1) Edit your user configs (secrets live here)
Open these files and fill in your real values:

**Git / hosting**
- `configs/git/git.env`
- `configs/github/github.env`
- `configs/glab/glab.env` (optional)

**LiteLLM gateway (recommended)**
- `configs/litellm/litellm.env` (proxy master key + salt key)
- `configs/providers/providers.env` (upstream provider keys used by LiteLLM)
- `configs/litellm/litellm-config.yaml` (model routing; non-secret)

**Agent-side LLM client settings**
- `configs/llm/llm.env` (your tools inside the agent container)

> By default, the scaffold is set up so your *agent tools* talk to the LiteLLM proxy at `http://litellm:4000`.

### 2) Create a project
```bash
make init PROJ=myproj
```

Edit `projects/myproj/project.env` and set (at minimum):
- `BASE_IMAGE`
- `REPO_URL` (optional, will auto-clone into `/workspace/repo`)

### 3) Build and start the agent container
```bash
make build PROJ=myproj
make up PROJ=myproj
```

Get a shell (loads secrets for the session):
```bash
make shell PROJ=myproj
```

Run a command inside the running container (loads secrets for the command):
```bash
make task PROJ=myproj TASK='cd /workspace/repo && git status'
```

Run a one-off command (container is created and removed):
```bash
make run PROJ=myproj TASK='echo hello && ls -la /workspace'
```

Stop and remove:
```bash
make down PROJ=myproj
```

---

## LiteLLM gateway (optional profile: `gateway`)

This scaffold includes a `litellm` service behind a Compose profile named `gateway`.

Start your project **with** LiteLLM:
```bash
make up PROJ=myproj PROFILES=gateway
```

LiteLLM is exposed on your host at:
- `http://127.0.0.1:4000`

Inside the agent container, reach it at:
- `http://litellm:4000`

The config file is mounted at:
- `configs/litellm/litellm-config.yaml`

---

## Ollama (optional profile: `ollama`)

Start your project **with** Ollama:
```bash
make up PROJ=myproj PROFILES=ollama
```

Start **both** LiteLLM + Ollama:
```bash
make up PROJ=myproj PROFILES="gateway ollama"
```

Inside the agent container, you can reach Ollama at:
- `http://ollama:11434`

---

## Project-specific overrides

If your project needs extra mounts/ports/devices, edit:
- `projects/<proj>/compose.yaml`

This file is automatically included by the Make targets when present.

---

## Repository cloning

If `REPO_URL` is set in `projects/<proj>/project.env`, the container bootstraps by cloning it into:
- `/workspace/repo`

Tokens are provided via mounted secrets, and a `.netrc` file is written in the container’s ephemeral `$HOME` (`/tmp/home`).

---

## Notes on security defaults

The agent service uses:
- `read_only: true` root filesystem
- `tmpfs` for `/tmp` and `/var/tmp`
- `cap_drop: [ALL]`
- `no-new-privileges`

If a particular project needs more privileges (debugging, ptrace, Docker-in-Docker, etc.), relax **only in that project’s** `projects/<proj>/compose.yaml`.

---

## Layout

```text
compose.yaml
Makefile
configs/
  git/ git.env
  github/ github.env
  glab/ glab.env
  llm/ llm.env
  litellm/ litellm-config.yaml + litellm.env (secret)
  providers/ providers.env (secret)
  opencode/ opencode.jsonc
gateways/
  litellm/ entrypoint.sh
templates/
  project/ (copied into new projects)
projects/
  <your projects live here>
```
