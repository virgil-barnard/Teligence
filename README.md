# AgentLab (OpenAI-only quickstart)

This scaffold gives you **project-scoped, containerized dev/agent environments** with:

- per-project images + `/workspace` persistence
- GitHub-first tooling (`gh`) + GitLab (`glab`)
- secrets not baked into images (Compose secrets mount to `/run/secrets/*`)
- optional Ollama sidecar (profile-based)

## Prereqs (Windows + WSL2 + Docker Desktop)

1) Enable Docker Desktop WSL2 integration  
Docker Desktop → Settings → Resources → WSL Integration → enable your distro.

2) Open your WSL distro and confirm Docker works:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

Tip: put this repo in your **WSL filesystem** (e.g. `/home/<you>/agentlab`) rather than `/mnt/c/...` for better performance.

## 1) Fill in your configs (gitignored)

Edit these files:

- `configs/llm/llm.env`  
  Put your OpenAI key as `OPENAI_API_KEY=...`
- `configs/github/github.env`  
  Put your GitHub PAT as `GITHUB_TOKEN=...`
- Optional:
  - `configs/glab/glab.env` for GitLab PAT
  - `configs/git/git.env` for git name/email

Compose secrets appear at `/run/secrets/<name>` inside the container.

## 2) Create a new project

```bash
make init PROJ=myproj
```

Then edit `projects/myproj/project.env`:
- `REPO_URL=https://github.com/<owner>/<repo>.git` (optional; auto-clones to `/workspace/repo`)
- `BASE_IMAGE=ubuntu:24.04` (or your preferred base)

## 3) Build + start

```bash
make build PROJ=myproj
make up PROJ=myproj
make shell PROJ=myproj
```

Run commands inside the running container:

```bash
make task PROJ=myproj TASK='cd /workspace/repo && git status'
```

## OpenCode usage

OpenCode can read provider keys from the environment (or a `.env` in your project).

```bash
make opencode PROJ=myproj
```

Config override is supported via `OPENCODE_CONFIG` / `OPENCODE_CONFIG_DIR`.

## Optional: start Ollama

```bash
make up PROJ=myproj PROFILES=ollama
```

From inside the agent container, Ollama is reachable at `http://ollama:11434`.

## Troubleshooting

### “permission denied” on scripts
If you edited scripts in Windows and they became CRLF, re-checkout with LF (this repo ships `.gitattributes` to force LF).

### Docker not found inside WSL
Re-check Docker Desktop WSL integration settings.
