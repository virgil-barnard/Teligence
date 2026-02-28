# configs/

This folder is for **user- and machine-specific** configuration.

Non-secret config can live here (e.g., OpenCode settings, LiteLLM model routing),
while secret `.env` files are **gitignored by default** (see repo `.gitignore`).

## Git + hosting
- `git/git.env` — your git identity
- `github/github.env` — GitHub token and optional settings (used for `.netrc` and `gh`)
- `glab/glab.env` — GitLab host + token (optional; used for `.netrc` and `glab`)

## Agent-side LLM settings
- `llm/llm.env` — what your tools inside the **agent** container should use
  - recommended: point OpenAI-compatible tools at LiteLLM (`OPENAI_BASE_URL=http://litellm:4000`)
  - store the **proxy key** as `OPENAI_API_KEY` (master key or virtual key)

## LiteLLM gateway
- `litellm/litellm-config.yaml` — **non-secret** LiteLLM proxy routing config (models + providers)
- `litellm/litellm.env` — **secret** LiteLLM proxy settings (`LITELLM_MASTER_KEY`, `LITELLM_SALT_KEY`, etc.)
- `providers/providers.env` — **secret** upstream provider keys (e.g. OpenAI / Anthropic keys) used by LiteLLM

## OpenCode
- `opencode/` — non-secret OpenCode settings (config file, prompts, etc.)
