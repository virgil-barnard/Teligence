SHELL := bash
PROJ ?=
PROFILES ?=
PROJECTS_DIR := projects

PROJ_DIR := $(PROJECTS_DIR)/$(PROJ)
PROJ_ENV := $(PROJ_DIR)/project.env

LOCAL_UID := $(shell id -u)
LOCAL_GID := $(shell id -g)

COMPOSE_BASE := docker compose -f compose.yaml
COMPOSE_PROJ := $(COMPOSE_BASE) --env-file $(PROJ_ENV) -p agent-$(PROJ)

# Auto-include per-project compose override if present
ifneq ("$(wildcard $(PROJ_DIR)/compose.yaml)","")
  COMPOSE_PROJ := $(COMPOSE_PROJ) -f $(PROJ_DIR)/compose.yaml
endif

# Enable Compose profiles if specified
ifneq ("$(strip $(PROFILES))","")
  COMPOSE_PROJ := $(COMPOSE_PROJ) --profile $(PROFILES)
endif

define REQUIRE_PROJ
	@if [[ -z "$(PROJ)" ]]; then 		echo "ERROR: set PROJ=<name> (e.g. make init PROJ=myproj)"; 		exit 2; 	fi
endef

.PHONY: help
help:
	@cat <<'EOF'
Targets:
  make init   PROJ=name           Create projects/name from templates
  make build  PROJ=name           Build image
  make up     PROJ=name [PROFILES=ollama]   Start agent (and optional profile services)
  make down   PROJ=name           Stop and remove services for that project
  make ps     PROJ=name           Show status
  make logs   PROJ=name           Tail logs
  make shell  PROJ=name           Shell into running agent (loads secrets)
  make task   PROJ=name TASK=...  Run a command in running agent (loads secrets)
  make run    PROJ=name TASK=...  Run one-off container (loads secrets)

Examples:
  make init PROJ=myproj
  make build PROJ=myproj
  make up PROJ=myproj
  make shell PROJ=myproj
  make task PROJ=myproj TASK='cd /workspace/repo && pytest -q'
  make up PROJ=myproj PROFILES=ollama
EOF

.PHONY: init
init:
	$(call REQUIRE_PROJ)
	@mkdir -p "$(PROJ_DIR)" "$(PROJ_DIR)/workspace"
	@cp -an "templates/project/." "$(PROJ_DIR)/"
	@if [[ ! -f "$(PROJ_ENV)" && -f "$(PROJ_DIR)/project.env.example" ]]; then 		cp "$(PROJ_DIR)/project.env.example" "$(PROJ_ENV)"; 	fi
	@# Ensure PROJ inside project.env matches the folder name
	@if [[ -f "$(PROJ_ENV)" ]]; then 		if grep -qE '^PROJ=' "$(PROJ_ENV)"; then 			sed -i 's/^PROJ=.*/PROJ=$(PROJ)/' "$(PROJ_ENV)"; 		else 			echo "PROJ=$(PROJ)" >> "$(PROJ_ENV)"; 		fi; 	fi
	@echo "Initialized: $(PROJ_DIR)"
	@echo "Edit:        $(PROJ_ENV)"
	@echo "Workspace:   $(PROJ_DIR)/workspace"

.PHONY: build
build:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) $(COMPOSE_PROJ) build

.PHONY: up
up:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) $(COMPOSE_PROJ) up -d

.PHONY: down
down:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) $(COMPOSE_PROJ) down

.PHONY: ps
ps:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) $(COMPOSE_PROJ) ps

.PHONY: logs
logs:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) $(COMPOSE_PROJ) logs -f --tail=200

# Load secret env files for an interactive shell or task.
# Compose secrets are files at /run/secrets/<name>.
define LOAD_SECRETS
set -e; for f in /run/secrets/git_env /run/secrets/github_env /run/secrets/glab_env /run/secrets/llm_env; do   [[ -f $$f ]] && source $$f || true; done; # Convenience: make gh work even if token is named GITHUB_TOKEN.
[[ -n "$${GITHUB_TOKEN:-}" && -z "$${GH_TOKEN:-}" ]] && export GH_TOKEN="$${GITHUB_TOKEN}" || true; true
endef

.PHONY: shell
shell:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) $(COMPOSE_PROJ) exec -it agent bash -lc '$(LOAD_SECRETS); exec bash'

.PHONY: task
task:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) $(COMPOSE_PROJ) exec -it agent bash -lc '$(LOAD_SECRETS); $(TASK)'

.PHONY: run
run:
	$(call REQUIRE_PROJ)
	@# One-off container run; doesn't require 'make up' first.
	@PROJ=$(PROJ) LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) $(COMPOSE_PROJ) run --rm -it agent bash -lc '$(LOAD_SECRETS); $(TASK)'

.PHONY: opencode
opencode:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) $(COMPOSE_PROJ) exec -it agent bash -lc '$(LOAD_SECRETS); opencode'
