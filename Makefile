SHELL := bash
PROJ ?=
PROFILES ?=

PROJECTS_DIR := projects
TEMPLATE_DIR := templates/project

PROJ_DIR := $(PROJECTS_DIR)/$(PROJ)
PROJ_ENV := $(PROJ_DIR)/project.env

LOCAL_UID := $(shell id -u)
LOCAL_GID := $(shell id -g)

# Build compose command with optional profiles and per-project override file.
COMPOSE_BASE := docker compose -f compose.yaml
COMPOSE_PROFILE_ARGS := $(foreach p,$(PROFILES),--profile $(p))

COMPOSE_PROJ := $(COMPOSE_BASE) $(COMPOSE_PROFILE_ARGS) --env-file $(PROJ_ENV) -p agent-$(PROJ)

ifneq ("$(wildcard $(PROJ_DIR)/compose.yaml)","")
  COMPOSE_PROJ := $(COMPOSE_PROJ) -f $(PROJ_DIR)/compose.yaml
endif

# Source secrets for commands executed via `docker compose exec`.
# Note: Docker exec processes don't inherit runtime-exported env vars; sourcing keeps it reliable.
SECRETS_SOURCE := source /usr/local/lib/agent/bootstrap.d/00-load-secrets.sh

define REQUIRE_PROJ
	@if [[ -z "$(PROJ)" ]]; then \
		echo "ERROR: set PROJ=<project> (e.g. make init PROJ=myproj)"; \
		exit 2; \
	fi
endef

.PHONY: help
help:
	@cat <<'EOF'
AgentLab Make targets (Compose-first)

Core:
  make init   PROJ=name                        Create projects/name from templates
  make build  PROJ=name                        Build the image
  make up     PROJ=name [PROFILES="ollama gateway"]   Start services (agent + enabled profiles)
  make down   PROJ=name                        Stop and remove containers
  make shell  PROJ=name                        Shell into running agent (loads secrets)
  make task   PROJ=name TASK='..'              Run command inside running agent (loads secrets)
  make run    PROJ=name TASK='..'              One-off command in a fresh agent container (loads secrets)

Ops:
  make ps     PROJ=name
  make logs   PROJ=name
  make nuke   PROJ=name                        down + remove local images + volumes

Examples:
  make init  PROJ=myproj
  make build PROJ=myproj
  make up    PROJ=myproj
  make up    PROJ=myproj PROFILES=ollama
  make up    PROJ=myproj PROFILES="gateway ollama"
  make shell PROJ=myproj
  make task  PROJ=myproj TASK='cd /workspace/repo && git status'
EOF

.PHONY: ls
ls:
	@ls -1 "$(PROJECTS_DIR)" 2>/dev/null || true

.PHONY: init
init:
	$(call REQUIRE_PROJ)
	@mkdir -p "$(PROJ_DIR)" "$(PROJ_DIR)/workspace"
	@cp -an "$(TEMPLATE_DIR)/." "$(PROJ_DIR)/"
	@if [[ -f "$(PROJ_DIR)/project.env.example" && ! -f "$(PROJ_ENV)" ]]; then \
		cp "$(PROJ_DIR)/project.env.example" "$(PROJ_ENV)"; \
	fi
	@echo "Initialized project: $(PROJ_DIR)"
	@echo "Edit: $(PROJ_ENV)"
	@echo "Optional overrides: $(PROJ_DIR)/compose.yaml"

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

.PHONY: shell
shell:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) $(COMPOSE_PROJ) exec -it agent bash -lc "$(SECRETS_SOURCE); exec bash -i"

.PHONY: task
task:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) $(COMPOSE_PROJ) exec -it agent bash -lc "$(SECRETS_SOURCE); $(TASK)"

.PHONY: run
run:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) $(COMPOSE_PROJ) run --rm agent bash -lc "$(SECRETS_SOURCE); $(TASK)"

.PHONY: nuke
nuke:
	$(call REQUIRE_PROJ)
	@PROJ=$(PROJ) LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID) $(COMPOSE_PROJ) down --rmi local --volumes --remove-orphans || true
