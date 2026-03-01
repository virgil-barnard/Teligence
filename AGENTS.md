# AGENTS.md

Guidance for coding agents working in `C:\Users\athan\Code\Prototyping\Teligence`.

This repository is a Python + TensorFlow GPT experimentation project, with Docker-first workflows and `unittest` smoke tests.

## 1) Repository Facts

- Main language: Python 3
- ML stack: TensorFlow 2.16.x + NumPy (`numpy<2`)
- Entrypoint: `gpt.py`
- Test framework: Python `unittest` (`tests/test_smoke.py`)
- Container workflow is primary (`Dockerfile`, `docker-compose.yml`)
- CI smoke workflow: `.github/workflows/smoke.yml`

## 2) Build / Run / Test Commands

Use these commands from repo root.

### Docker build and run

- Build image used by training and CI:
  - `docker build -t teligence-smoke -f Dockerfile .`
- Start training + TensorBoard with compose:
  - `docker compose up --build`
- Re-run without rebuild:
  - `docker compose up`
- Run one-off training container:
  - `docker compose run --rm gpt`
- Run one-off script in training image:
  - `docker compose run --rm gpt python benchmark_matrix.py`

### Local (non-Docker) run

There is no pinned local requirements file; Docker is canonical.
If running locally anyway, mirror Docker dependency assumptions.

- Minimal local install baseline:
  - `python -m pip install --upgrade pip`
  - `python -m pip install "numpy<2" tensorflow`
- Run training locally:
  - `python gpt.py`
- Run benchmark matrix locally:
  - `python benchmark_matrix.py`

### Tests

- Run all smoke tests locally:
  - `python -m unittest tests.test_smoke`
- Run all smoke tests in compose container:
  - `docker compose run --rm gpt python -m unittest tests.test_smoke`
- Run all smoke tests in direct docker image (CI-like):
  - `docker run --rm -v "${PWD}:/app" -w /app teligence-smoke python -m unittest tests.test_smoke`

### Run a single test (important)

- Single test class:
  - `python -m unittest tests.test_smoke.SmokeTests`
- Single test method:
  - `python -m unittest tests.test_smoke.SmokeTests.test_model_forward_shape`
- Single test method in container:
  - `docker compose run --rm gpt python -m unittest tests.test_smoke.SmokeTests.test_model_forward_shape`

### Lint / format / type-check

No dedicated linter/formatter/type-checker config is currently present
(`ruff`, `black`, `isort`, `mypy`, `pyright`, `flake8` not configured).

Agent expectation:

- Keep style consistent with existing source (see style section below).
- Validate changes by running smoke tests when behavior changes.
- Avoid introducing new tooling/config unless explicitly requested.

## 3) Project Structure and Ownership Hints

- `gpt.py`: training entrypoint and orchestration
- `config.py`: config/env parsing + validation
- `modeling.py`: model layers, attention, precision controls
- `train_utils.py`, `data_utils.py`, `tokenizer.py`: training/data/tokenization helpers
- `run_utils.py`, `runtime.py`: run metadata + runtime setup
- `benchmark_matrix.py`, `sweep.py`: experiment runners
- `tests/test_smoke.py`: fast safety tests

## 4) Code Style Guidelines (Observed Conventions)

Follow these conventions unless user asks otherwise.

### Imports

- Order imports as:
  1) standard library
  2) third-party
  3) local project modules
- Separate each group with one blank line.
- Prefer explicit imports over wildcard imports.

### Formatting

- 4-space indentation; no tabs.
- Keep functions focused and imperative.
- Prefer readable line breaks over dense one-liners.
- Use f-strings for user-facing logs/prints.
- Keep comments sparse; only for non-obvious logic.

### Types and data modeling

- Use type hints on public functions where practical.
- Use `@dataclass` for structured configuration/state objects.
- Follow existing hint style (`list[int]`, typed fields).
- Do not add strict typing frameworks unless requested.

### Naming

- `snake_case` for functions/variables/modules.
- `PascalCase` for classes.
- `UPPER_CASE` for module-level constants.
- Config/env names are uppercase (e.g., `NUM_UPDATES`).
- Test methods use `test_*` naming.

### Error handling and validation

- Use `ValueError` for invalid user/config inputs.
- Use assertions for strict internal invariants already assumed by code paths.
- Preserve exception context (`raise ... from e`) when wrapping failures.
- Fail early when required files/data are missing or malformed.

### Logging and output

- Use plain `print` for runtime logs (current project pattern).
- Keep logs compact and metric-focused.
- Escape generated preview text when needed (`escape_preview_text`).

### TensorFlow and numeric safety

- Preserve explicit casts around logits/loss math (`tf.float32` where needed).
- Maintain dtype discipline across compute/cache precision helpers.
- Guard against non-finite gradients where existing code does so.
- Keep shape assumptions explicit; update validation if changing dimensions.

### Configuration and environment variables

- Prefer adding tunables in `GPTConfig` via env defaults.
- Keep env var names uppercase and documented in README when user-visible.
- Validate new config interactions in `validate_config`.

### Tests

- Add/extend smoke tests for behavior-level changes.
- Keep tests deterministic and lightweight.
- Avoid dataset/network-heavy tests in smoke suite.
- Use `runs_test/` or similarly disposable paths for test artifacts.

## 5) Cursor / Copilot Rule Files

Checked locations requested by user:

- `.cursor/rules/` -> not present
- `.cursorrules` -> not present
- `.github/copilot-instructions.md` -> not present

No additional Cursor/Copilot repository instruction files are currently available to incorporate.

## 6) Quick Pre-PR Verification Checklist

- Run: `python -m unittest tests.test_smoke`
- If Docker-related change, also run: `docker build -t teligence-smoke -f Dockerfile .`
- Confirm no artifacts are staged from `runs/`, `data/`, or checkpoints
