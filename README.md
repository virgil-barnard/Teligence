## Teligence Mini-GPT Lab

This repository contains a shared TensorFlow GPT backbone plus multiple training scripts for different action/language spaces:

- text language modeling (`scripts/gpt_text.py`, with `gpt.py` compatibility wrapper)
- symbolic algebra rewrite proofs (`experiments/proof_rewrite_gpt.py`)
- finite affine/projective geometry theorem control (`experiments/icarus_projective_actionptr_v2.py`)

All variants now default to outputs under `runs/`.

## Repository layout

- `scripts/gpt_text.py` - text-domain training entrypoint
- `gpt.py` - compatibility wrapper for legacy commands
- `experiments/proof_rewrite_gpt.py` - proof-rewrite environment + GPT policy
- `experiments/icarus_projective_actionptr_v2.py` - affine/projective verifier + action-pointer GPT
- `scripts/launcher.py` - unified script launcher via `APP_ENTRY`
- `scripts/benchmark_matrix.py`, `scripts/sweep.py` - experiment orchestration scripts
- `teligence/config.py`, `teligence/modeling.py`, `teligence/train_utils.py` - shared GPT architecture/training utilities
- `docker-compose.yml`, `Dockerfile` - Docker-first workflow
- `tests/test_smoke.py` - smoke tests

## Quick start

Build and run default text training:

```bash
docker compose up --build
```

TensorBoard is available at `http://localhost:6006`.

If image already exists:

```bash
docker compose up
```

## Unified startup options

The launcher is the single entry path:

```bash
docker compose run --rm gpt
```

Switch variants with `APP_ENTRY`:

```bash
docker compose run --rm -e APP_ENTRY=gpt_text gpt
docker compose run --rm -e APP_ENTRY=proof_math gpt
docker compose run --rm -e APP_ENTRY=icarus_affine gpt
```

Pass variant-specific flags using `APP_ARGS`:

```bash
docker compose run --rm \
  -e APP_ENTRY=proof_math \
  -e APP_ARGS="--mode train --train_steps 4000" \
  gpt
```

Profile services are also available:

```bash
docker compose --profile proof run --rm proof_math
docker compose --profile icarus run --rm icarus_affine
```

Legacy launcher aliases remain supported:

- `proof_agent` -> `proof_math`
- `icarus_projective_v2` -> `icarus_affine`

## Output directories (consistent defaults)

- `gpt_text` (`scripts/gpt_text.py`): `runs/<run_name>/...`
- `proof_math` (`experiments/proof_rewrite_gpt.py`): `runs/proof_rewrite_gpt/ckpt_last` (or `runs/<run_name>/...` via args)
- `icarus_affine` (`experiments/icarus_projective_actionptr_v2.py`): `runs/icarus_projective_v2/ckpt_last` and `ckpt_best`

Use these knobs to override paths:

- text: `RUN_NAME`, `RUNS_DIR`
- proof: `--run_name`, `--runs_dir`, `--ckpt`
- icarus: `--out_dir`, `--resume_checkpoint`

## Architecture and action-space differences

### 1) `gpt_text` (`scripts/gpt_text.py`)

- **Backbone**: shared `ExplicitGPT` (TF, GQA, RoPE/ALiBi options, flash-attn path)
- **Input space**: raw token streams from text datasets (`enwik8`, `tinyshakespeare`, `names`)
- **Output space**: next-token language modeling over tokenizer vocabulary
- **Objective**: standard autoregressive LM loss

### 2) `proof_math` (`experiments/proof_rewrite_gpt.py`)

- **Backbone**: shared `ExplicitGPT`
- **Input space**: serialized symbolic equation states + control tokens (`GOAL`, `STATE`, `ACT`)
- **Action/language space**: rewrite action tokens (`A_<rule>_<node_index>`)
- **Objective**: masked supervised action prediction + environment rollout solve-rate eval
- **Notes**: curriculum support and traceable eval episodes

### 3) `icarus_affine` (`experiments/icarus_projective_actionptr_v2.py`)

- **Backbone**: shared `ExplicitGPT` as state encoder
- **Input space**: serialized finite-geometry proof state (objects, facts, goal, history)
- **Action/language space**: legal-action list over typed theorem actions (`CONSTRUCT_*`, `ASSERT_*`, `STOP`)
- **Policy head**: compositional action-pointer (op + args embeddings -> action key)
- **Extra head**: value prediction (`steps_to_goal`) for search guidance
- **Inference**: greedy or best-first search over verifier-legal actions

## Common commands

Run smoke tests:

```bash
docker compose run --rm gpt python -m unittest tests.test_smoke
```

Run single smoke test method:

```bash
docker compose run --rm gpt python -m unittest tests.test_smoke.SmokeTests.test_model_forward_shape
```

Run benchmark matrix (text profiles):

```bash
python scripts/benchmark_matrix.py
```

## Resume training examples

Text model:

```bash
docker compose run --rm \
  -e APP_ENTRY=gpt_text \
  -e RUN_NAME=my_run \
  -e RESUME_FROM=my_run \
  -e EXTRA_UPDATES=5000 \
  gpt
```

Proof model:

```bash
docker compose run --rm \
  -e APP_ENTRY=proof_math \
  -e APP_ARGS="--run_name proof_rewrite_gpt --train_steps 12000" \
  gpt
```

Icarus model:

```bash
docker compose run --rm \
  -e APP_ENTRY=icarus_affine \
  -e APP_ARGS="--mode train --out_dir runs/icarus_projective_v2 --resume_checkpoint runs/icarus_projective_v2/ckpt_last/ckpt-200 --max_iters 8000" \
  gpt
```
