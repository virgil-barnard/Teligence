## Teligence Mini-GPT Lab

This repository contains a shared TensorFlow GPT backbone plus multiple training scripts for different action/language spaces:

- text language modeling (`scripts/gpt_text.py`)
- symbolic algebra factoring/rewrite proofs (`experiments/proof_factoring_gpt.py`)
- finite affine/projective proof control (`experiments/proof_affine_gpt.py`)

All variants now default to outputs under `runs/`.

## Research framing and inspirations

This project follows the spirit of Karpathy's autonomous iteration workflows:

- `autoresearch`: https://github.com/karpathy/autoresearch
- `nanoGPT`: https://github.com/karpathy/nanoGPT

Like those projects, we optimize the core GPT architecture and training loop through fast, repeatable runs. The main difference here is that we evaluate improvements across multiple domains instead of language modeling only:

- text next-token prediction (`gpt_text`)
- symbolic algebra rewrite control (`proof_factoring`)
- finite affine/projective theorem control (`proof_affine`)

This multi-domain setup is intended to reward architecture and optimization changes that generalize beyond one task family.

## Autoresearch-style workflow in this repo

Research automation scaffolding lives under `research/` and `scripts/autoresearch_loop.py`.

Initialize run ledger:

```bash
python scripts/autoresearch_loop.py init
```

Record one run:

```bash
python scripts/autoresearch_loop.py run --track gpt_text --run-name auto_text --description "baseline"
```

View best/recent results:

```bash
python scripts/autoresearch_loop.py leaderboard
```

## Repository layout

- `scripts/gpt_text.py` - text-domain training entrypoint
- `experiments/proof_factoring_gpt.py` - proof-factoring environment + GPT policy
- `experiments/proof_factoring_domain.py` - expression factoring/task/data-generation domain layer
- `experiments/proof_affine_gpt.py` - affine/projective verifier + action-pointer GPT
- `experiments/proof_affine_domain.py` - affine/projective task/data-generation domain layer
- `scripts/launcher.py` - unified script launcher via `APP_ENTRY`
- `scripts/benchmark_matrix.py`, `scripts/sweep.py` - experiment orchestration scripts
- `teligence/config.py`, `teligence/modeling.py`, `teligence/train_utils.py` - shared GPT architecture/training utilities
- `teligence/action_heads.py` - task-specific heads built on top of `ExplicitGPT`
- `teligence/experiment_utils.py` - shared experiment scheduling/optimizer helpers
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
docker compose run --rm -e APP_ENTRY=proof_factoring gpt
docker compose run --rm -e APP_ENTRY=proof_affine gpt
```

Pass variant-specific flags using `APP_ARGS`:

```bash
docker compose run --rm \
  -e APP_ENTRY=proof_factoring \
  -e APP_ARGS="--mode train --train_steps 4000" \
  gpt
```

Profile services are also available:

```bash
docker compose --profile proof run --rm proof_factoring_gpt
docker compose --profile icarus run --rm proof_affine_gpt
```

Legacy launcher aliases remain supported:

- `proof_math` -> `proof_factoring`
- `proof_agent` -> `proof_factoring`
- `icarus_affine` -> `proof_affine`
- `icarus_projective_v2` -> `proof_affine`

## Output directories (consistent defaults)

- `gpt_text` (`scripts/gpt_text.py`): `runs/<run_name>/...`
- `proof_factoring` (`experiments/proof_factoring_gpt.py`): `runs/proof_factoring_gpt/ckpt_last` (or `runs/<run_name>/...` via args)
- `proof_affine` (`experiments/proof_affine_gpt.py`): `runs/proof_affine_gpt/ckpt_last` and `ckpt_best`

Use these knobs to override paths:

- text: `RUN_NAME`, `RUNS_DIR`
- proof: `--run_name`, `--runs_dir`, `--ckpt`
- icarus: `--out_dir`, `--resume_checkpoint`

CLI naming is now aligned across experiment scripts:

- Iterations: `--train_steps` and `--max_iters` are accepted aliases
- Eval cadence: `--eval_every` and `--eval_interval` are accepted aliases

Experiment feedback is standardized:

- Console progress prints use consistent `update .../N | loss ... | lr ...` style
- `metrics.jsonl` is written to each run directory with train/eval events
- `summary.json` is written at the end of training with key run outcomes

## Architecture and action-space differences

### 1) `gpt_text` (`scripts/gpt_text.py`)

- **Backbone**: shared `ExplicitGPT` (TF, GQA, RoPE/ALiBi options, flash-attn path)
- **Input space**: raw token streams from text datasets (`enwik8`, `tinyshakespeare`, `names`)
- **Output space**: next-token language modeling over tokenizer vocabulary
- **Objective**: standard autoregressive LM loss

### 2) `proof_factoring` (`experiments/proof_factoring_gpt.py`)

- **Backbone**: shared `ExplicitGPT`
- **Input space**: serialized symbolic equation states + control tokens (`GOAL`, `STATE`, `ACT`)
- **Action/language space**: rewrite action tokens (`A_<rule>_<node_index>`)
- **Objective**: masked supervised action prediction + environment rollout solve-rate eval
- **Notes**: curriculum support and traceable eval episodes

### 3) `proof_affine` (`experiments/proof_affine_gpt.py`)

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
  -e APP_ENTRY=proof_factoring \
  -e APP_ARGS="--run_name proof_factoring_gpt --train_steps 12000" \
  gpt
```

Icarus model:

```bash
docker compose run --rm \
  -e APP_ENTRY=proof_affine \
  -e APP_ARGS="--mode train --out_dir runs/proof_affine_gpt --resume_checkpoint runs/proof_affine_gpt/ckpt_last/ckpt-200 --max_iters 8000" \
  gpt
```
