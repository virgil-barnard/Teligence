# Teligence Autoresearch Program

This file is the human-authored instruction set for coding/research agents that run autonomous experiment cycles in this repository.

## Scope

Primary optimization targets:

- `scripts/gpt_text.py`
- `experiments/proof_factoring_gpt.py`
- `experiments/proof_affine_gpt.py`
- shared model and training utilities under `teligence/`

Read-only unless explicitly approved by the human:

- datasets under `data/`
- generated artifacts under `runs/`
- Docker/runtime infrastructure unless required for reliability

## Tracks and objectives

Run each track independently and compare only within the same track.

1. `gpt_text`
   - Primary objective: lower `best_val_bpc`.
   - Secondary objectives: lower `test_bpc`, maintain throughput.

2. `proof_factoring`
   - Primary objective: higher `solve_rate` (from summary `best_primary_metric` with name `solve_rate`).
   - Secondary objectives: lower `avg_steps`, maintain valid action quality.

3. `proof_affine`
   - Primary objective: lower `val_policy_loss` (summary `best_primary_metric` with name `val_policy_loss`).
   - Secondary objectives: higher `quick_rollout_success`, lower search cost.

## Experiment protocol

1. Start from a clean branch (recommended naming: `autoresearch/<track>/<tag>`).
2. Make one coherent code change.
3. Commit with a short rationale.
4. Run exactly one experiment command for the target track.
   - Preferred: use `python scripts/autoresearch_loop.py run ...` so logs and scoring are recorded consistently.
5. Parse `summary.json` and `metrics.jsonl` into a normalized row in `research/results.tsv`.
6. Keep/discard decision:
   - keep only if primary metric improves by a meaningful amount
   - discard if worse/equal and not materially simpler
   - mark `crash` or `timeout` explicitly
7. Repeat.

Timeout policy:

- If no terminal output for an extended period, inspect logs before assuming hang.
- Hard timeout per experiment should be set per track and machine.

## Standard commands (examples)

Text:

```bash
docker compose run --rm -e APP_ENTRY=gpt_text -e DATASET=tinyshakespeare -e RUN_NAME=auto_text gpt
```

Factoring:

```bash
docker compose run --rm -e APP_ENTRY=proof_factoring -e APP_ARGS="--run_name auto_factoring" gpt
```

Affine:

```bash
docker compose run --rm -e APP_ENTRY=proof_affine -e APP_ARGS="--out_dir runs/auto_affine" gpt
```

## Required result fields

Every run should record:

- `timestamp`
- `track`
- `branch`
- `commit`
- `status` (`keep|discard|crash|timeout`)
- `primary_metric_name`
- `primary_metric_value`
- `score` (direction-normalized scalar; higher is better)
- `run_dir`
- `summary_path`
- `description`

## High-value metrics to add over time

Common:

- `wall_clock_s`, `time_to_first_eval_s`
- `steps_per_sec` and task-specific throughput
- explicit resume metadata (`restore_checkpoint`, start/end steps)
- stability flags (`nan_detected`, `inf_detected`)

Text-specific:

- `train_loss_last`, `val_minus_train_gap`
- sample degeneration proxies (repeat ratios)

Factoring-specific:

- valid/invalid action rates
- solved-by-step histogram
- per-rule action/usefulness stats

Affine-specific:

- per-task solve rate
- average search expansions / branching factor
- value calibration diagnostics
