## Teligence GPT Experiments

This repo trains a small TensorFlow GPT model with GPU support in Docker.

### Quick start (single run)

```bash
docker compose up --build
```

By default this trains on `enwik8` and prints:

- training loss and throughput
- prompt preview samples every `LOG_EVERY`
- validation metrics every `EVAL_EVERY` (`val_nll`, `val_bpc`, `val_ppl`)
- final test metrics

### Useful env vars

```bash
docker compose run --rm \
  -e DATASET=enwik8 \
  -e NUM_UPDATES=3000 \
  -e LOG_EVERY=50 \
  -e EVAL_EVERY=50 \
  -e EVAL_TOKENS=262144 \
  -e RUN_NAME=my_run \
  gpt
```

Resume training options:

- Resume same run directory (same run id): set `RUN_NAME` to the old run id.
- Resume from another run id/path:
  - `RESUME_FROM=<run_id>` (looks in `runs/<run_id>/ckpt_last/`), or
  - `RESUME_FROM=<checkpoint_prefix_or_dir>`.
- Continue for additional updates relative to restored step:
  - `EXTRA_UPDATES=<n>`.

Example:

```bash
docker compose run --rm \
  -e RUN_NAME=my_run \
  -e RESUME_FROM=my_old_run \
  -e EXTRA_UPDATES=5000 \
  gpt
```

Model/hparam knobs (examples):

- `BASE_LR`, `MIN_LR`, `WARMUP_STEPS`, `WEIGHT_DECAY`, `DROPOUT`
- `N_LAYER`, `N_EMBD`, `N_HEAD`, `N_KV_HEAD`, `MLP_MULT`
- `SEQ_LEN`, `ATTN_WINDOW`, `BATCH_SIZE`

### Artifacts

Each run writes to `runs/<run_id>/`:

- `metrics.jsonl` (train/eval events)
- `summary.json` (best validation + final test metrics)
- `ckpt_last/` (latest checkpoints)
- `ckpt_best/` (best by validation BPC)

Repo hygiene defaults:

- experiment outputs (`runs/`, `data/`, checkpoints, logs) are git-ignored
- Docker build context is minimized via `.dockerignore` for faster `docker compose up --build`

### Sweep mode

Run a short hyperparameter sweep:

```bash
python sweep.py
```

It launches multiple trials (via `docker compose run` by default), ranks them by `best_val_bpc`, and writes a sweep report to `runs/<sweep_name>_report.json`.

### Smoke tests

Run lightweight module-integrity tests in Docker:

```bash
docker compose run --rm gpt python -m unittest tests.test_smoke
```

### Project layout

- `gpt.py`: main training entrypoint
- `config.py`: config/env parsing and validation
- `data_utils.py`: dataset loading and dataloaders
- `modeling.py`: model/attention implementation
- `train_utils.py`: optimizer, train step, eval, sampling helpers
- `run_utils.py`: run artifact and metrics logging
- `runtime.py`: GPU memory-growth and reproducibility setup
