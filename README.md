## Teligence GPT Experiments

This repo trains a small TensorFlow GPT model with GPU support in Docker, focused on text-domain architecture evaluation.

### Quick start

```bash
docker compose up --build
```

This starts:

- `gpt` training
- `tensorboard` at `http://localhost:6006`

If you already built the image:

```bash
docker compose up
```

### Startup script options

The default startup path is still `gpt.py`.

- Default (existing behavior):

```bash
docker compose run --rm gpt
```

- Switch startup script through launcher env:

```bash
docker compose run --rm -e APP_ENTRY=proof_agent gpt
```

- Pass CLI args to selected startup script:

```bash
docker compose run --rm \
  -e APP_ENTRY=proof_agent \
  -e APP_ARGS="--mode sample --seq_len 128 --train_steps 200" \
  gpt
```

- Optional dedicated compose service for proof prototype:

```bash
docker compose --profile proof run --rm proof_agent
```

- Run the projective action-pointer prototype (TensorFlow backbone):

```bash
docker compose run --rm -e APP_ENTRY=icarus_projective_v2 gpt
docker compose run --rm -e APP_ENTRY=icarus_projective_v2 -e APP_ARGS="--mode rollout --rollout_mode search --demo_tasks 5" gpt
```

- Optional dedicated compose service for Icarus prototype:

```bash
docker compose --profile icarus run --rm icarus_projective_v2
```

### Text datasets

Supported `DATASET` values:

- `enwik8` (byte-level default)
- `tinyshakespeare` (char-level default)
- `names` (char-level default)

Examples:

```bash
docker compose run --rm -e DATASET=enwik8 gpt
docker compose run --rm -e DATASET=tinyshakespeare gpt
docker compose run --rm -e DATASET=names gpt
```

For `tinyshakespeare`, if download is blocked, set a local file path:

```bash
docker compose run --rm \
  -e DATASET=tinyshakespeare \
  -e TINY_SHAKESPEARE_PATH=./data/tinyshakespeare.txt \
  gpt
```

### Resume training

- Resume same run id: set `RUN_NAME` to prior run id.
- Resume from specific run/path: `RESUME_FROM=<run_id|checkpoint_dir|checkpoint_prefix>`.
- Continue for relative extra steps: `EXTRA_UPDATES=<n>`.

Example:

```bash
docker compose run --rm \
  -e RUN_NAME=my_run \
  -e RESUME_FROM=my_run \
  -e EXTRA_UPDATES=5000 \
  gpt
```

### TensorBoard: what to watch

Enable visualization logs (already enabled by default in compose):

- `train/loss`, `train/lr`, `train/grad_norm`, `train/tok_per_s`
- `eval/val_nll`, `eval/val_bpc`, `eval/val_ppl`
- weight histograms (`weights/*`)
- attention probe images (`attention/*`) on control prompt

Useful knobs:

- `LOG_EVERY`, `EVAL_EVERY`
- `TB_HIST_EVERY`
- `ATTN_VIZ_EVERY`, `ATTN_VIZ_MAX_LAYERS`, `ATTN_VIZ_MAX_HEADS`

Tip: if you only see one point, check you selected the latest run in TensorBoard and your `LOG_EVERY` is not too sparse.

### Cross-domain benchmark matrix (text-only)

Run architecture comparison across text datasets:

```bash
python benchmark_matrix.py
```

Defaults:

- profiles: `enwik8,tinyshakespeare,names`
- outputs:
  - `runs/<matrix_name>_report.json`
  - `runs/<matrix_name>_report.csv`

Useful knobs:

- `BENCH_NUM_UPDATES`, `BENCH_LOG_EVERY`, `BENCH_EVAL_EVERY`, `BENCH_EVAL_TOKENS`
- `BENCH_PROFILES=enwik8,tinyshakespeare,names`
- `BENCH_USE_DOCKER=1`

### Smoke tests

```bash
docker compose run --rm gpt python -m unittest tests.test_smoke
```

### Project layout

- `gpt.py`: training entrypoint
- `config.py`: config/env parsing and validation
- `data_utils.py`: text dataset loading and dataloaders
- `tokenizer.py`: tokenizer strategies
- `modeling.py`: model and attention implementation
- `train_utils.py`: train/eval/sampling/TensorBoard helpers
- `run_utils.py`: run artifact and metrics logging
- `runtime.py`: runtime setup (GPU memory growth, seeds)
- `benchmark_matrix.py`: text-domain benchmark runner
