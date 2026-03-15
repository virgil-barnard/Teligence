# Research Automation Scaffolding

This directory contains scaffolding for an autoresearch-style workflow tailored to this repository.

## Files

- `research/program.md`: human-authored instructions for research agents.
- `research/score.py`: shared metric scoring helpers by track.
- `research/results.template.tsv`: header template for result logging.

## Initialize a run ledger

If you are running locally (not Docker), first set up a virtualenv from repo root:

```bash
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

```bash
python scripts/autoresearch_loop.py init
```

This creates `research/results.tsv` (gitignored).

## Record one experiment run (simple mode)

Use track + run name and the helper builds the Docker command and summary path:

```bash
python scripts/autoresearch_loop.py run \
  --track gpt_text \
  --run-name auto_text \
  --dataset tinyshakespeare \
  --description "baseline"
```

Proof factoring with extra args:

```bash
python scripts/autoresearch_loop.py run \
  --track proof_factoring \
  --run-name auto_factoring \
  --extra-args "--train_steps 400 --eval_every 100 --eval_episodes 30" \
  --description "baseline"
```

Proof affine with extra args:

```bash
python scripts/autoresearch_loop.py run \
  --track proof_affine \
  --run-name auto_affine \
  --extra-args "--max_iters 400 --eval_interval 100" \
  --description "baseline"
```

## Record one experiment run (custom mode)

```bash
python scripts/autoresearch_loop.py run \
  --track gpt_text \
  --command "docker compose run --rm -e APP_ENTRY=gpt_text -e RUN_NAME=auto_text gpt" \
  --summary-path runs/auto_text/summary.json \
  --description "baseline"
```

The script writes:

- command output log to `research/logs/`
- one row to `research/results.tsv`
- automatic `keep`/`discard` decision based on normalized score

## View leaderboard

```bash
python scripts/autoresearch_loop.py leaderboard
```

Optional:

```bash
python scripts/autoresearch_loop.py leaderboard --limit 20
```
