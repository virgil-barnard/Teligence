import os
import time

import numpy as np

from config import GPTConfig, validate_config
from data_utils import load_dataset, make_random_window_dataset
from modeling import ExplicitGPT, get_precision_dtypes, set_precision
from run_utils import RunTracker
from runtime import setup_runtime
from train_utils import (
    build_train_micro_step,
    build_train_state,
    escape_preview_text,
    evaluate_model,
    generate_from_prompt,
    generate_names,
    print_runtime_banner,
)


def main():
    setup_runtime(seed=42)

    dataset_name = os.environ.get("DATASET", "enwik8").strip().lower()
    data_dir = os.environ.get("DATA_DIR", "./data")
    ds_bundle = load_dataset(dataset_name, data_dir)

    cfg = GPTConfig(vocab_size=ds_bundle.vocab_size)
    validate_config(cfg)
    if ds_bundle.name == "names" and cfg.preview_prompt == "The ":
        cfg.preview_prompt = "mar"

    tracker = RunTracker(ds_bundle.name, cfg.run_name, cfg.runs_dir)

    set_precision(cfg.use_bf16)
    compute_dtype, cache_dtype = get_precision_dtypes()
    print_runtime_banner(cfg, compute_dtype, cache_dtype, tracker.run_id, tracker.run_dir)

    block_len = cfg.seq_len + 1
    if len(ds_bundle.train_tokens) <= block_len:
        raise ValueError("Training token stream is too short for current seq_len")
    print(
        f"train/val/test tokens: {len(ds_bundle.train_tokens):,} / {len(ds_bundle.val_tokens):,} / {len(ds_bundle.test_tokens):,}"
    )
    print(f"block_len: {block_len}")

    model = ExplicitGPT(cfg)
    state = build_train_state(cfg, model, tracker.run_dir)
    train_micro_step = build_train_micro_step(cfg, model, state)

    param_count = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
    print(f"num params: {param_count:,}")

    ds = make_random_window_dataset(ds_bundle.train_tokens, cfg)
    it = iter(ds)

    print("\n--- training ---")
    micro = 0
    t0 = time.time()
    last_log_time = t0
    last_log_updates = int(state.update_step.numpy())
    best_val_bpc = float("inf")
    best_val_step = 0

    tracker.log_event(
        {
            "event": "run_start",
            "num_updates": cfg.num_updates,
            "batch_size": cfg.batch_size,
            "seq_len": cfg.seq_len,
            "attn_window": cfg.attn_window,
            "eval_every": cfg.eval_every,
            "eval_tokens": cfg.eval_tokens,
        }
    )

    while int(state.update_step.numpy()) < cfg.num_updates:
        x, y = next(it)
        loss, gnorm, lr, updated = train_micro_step(x, y)
        micro += 1

        upd = int(state.update_step.numpy())
        if upd == 0 and micro == 1:
            print(f"micro {micro:6d} | update {upd:5d}/{cfg.num_updates} | loss {loss.numpy():.4f}")

        if (upd > 0) and bool(updated.numpy()) and (upd % cfg.log_every == 0):
            now = time.time()
            dt = now - last_log_time
            du = upd - last_log_updates
            tok_per_update = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps
            toks = du * tok_per_update
            tps = toks / max(1e-9, dt)

            print(
                f"micro {micro:6d} | update {upd:5d}/{cfg.num_updates} | loss {loss.numpy():.4f} "
                f"| gnorm {gnorm.numpy():.3f} | lr {lr.numpy():.2e} | tok/s {tps:,.0f}"
            )

            train_rec = {
                "event": "train",
                "micro_step": micro,
                "update": upd,
                "train_loss": float(loss.numpy()),
                "grad_norm": float(gnorm.numpy()),
                "lr": float(lr.numpy()),
                "tok_per_s": float(tps),
            }

            preview = generate_from_prompt(
                model,
                cfg.preview_prompt,
                ds_bundle.stoi,
                ds_bundle.itos,
                ds_bundle.bos,
                max_new_tokens=cfg.preview_max_new_tokens,
                temperature=0.7,
                top_p=0.9,
            )
            if preview:
                shown = escape_preview_text(preview)
                print(f"prompt '{cfg.preview_prompt}' -> {shown}")
                train_rec["preview"] = shown

            tracker.log_event(train_rec)
            last_log_time = now
            last_log_updates = upd

        if (upd > 0) and bool(updated.numpy()) and (upd % cfg.eval_every == 0):
            val_nll, val_bpc, val_ppl = evaluate_model(model, ds_bundle.val_tokens, cfg, cfg.eval_tokens)
            is_best = False
            mark = ""
            if val_bpc < best_val_bpc:
                best_val_bpc = val_bpc
                best_val_step = upd
                best_path = state.best_manager.save(checkpoint_number=upd)
                is_best = True
                mark = " | best"
                print(f"saved best checkpoint: {best_path}")

            tracker.log_event(
                {
                    "event": "eval",
                    "update": upd,
                    "val_nll": float(val_nll),
                    "val_bpc": float(val_bpc),
                    "val_ppl": float(val_ppl),
                    "is_best": is_best,
                }
            )
            print(
                f"eval  update {upd:5d}/{cfg.num_updates} | val_nll {val_nll:.4f} | "
                f"val_bpc {val_bpc:.4f} | val_ppl {val_ppl:.3f}{mark}"
            )

        if cfg.save_every and bool(updated.numpy()) and (upd % cfg.save_every == 0):
            path = state.manager.save(checkpoint_number=upd)
            print(f"saved checkpoint: {path}")

    path = state.manager.save(checkpoint_number=int(state.update_step.numpy()))
    print(f"saved checkpoint: {path}")

    test_nll, test_bpc, test_ppl = evaluate_model(model, ds_bundle.test_tokens, cfg, cfg.eval_tokens)
    print(f"test metrics | nll {test_nll:.4f} | bpc {test_bpc:.4f} | ppl {test_ppl:.3f}")

    summary = {
        "run_id": tracker.run_id,
        "dataset": ds_bundle.name,
        "num_updates": int(state.update_step.numpy()),
        "best_val_bpc": float(best_val_bpc),
        "best_val_step": int(best_val_step),
        "test_nll": float(test_nll),
        "test_bpc": float(test_bpc),
        "test_ppl": float(test_ppl),
        "last_ckpt": path,
        "best_ckpt": state.best_manager.latest_checkpoint,
    }
    tracker.write_summary(summary)
    tracker.log_event({"event": "run_end", **summary})
    print(f"wrote run summary: {tracker.summary_json}")

    if ds_bundle.name == "names":
        generate_names(model, ds_bundle.bos, ds_bundle.itos, num=20, max_new_tokens=256, temperature=0.7, top_p=0.9)


if __name__ == "__main__":
    main()
