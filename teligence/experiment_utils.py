import math
import random
import time
import json
import os

import numpy as np
import tensorflow as tf

from teligence.config import GPTConfig, validate_config


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def cosine_lr(step: int, base_lr: float, min_lr: float, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
    t = min(1.0, max(0.0, t))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return min_lr + (base_lr - min_lr) * cos


def assign_optimizer_lr(opt: tf.keras.optimizers.Optimizer, lr: float) -> None:
    if isinstance(opt.learning_rate, tf.Variable):
        opt.learning_rate.assign(lr)
    else:
        opt.learning_rate = lr


def apply_weight_decay(vars_, lr: float, wd: float) -> None:
    if wd <= 0.0:
        return
    for v in vars_:
        if v.shape.rank is not None and v.shape.rank >= 2:
            v.assign_sub(tf.cast(lr * wd, v.dtype) * v)


def build_backbone_cfg(
    *,
    vocab_size: int,
    seq_len: int,
    n_layer: int,
    n_embd: int,
    n_head: int,
    n_kv_head: int,
    dropout: float,
    batch_size: int,
    num_updates: int,
    base_lr: float,
    min_lr: float,
    warmup_steps: int,
    weight_decay: float,
    use_bf16: bool,
    use_flash_attn: bool,
    flash_q_block: int,
    flash_k_block: int,
) -> GPTConfig:
    qblk = flash_q_block
    kblk = flash_k_block
    while qblk > 1 and seq_len % qblk != 0:
        qblk //= 2
    while kblk > 1 and seq_len % kblk != 0:
        kblk //= 2
    gcfg = GPTConfig(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_embd=n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        mlp_mult=4,
        seq_len=seq_len,
        attn_window=min(seq_len, 256),
        dropout=dropout,
        use_bf16=use_bf16,
        use_flash_attn=use_flash_attn,
        flash_q_block=qblk,
        flash_k_block=kblk,
        batch_size=batch_size,
        num_updates=num_updates,
        base_lr=base_lr,
        min_lr=min_lr,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )
    validate_config(gcfg)
    return gcfg


def should_eval(step_value: int, end_step_value: int, eval_every: int) -> bool:
    if eval_every <= 0:
        return step_value == end_step_value
    return (step_value % eval_every == 0) or (step_value == end_step_value)


def maybe_print_train_step(
    *,
    step_value: int,
    end_step_value: int,
    print_every: int,
    loss_value: float,
    lr_value: float,
    timer_state: dict,
    prefix: str = "step",
) -> None:
    if print_every <= 0:
        return
    if step_value % print_every != 0:
        return
    t0 = timer_state.get("t0")
    if t0 is None:
        t0 = time.time()
    dt = time.time() - t0
    print(f"{prefix} {step_value:5d}/{end_step_value} loss {loss_value:.4f} lr {lr_value:.2e}  ({dt:.1f}s)")
    timer_state["t0"] = time.time()


class ExperimentLogger:
    def __init__(self, run_dir: str, dataset_name: str, run_id: str):
        self.run_dir = run_dir
        self.dataset_name = dataset_name
        self.run_id = run_id
        os.makedirs(self.run_dir, exist_ok=True)
        self.metrics_jsonl = os.path.join(self.run_dir, "metrics.jsonl")
        self.summary_json = os.path.join(self.run_dir, "summary.json")

    def log_event(self, event: dict) -> None:
        rec = dict(event)
        rec["dataset"] = self.dataset_name
        rec["run_id"] = self.run_id
        rec["time"] = time.time()
        with open(self.metrics_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")

    def write_summary(self, summary: dict) -> None:
        payload = dict(summary)
        payload["dataset"] = self.dataset_name
        payload["run_id"] = self.run_id
        with open(self.summary_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
