import math
import os
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

from config import GPTConfig
from data_utils import iter_eval_batches
from modeling import get_precision_dtypes


def lr_schedule(cfg: GPTConfig, update_step_int64):
    step = tf.cast(update_step_int64, tf.float32)
    warm = tf.cast(cfg.warmup_steps, tf.float32)
    total = tf.cast(cfg.num_updates, tf.float32)
    warm_lr = cfg.base_lr * (step / tf.maximum(1.0, warm))
    progress = (step - warm) / tf.maximum(1.0, total - warm)
    progress = tf.clip_by_value(progress, 0.0, 1.0)
    cosine = 0.5 * (1.0 + tf.cos(math.pi * progress))
    cos_lr = cfg.min_lr + (cfg.base_lr - cfg.min_lr) * cosine
    return tf.where(step < warm, warm_lr, cos_lr)


def apply_weight_decay(cfg: GPTConfig, vars_, lr):
    if cfg.weight_decay <= 0.0:
        return
    for v in vars_:
        r = v.shape.rank
        if r is not None and r >= 2:
            v.assign_sub(lr * cfg.weight_decay * v)


def sample_from_logits(logits_1d_f32, temperature=1.0, top_k=0, top_p=0.0):
    logits = tf.cast(logits_1d_f32, tf.float32) / tf.cast(temperature, tf.float32)
    if top_k and top_k > 0:
        values, _ = tf.math.top_k(logits, k=top_k)
        kth = values[-1]
        logits = tf.where(logits < kth, tf.constant(-1e9, tf.float32), logits)
    if top_p and top_p > 0.0:
        idx = tf.argsort(logits, direction="DESCENDING")
        sorted_logits = tf.gather(logits, idx)
        probs = tf.nn.softmax(sorted_logits)
        cdf = tf.cumsum(probs)
        keep = cdf <= top_p
        keep = tf.tensor_scatter_nd_update(keep, [[0]], [True])
        sorted_logits = tf.where(keep, sorted_logits, tf.constant(-1e9, tf.float32))
        logits = tf.scatter_nd(tf.expand_dims(idx, 1), sorted_logits, [tf.shape(logits)[0]])
    return int(tf.random.categorical(tf.expand_dims(logits, 0), 1)[0, 0].numpy())


def escape_preview_text(s):
    return s.encode("unicode_escape", "backslashreplace").decode("ascii")


@dataclass
class TrainState:
    opt: tf.keras.optimizers.Optimizer
    train_vars: list
    grad_accums: list
    accum_count: tf.Variable
    update_step: tf.Variable
    manager: tf.train.CheckpointManager
    best_manager: tf.train.CheckpointManager


def build_train_state(cfg: GPTConfig, model, run_dir: str) -> TrainState:
    opt = tf.keras.optimizers.Adam(learning_rate=cfg.base_lr, beta_1=0.9, beta_2=0.95, epsilon=1e-8)

    _ = model(tf.zeros([cfg.batch_size, cfg.seq_len], dtype=tf.int32), training=False)
    train_vars = model.trainable_variables
    grad_accums = [tf.Variable(tf.zeros_like(v), trainable=False) for v in train_vars]
    accum_count = tf.Variable(0, dtype=tf.int32, trainable=False)
    update_step = tf.Variable(0, dtype=tf.int64, trainable=False)

    ckpt = tf.train.Checkpoint(model=model, update_step=update_step)
    last_ckpt_dir = os.path.join(run_dir, "ckpt_last")
    best_ckpt_dir = os.path.join(run_dir, "ckpt_best")
    os.makedirs(last_ckpt_dir, exist_ok=True)
    os.makedirs(best_ckpt_dir, exist_ok=True)

    manager = tf.train.CheckpointManager(ckpt, last_ckpt_dir, max_to_keep=cfg.keep_last_ckpts)
    best_manager = tf.train.CheckpointManager(ckpt, best_ckpt_dir, max_to_keep=cfg.keep_best_ckpts)

    def resolve_resume_checkpoint():
        if not cfg.resume_from.strip():
            return None
        resume_from = cfg.resume_from.strip()

        if os.path.exists(resume_from + ".index"):
            return resume_from

        if os.path.isdir(resume_from):
            return tf.train.latest_checkpoint(resume_from)

        run_ckpt_dir = os.path.join(cfg.runs_dir, resume_from, "ckpt_last")
        if os.path.isdir(run_ckpt_dir):
            return tf.train.latest_checkpoint(run_ckpt_dir)

        return None

    resume_ckpt = manager.latest_checkpoint or resolve_resume_checkpoint()
    if resume_ckpt:
        ckpt.restore(resume_ckpt)
        print(f"restored checkpoint: {resume_ckpt} (update_step={int(update_step.numpy())})")
    else:
        print("no checkpoint found, starting fresh")

    return TrainState(opt, train_vars, grad_accums, accum_count, update_step, manager, best_manager)


def build_train_micro_step(cfg: GPTConfig, model, state: TrainState):
    @tf.function(jit_compile=cfg.jit_compile)
    def train_micro_step(x, y):
        with tf.GradientTape() as tape:
            logits = model(x, training=True)
            logits_f = tf.cast(logits, tf.float32)
            loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits_f))

        grads = tape.gradient(loss, state.train_vars)
        for ga, g in zip(state.grad_accums, grads):
            if g is None:
                continue
            ga.assign_add(tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)))

        state.accum_count.assign_add(1)
        did_update = tf.equal(state.accum_count, tf.cast(cfg.grad_accum_steps, tf.int32))

        def apply_update():
            lr = lr_schedule(cfg, state.update_step)
            state.opt.learning_rate.assign(lr)

            denom = tf.cast(cfg.grad_accum_steps, tf.float32)
            avg_grads_f32 = [tf.cast(ga, tf.float32) / denom for ga in state.grad_accums]
            avg_grads_f32, gnorm = tf.clip_by_global_norm(avg_grads_f32, cfg.grad_clip_norm)

            cast_grads = [tf.cast(g, v.dtype) for g, v in zip(avg_grads_f32, state.train_vars)]
            state.opt.apply_gradients(zip(cast_grads, state.train_vars))
            apply_weight_decay(cfg, state.train_vars, lr)

            for ga in state.grad_accums:
                ga.assign(tf.zeros_like(ga))
            state.accum_count.assign(0)
            state.update_step.assign_add(1)
            return gnorm, lr, tf.constant(True)

        def no_update():
            return tf.constant(0.0, tf.float32), tf.cast(state.opt.learning_rate, tf.float32), tf.constant(False)

        gnorm, lr, updated = tf.cond(did_update, apply_update, no_update)
        return loss, gnorm, lr, updated

    return train_micro_step


def evaluate_model(model, tokens_np: np.ndarray, cfg: GPTConfig, max_tokens: int):
    total_nll = 0.0
    total_tok = 0
    for x, y in iter_eval_batches(tokens_np, cfg, max_tokens):
        logits = model(tf.constant(x, dtype=tf.int32), training=False)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.constant(y, dtype=tf.int32), logits=tf.cast(logits, tf.float32))
        total_nll += float(tf.reduce_sum(loss).numpy())
        total_tok += int(np.prod(y.shape))
    nll = total_nll / max(1, total_tok)
    bpc = nll / math.log(2.0)
    ppl = math.exp(min(20.0, nll))
    return nll, bpc, ppl


def generate_names(model, bos, itos, num=20, max_new_tokens=256, temperature=0.8, top_k=0, top_p=0.9):
    print("\n--- inference (new, hallucinated names) ---")
    cache = model.init_kv_cache(batch_size=1, as_variables=True)
    for i in range(num):
        model.reset_kv_cache(cache)
        token = bos
        out = []
        for t in range(max_new_tokens):
            logits, cache = model.step_ring(tf.constant([token], tf.int32), t, cache)
            next_id = sample_from_logits(tf.cast(logits[0], tf.float32), temperature=temperature, top_k=top_k, top_p=top_p)
            if next_id == bos:
                break
            out.append(itos[next_id])
            token = next_id
        print(f"sample {i+1:2d}: {''.join(out)}")


def generate_from_prompt(model, prompt, stoi, itos, bos, max_new_tokens=64, temperature=0.8, top_k=0, top_p=0.9):
    clean = "".join(ch for ch in prompt if ch in stoi)
    if not clean:
        return ""

    cache = model.init_kv_cache(batch_size=1, as_variables=True)
    model.reset_kv_cache(cache)
    token = bos
    t = 0

    for ch in clean:
        _, cache = model.step_ring(tf.constant([token], tf.int32), t, cache)
        token = stoi[ch]
        t += 1

    logits, cache = model.step_ring(tf.constant([token], tf.int32), t, cache)
    t += 1

    out = []
    for _ in range(max_new_tokens):
        next_id = sample_from_logits(tf.cast(logits[0], tf.float32), temperature=temperature, top_k=top_k, top_p=top_p)
        if next_id == bos:
            break
        out.append(itos[next_id])
        logits, cache = model.step_ring(tf.constant([next_id], tf.int32), t, cache)
        t += 1
    return clean + "".join(out)


def print_runtime_banner(cfg: GPTConfig, compute_dtype, cache_dtype, run_id, run_dir):
    print(f"pos_encoding={cfg.pos_encoding} | fuse_qkv={cfg.fuse_qkv}")
    print(f"compute dtype: {compute_dtype.name} | cache dtype: {cache_dtype.name}")
    print(f"T(seq_len)={cfg.seq_len} | W(attn_window)={cfg.attn_window}")
    print(f"use_flash_attn={cfg.use_flash_attn} | qblk={cfg.flash_q_block} | kblk={cfg.flash_k_block}")
    print(f"flash_parallel_iterations={cfg.flash_parallel_iterations} | swap_memory={cfg.flash_swap_memory}")
    print(f"flash_recompute_grad={cfg.flash_recompute_grad} | jit_compile={cfg.jit_compile}")
    print(f"grad_accum_steps={cfg.grad_accum_steps} | batch_size={cfg.batch_size} | effective_batch={cfg.batch_size * cfg.grad_accum_steps}")
    print(f"run_id={run_id} | run_dir={run_dir}")
