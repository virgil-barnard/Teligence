"""
The most atomic way to train + run a modern(ish) GPT in TensorFlow, with explicit math.
This file is the complete algorithm.
Everything else is just efficiency.

Adds ALL THREE requested upgrades:
1) Sliding-window training matching inference ring-cache semantics:
   - train with seq_len T, but attention window W (same W used by inference ring cache)
   - mask: k in [q-(W-1), q] (causal + local), i.e. exactly what the ring cache stores
2) ALiBi option:
   - pos_encoding = "rope" or "alibi"
   - ALiBi works in flash-training and ring-cache inference (stores pos ring cache)
3) Fused QKV:
   - one matmul to get q,k,v

Also:
- FlashAttention-style online softmax via tf.while_loop (no (T,T))
- Precomputed block tables (positions + prune bounds)
- Optional recompute-grad (checkpointing flash)
- BF16 compute optional (softmax/loss FP32)
- BOS-aligned packed blocks
- Gradient accumulation
- Save/Load checkpoints
"""

import os
import math
import time
import random
import urllib.request
from dataclasses import dataclass

import numpy as np
import tensorflow as tf


# ----------------------------
# GPU friendliness for local training
# ----------------------------
for g in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(g, True)
    except Exception:
        pass


# ----------------------------
# Reproducibility
# ----------------------------
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)


# ----------------------------
# Dataset: list[str] docs
# ----------------------------
if not os.path.exists("input.txt"):
    names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")

docs = [line.strip() for line in open("input.txt") if line.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")


# ----------------------------
# Tokenizer: char-level + single BOS token (also used as EOS/boundary)
# ----------------------------
uchars = sorted(set("".join(docs)))
stoi = {ch: i for i, ch in enumerate(uchars)}
itos = {i: ch for ch, i in stoi.items()}

BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size} (including BOS/EOS/boundary)")


# ----------------------------
# ALiBi slopes (standard recipe)
# ----------------------------
def alibi_slopes(n_head: int) -> np.ndarray:
    """
    Returns slopes for ALiBi, length n_head.
    Common reference implementation used across many repos.
    """
    def slopes_power_of_2(n: int):
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    if float(math.log2(n_head)).is_integer():
        s = slopes_power_of_2(n_head)
    else:
        closest = 2 ** math.floor(math.log2(n_head))
        s = slopes_power_of_2(closest)
        s2 = slopes_power_of_2(2 * closest)
        s += s2[0::2][: (n_head - closest)]
    return np.array(s, dtype=np.float32)


# ----------------------------
# Config
# ----------------------------
@dataclass
class GPTConfig:
    vocab_size: int

    # shape
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6
    n_kv_head: int = 2          # GQA: must divide n_head
    mlp_mult: int = 4

    # training length (T) and local window (W)
    seq_len: int = 512
    attn_window: int = 256      # also used as inference ring-cache length

    # positional encoding
    pos_encoding: str = "rope"  # "rope" or "alibi"

    # RoPE
    rope_base: float = 10000.0
    rope_scale_factor: float = 1.0

    # norms + MLP
    rmsnorm_eps: float = 1e-5
    rmsnorm_scale: bool = True
    use_swiglu: bool = True

    # fused qkv
    fuse_qkv: bool = True

    # regularization
    dropout: float = 0.1

    # output
    tie_embeddings: bool = True

    # compute
    use_bf16: bool = True

    # flash attention
    use_flash_attn: bool = True
    flash_q_block: int = 128
    flash_k_block: int = 128

    # while_loop tuning
    flash_parallel_iterations: int = 1
    flash_swap_memory: bool = False

    # activation checkpointing for flash
    flash_recompute_grad: bool = True

    # graph compilation
    jit_compile: bool = False  # enable only after it runs cleanly on your machine

    # optimization
    base_lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0

    # gradient accumulation
    grad_accum_steps: int = 1

    # run
    batch_size: int = 32
    num_updates: int = 3000
    log_every: int = 50
    save_every: int = 500

    # checkpoint
    ckpt_dir: str = "./ckpt_explicit_gpt"


cfg = GPTConfig(vocab_size=vocab_size)

assert cfg.n_embd % cfg.n_head == 0
assert cfg.n_head % cfg.n_kv_head == 0
assert cfg.attn_window <= cfg.seq_len
assert cfg.pos_encoding in ("rope", "alibi")

head_dim = cfg.n_embd // cfg.n_head
assert head_dim % 2 == 0, "RoPE requires even head_dim (n_embd//n_head)."

if cfg.use_flash_attn:
    assert cfg.seq_len % cfg.flash_q_block == 0, "seq_len must be divisible by flash_q_block"
    assert cfg.seq_len % cfg.flash_k_block == 0, "seq_len must be divisible by flash_k_block"

compute_dtype = tf.bfloat16 if cfg.use_bf16 else tf.float32
cache_dtype = compute_dtype

print(f"pos_encoding={cfg.pos_encoding} | fuse_qkv={cfg.fuse_qkv}")
print(f"compute dtype: {compute_dtype.name} | cache dtype: {cache_dtype.name}")
print(f"T(seq_len)={cfg.seq_len} | W(attn_window)={cfg.attn_window}")
print(f"use_flash_attn={cfg.use_flash_attn} | qblk={cfg.flash_q_block} | kblk={cfg.flash_k_block}")
print(f"flash_parallel_iterations={cfg.flash_parallel_iterations} | swap_memory={cfg.flash_swap_memory}")
print(f"flash_recompute_grad={cfg.flash_recompute_grad} | jit_compile={cfg.jit_compile}")
print(f"grad_accum_steps={cfg.grad_accum_steps} | batch_size={cfg.batch_size} | effective_batch={cfg.batch_size * cfg.grad_accum_steps}")


# ----------------------------
# True BOS-aligned packing: BOS name BOS name BOS ...
# Every training block starts at a BOS boundary.
# Wrap-extend so slicing never runs off the end.
# ----------------------------
def build_stream_and_doc_starts(docs_list):
    tokens = [BOS]
    doc_starts = [0]
    for d in docs_list:
        for ch in d:
            tokens.append(stoi[ch])
        tokens.append(BOS)
        doc_starts.append(len(tokens) - 1)
    return np.array(tokens, dtype=np.int32), np.array(doc_starts, dtype=np.int32)

stream, doc_starts = build_stream_and_doc_starts(docs)
block_len = cfg.seq_len + 1
stream_ext = np.concatenate([stream, stream[:block_len]], axis=0)

print(f"stream tokens (orig): {len(stream):,} | (extended): {len(stream_ext):,}")
print(f"doc_starts: {len(doc_starts):,} | block_len: {block_len}")

stream_tf = tf.constant(stream_ext, dtype=tf.int32)
doc_starts_tf = tf.constant(doc_starts, dtype=tf.int32)

def make_aligned_packed_dataset():
    ds = tf.data.Dataset.from_tensor_slices(doc_starts_tf)
    ds = ds.shuffle(buffer_size=min(len(doc_starts), 200_000), seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat()

    def map_start(s):
        s = tf.cast(s, tf.int32)
        seg = tf.slice(stream_tf, [s], [block_len])  # (T+1,)
        x = seg[:-1]                                 # (T,)
        y = seg[1:]                                  # (T,)
        return x, y

    ds = ds.map(map_start, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(cfg.batch_size, drop_remainder=True)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

ds = make_aligned_packed_dataset()
it = iter(ds)


# ----------------------------
# RoPE helpers (explicit rotation)
# ----------------------------
def make_inv_freq(head_dim, base=10000.0, dtype=tf.float32):
    assert head_dim % 2 == 0
    i = tf.range(0, head_dim, 2, dtype=dtype)
    return 1.0 / (base ** (i / tf.cast(head_dim, dtype)))

def rope_cos_sin(positions, inv_freq, scale_factor=1.0):
    pos = tf.cast(positions, inv_freq.dtype)
    if scale_factor != 1.0:
        pos = pos / tf.cast(scale_factor, inv_freq.dtype)
    freqs = tf.einsum("t,d->td", pos, inv_freq)  # (T, D/2)
    return tf.cos(freqs), tf.sin(freqs)

def apply_rope(x, cos, sin, head_dim):
    x_even = x[..., 0::2]
    x_odd  = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd  = x_even * sin + x_odd * cos
    out = tf.stack([out_even, out_odd], axis=-1)  # (..., D/2, 2)
    out_shape = tf.concat([tf.shape(x)[:-1], [head_dim]], axis=0)
    return tf.reshape(out, out_shape)

def cast_like(w, x):
    return tf.cast(w, x.dtype) if w.dtype != x.dtype else w


# ----------------------------
# FlashAttention core (GQA-aware) with optional ALiBi bias
# Uses tf.while_loop + online softmax.
# Sliding-window mask matches inference cache window:
#   valid iff k <= q AND k >= q-(W-1)
# ----------------------------
def flash_attn_causal_gqa_while_core(
    q, k, v,
    q_pos_table, k_pos_table,
    k_start_table, k_end_table,
    scale, W, q_block, k_block,
    alibi_slopes_hg=None,                 # (HKV,G) float32 or None
    parallel_iters=1, swap_memory=False
):
    """
    q: (B,HKV,G,T,D) compute dtype
    k: (B,HKV,T,D)   compute dtype
    v: (B,HKV,T,D)   compute dtype

    q_pos_table: (n_q,q_block) int32
    k_pos_table: (n_k,k_block) int32
    k_start_table/k_end_table: (n_q,) int32

    alibi_slopes_hg: (HKV,G) float32 (ALiBi slopes per query head) or None
    """
    q_dtype = q.dtype
    NEG = tf.constant(-1e9, tf.float32)
    eps = tf.constant(1e-9, tf.float32)

    B   = tf.shape(q)[0]
    HKV = tf.shape(q)[1]
    G   = tf.shape(q)[2]
    T   = tf.shape(q)[3]
    D   = tf.shape(q)[4]

    n_q = tf.shape(q_pos_table)[0]
    n_k = tf.shape(k_pos_table)[0]
    W_i = tf.cast(W, tf.int32)

    # reshape once so blocks are gatherable with tensor indices
    # q6: (B,HKV,G,n_q,q_block,D)
    q6 = tf.reshape(q, [B, HKV, G, n_q, q_block, D])
    # k5/v5: (B,HKV,n_k,k_block,D)
    k5 = tf.reshape(k, [B, HKV, n_k, k_block, D])
    v5 = tf.reshape(v, [B, HKV, n_k, k_block, D])

    if alibi_slopes_hg is not None:
        slopes = tf.convert_to_tensor(alibi_slopes_hg, dtype=tf.float32)  # (HKV,G)
    else:
        slopes = None

    out_ta = tf.TensorArray(
        dtype=tf.float32,
        size=n_q,
        element_shape=tf.TensorShape([None, None, None, q_block, None]),
        clear_after_read=False
    )

    def q_cond(qi, out_ta_):
        return qi < n_q

    def q_body(qi, out_ta_):
        q_abs = tf.gather(q_pos_table, qi)  # (q_block,)
        q_blk = tf.gather(q6, qi, axis=3)   # (B,HKV,G,q_block,D)

        lo = q_abs - (W_i - 1)              # (q_block,)

        # online softmax state (fp32)
        m = tf.fill([B, HKV, G, q_block], NEG)
        l = tf.zeros([B, HKV, G, q_block], tf.float32)
        o = tf.zeros([B, HKV, G, q_block, D], tf.float32)

        k_start = tf.gather(k_start_table, qi)
        k_end   = tf.gather(k_end_table, qi)

        def k_cond(ki, m_, l_, o_):
            return ki < k_end

        def k_body(ki, m_, l_, o_):
            k_abs = tf.gather(k_pos_table, ki)     # (k_block,)
            # cheap prune: if entire k-block is older than the earliest low bound, skip
            if_skip = tf.less(k_abs[-1], lo[0])

            def skip():
                return ki + 1, m_, l_, o_

            def do_block():
                k_blk = tf.gather(k5, ki, axis=2)  # (B,HKV,k_block,D)
                v_blk = tf.gather(v5, ki, axis=2)  # (B,HKV,k_block,D)

                # scores in compute dtype -> fp32
                scores = tf.einsum("bkgqd,bkmd->bkgqm", q_blk, k_blk) * tf.cast(scale, q_dtype)
                scores_f = tf.cast(scores, tf.float32)  # (B,HKV,G,q_block,k_block)

                # mask: causal + local window, using absolute positions
                mask = (k_abs[None, :] <= q_abs[:, None]) & (k_abs[None, :] >= lo[:, None])  # (q_block,k_block)
                mask_b = mask[None, None, None, :, :]  # (1,1,1,q_block,k_block)

                # ALiBi: add bias = -slope * (q-k)
                if slopes is not None:
                    dist = tf.cast(q_abs[:, None] - k_abs[None, :], tf.float32)  # (q_block,k_block)
                    bias = -slopes[None, :, :, None, None] * dist[None, None, None, :, :]  # (1,HKV,G,q,k)
                    scores_f = scores_f + bias

                # apply mask last
                scores_f = tf.where(mask_b, scores_f, tf.fill(tf.shape(scores_f), NEG))

                block_max = tf.reduce_max(scores_f, axis=-1)      # (B,HKV,G,q_block)
                m_new = tf.maximum(m_, block_max)

                exp_m = tf.exp(m_ - m_new)                        # (B,HKV,G,q_block)
                exp_b = tf.exp(block_max - m_new)

                exp_scores = tf.exp(scores_f - block_max[..., None])  # (B,HKV,G,q,k)
                exp_scores = tf.where(mask_b, exp_scores, tf.zeros_like(exp_scores))

                block_sum = tf.reduce_sum(exp_scores, axis=-1)    # (B,HKV,G,q_block)

                v_f = tf.cast(v_blk, tf.float32)
                block_out = tf.einsum("bkgqm,bkmd->bkgqd", exp_scores, v_f)  # (B,HKV,G,q,D)

                l_new = exp_m * l_ + exp_b * block_sum
                o_new = exp_m[..., None] * o_ + exp_b[..., None] * block_out
                return ki + 1, m_new, l_new, o_new

            return tf.cond(if_skip, skip, do_block)

        ki0 = tf.cast(k_start, tf.int32)
        _, m, l, o = tf.while_loop(
            k_cond, k_body,
            loop_vars=[ki0, m, l, o],
            shape_invariants=[
                ki0.shape,
                tf.TensorShape([None, None, None, q_block]),
                tf.TensorShape([None, None, None, q_block]),
                tf.TensorShape([None, None, None, q_block, None]),
            ],
            parallel_iterations=parallel_iters,
            swap_memory=swap_memory
        )

        out_blk = o / (l[..., None] + eps)  # (B,HKV,G,q_block,D)
        out_ta_ = out_ta_.write(qi, out_blk)
        return qi + 1, out_ta_

    qi0 = tf.constant(0, tf.int32)
    _, out_ta = tf.while_loop(
        q_cond, q_body,
        loop_vars=[qi0, out_ta],
        parallel_iterations=parallel_iters,
        swap_memory=swap_memory
    )

    out_stack = out_ta.stack()  # (n_q,B,HKV,G,q_block,D)
    out = tf.transpose(out_stack, [1, 2, 3, 0, 4, 5])  # (B,HKV,G,n_q,q_block,D)
    out = tf.reshape(out, [B, HKV, G, T, D])           # (B,HKV,G,T,D)
    return tf.cast(out, q_dtype)


def flash_attn_causal_gqa(
    q, k, v,
    q_pos_table, k_pos_table,
    k_start_table, k_end_table,
    scale, W, q_block, k_block,
    alibi_slopes_hg=None,
    parallel_iters=1, swap_memory=False,
    recompute_grad=False
):
    """
    Wrapper that optionally recomputes flash forward during backward (checkpointing).
    """
    if not recompute_grad:
        return flash_attn_causal_gqa_while_core(
            q, k, v,
            q_pos_table, k_pos_table,
            k_start_table, k_end_table,
            scale, W, q_block, k_block,
            alibi_slopes_hg=alibi_slopes_hg,
            parallel_iters=parallel_iters,
            swap_memory=swap_memory
        )

    @tf.custom_gradient
    def _flash(q_in, k_in, v_in):
        out = flash_attn_causal_gqa_while_core(
            q_in, k_in, v_in,
            q_pos_table, k_pos_table,
            k_start_table, k_end_table,
            scale, W, q_block, k_block,
            alibi_slopes_hg=alibi_slopes_hg,
            parallel_iters=parallel_iters,
            swap_memory=swap_memory
        )

        def grad(dout):
            with tf.GradientTape() as tape:
                tape.watch([q_in, k_in, v_in])
                out2 = flash_attn_causal_gqa_while_core(
                    q_in, k_in, v_in,
                    q_pos_table, k_pos_table,
                    k_start_table, k_end_table,
                    scale, W, q_block, k_block,
                    alibi_slopes_hg=alibi_slopes_hg,
                    parallel_iters=parallel_iters,
                    swap_memory=swap_memory
                )
            dq, dk, dv = tape.gradient(out2, [q_in, k_in, v_in], output_gradients=tf.cast(dout, out2.dtype))
            return dq, dk, dv

        return out, grad

    return _flash(q, k, v)


# ----------------------------
# Model components (explicit math)
# ----------------------------
INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.08)


class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, eps=1e-5, use_scale=True, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.use_scale = use_scale
        self.scale = None

    def build(self, input_shape):
        d = int(input_shape[-1])
        if self.use_scale:
            self.scale = self.add_weight(name="scale", shape=(d,), initializer="ones", trainable=True)

    def call(self, x):
        ms = tf.reduce_mean(tf.cast(tf.square(x), tf.float32), axis=-1, keepdims=True)
        inv = tf.cast(tf.math.rsqrt(ms + self.eps), x.dtype)
        x = x * inv
        if self.scale is not None:
            x = x * tf.cast(self.scale, x.dtype)
        return x


class MLP(tf.keras.layers.Layer):
    def __init__(self, cfg: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.hidden = cfg.mlp_mult * cfg.n_embd

    def build(self, input_shape):
        C = self.cfg.n_embd
        H = self.hidden
        if self.cfg.use_swiglu:
            self.w1 = self.add_weight(name="w1", shape=(C, H), initializer=INIT, trainable=True)
            self.w2 = self.add_weight(name="w2", shape=(C, H), initializer=INIT, trainable=True)
            self.w3 = self.add_weight(name="w3", shape=(H, C), initializer=INIT, trainable=True)
        else:
            self.w1 = self.add_weight(name="w1", shape=(C, H), initializer=INIT, trainable=True)
            self.w2 = self.add_weight(name="w2", shape=(H, C), initializer=INIT, trainable=True)

    def call(self, x, training=False):
        if self.cfg.use_swiglu:
            a = tf.einsum("btc,cd->btd", x, cast_like(self.w1, x))
            b = tf.einsum("btc,cd->btd", x, cast_like(self.w2, x))
            x = tf.nn.silu(a) * b
            x = tf.einsum("btc,cd->btd", x, cast_like(self.w3, x))
        else:
            x = tf.einsum("btc,cd->btd", x, cast_like(self.w1, x))
            x = tf.nn.relu(x)
            x = tf.einsum("btc,cd->btd", x, cast_like(self.w2, x))

        if training and self.cfg.dropout > 0.0:
            x = tf.nn.dropout(x, rate=self.cfg.dropout)
        return x

    def step(self, x_t):
        if self.cfg.use_swiglu:
            a = tf.einsum("bc,cd->bd", x_t, cast_like(self.w1, x_t))
            b = tf.einsum("bc,cd->bd", x_t, cast_like(self.w2, x_t))
            x = tf.nn.silu(a) * b
            x = tf.einsum("bc,cd->bd", x, cast_like(self.w3, x_t))
        else:
            x = tf.einsum("bc,cd->bd", x_t, cast_like(self.w1, x_t))
            x = tf.nn.relu(x)
            x = tf.einsum("bc,cd->bd", x, cast_like(self.w2, x_t))
        return x


class CausalSelfAttention(tf.keras.layers.Layer):
    """
    Training:
      - Fused QKV projection
      - GQA (HKV KV heads, H query heads)
      - RoPE OR ALiBi
      - FlashAttention online softmax with causal+local window mask (sliding window W)
    Inference:
      - ring KV cache length W (attn_window)
      - for ALiBi, also store a ring position cache to compute distance bias
    """
    def __init__(self, cfg: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.head_dim = cfg.n_embd // cfg.n_head
        self.groups = cfg.n_head // cfg.n_kv_head
        self.scale = self.head_dim ** -0.5
        self.inv_freq = make_inv_freq(self.head_dim, base=cfg.rope_base, dtype=tf.float32)

        # caches / tables
        self.cos_cache = None
        self.sin_cache = None
        self.q_pos_table = None
        self.k_pos_table = None
        self.k_start_table = None
        self.k_end_table = None

        # ALiBi slopes (HKV,G) if enabled
        self.alibi_slopes_hg = None  # float32

    def build(self, input_shape):
        cfg = self.cfg
        C = cfg.n_embd
        D = self.head_dim
        HKV = cfg.n_kv_head

        if cfg.fuse_qkv:
            out_dim = C + 2 * HKV * D
            self.wqkv = self.add_weight(name="wqkv", shape=(C, out_dim), initializer=INIT, trainable=True)
        else:
            self.wq = self.add_weight(name="wq", shape=(C, C), initializer=INIT, trainable=True)
            self.wk = self.add_weight(name="wk", shape=(C, HKV * D), initializer=INIT, trainable=True)
            self.wv = self.add_weight(name="wv", shape=(C, HKV * D), initializer=INIT, trainable=True)

        self.wo = self.add_weight(name="wo", shape=(C, C), initializer=INIT, trainable=True)

        # RoPE cache for training positions
        if cfg.pos_encoding == "rope":
            pos = tf.range(cfg.seq_len, dtype=tf.int32)
            cos, sin = rope_cos_sin(pos, self.inv_freq, scale_factor=cfg.rope_scale_factor)
            self.cos_cache = tf.constant(cos, dtype=tf.float32)
            self.sin_cache = tf.constant(sin, dtype=tf.float32)

        # ALiBi slopes per query head
        if cfg.pos_encoding == "alibi":
            slopes_h = alibi_slopes(cfg.n_head)  # (H,)
            slopes_hg = slopes_h.reshape(cfg.n_kv_head, self.groups)  # (HKV,G)
            self.alibi_slopes_hg = tf.constant(slopes_hg, dtype=tf.float32)

        # flash tables
        if cfg.use_flash_attn:
            T = cfg.seq_len
            qb = cfg.flash_q_block
            kb = cfg.flash_k_block
            n_q = T // qb
            n_k = T // kb

            self.q_pos_table = tf.reshape(tf.range(T, dtype=tf.int32), [n_q, qb])
            self.k_pos_table = tf.reshape(tf.range(T, dtype=tf.int32), [n_k, kb])

            W = cfg.attn_window
            k_start = []
            k_end = []
            for qi in range(n_q):
                q0 = qi * qb
                q1 = q0 + qb
                ke = min(n_k, (q1 + kb - 1) // kb)         # causal prune
                ks_pos = max(0, q0 - (W - 1))              # local prune
                ks = ks_pos // kb
                k_start.append(ks)
                k_end.append(ke)
            self.k_start_table = tf.constant(k_start, dtype=tf.int32)
            self.k_end_table = tf.constant(k_end, dtype=tf.int32)

    def _project_qkv_train(self, x):
        cfg = self.cfg
        B = tf.shape(x)[0]
        T = cfg.seq_len
        C = cfg.n_embd
        HKV = cfg.n_kv_head
        D = self.head_dim
        H = cfg.n_head

        if cfg.fuse_qkv:
            qkv = tf.einsum("btc,cd->btd", x, cast_like(self.wqkv, x))
            q = qkv[:, :, :C]                                  # (B,T,C)
            k = qkv[:, :, C:C + HKV * D]                       # (B,T,HKV*D)
            v = qkv[:, :, C + HKV * D:]                        # (B,T,HKV*D)
        else:
            q = tf.einsum("btc,cd->btd", x, cast_like(self.wq, x))
            k = tf.einsum("btc,cd->btd", x, cast_like(self.wk, x))
            v = tf.einsum("btc,cd->btd", x, cast_like(self.wv, x))

        # reshape
        q = tf.reshape(q, [B, T, H, D])
        q = tf.transpose(q, [0, 2, 1, 3])                       # (B,H,T,D)
        q = tf.reshape(q, [B, HKV, self.groups, T, D])          # (B,HKV,G,T,D)

        k = tf.reshape(k, [B, T, HKV, D])
        k = tf.transpose(k, [0, 2, 1, 3])                       # (B,HKV,T,D)

        v = tf.reshape(v, [B, T, HKV, D])
        v = tf.transpose(v, [0, 2, 1, 3])                       # (B,HKV,T,D)

        return q, k, v

    def call(self, x, training=False):
        cfg = self.cfg
        B = tf.shape(x)[0]
        T = cfg.seq_len
        C = cfg.n_embd
        H = cfg.n_head
        D = self.head_dim

        q, k, v = self._project_qkv_train(x)

        # positional encoding
        if cfg.pos_encoding == "rope":
            cos = tf.cast(self.cos_cache, q.dtype)  # (T,D/2)
            sin = tf.cast(self.sin_cache, q.dtype)
            q = apply_rope(q, cos[None, None, None, :, :], sin[None, None, None, :, :], D)
            k = apply_rope(k, cos[None, None, :, :],       sin[None, None, :, :],       D)

        # flash attention (or naive)
        if cfg.use_flash_attn:
            out = flash_attn_causal_gqa(
                q, k, v,
                self.q_pos_table, self.k_pos_table,
                self.k_start_table, self.k_end_table,
                scale=self.scale,
                W=cfg.attn_window,
                q_block=cfg.flash_q_block,
                k_block=cfg.flash_k_block,
                alibi_slopes_hg=(self.alibi_slopes_hg if cfg.pos_encoding == "alibi" else None),
                parallel_iters=cfg.flash_parallel_iterations,
                swap_memory=cfg.flash_swap_memory,
                recompute_grad=cfg.flash_recompute_grad
            )  # (B,HKV,G,T,D)
        else:
            # naive (debug only): materializes (T,T)
            # scores: (B,HKV,G,T,T)
            scores = tf.einsum("bkgtd,bkmd->bkgtm", q, k) * tf.cast(self.scale, q.dtype)

            qp = tf.range(T, dtype=tf.int32)[:, None]
            kp = tf.range(T, dtype=tf.int32)[None, :]
            mask = (kp <= qp) & (kp >= (qp - (cfg.attn_window - 1)))
            mask = mask[None, None, None, :, :]

            sf = tf.cast(scores, tf.float32)
            if cfg.pos_encoding == "alibi":
                slopes = self.alibi_slopes_hg  # (HKV,G)
                dist = tf.cast(qp - kp, tf.float32)            # (T,T)
                bias = -slopes[None, :, :, None, None] * dist[None, None, None, :, :]
                sf = sf + bias

            sf = tf.where(mask, sf, tf.fill(tf.shape(sf), tf.constant(-1e9, tf.float32)))
            w = tf.nn.softmax(sf, axis=-1)
            w = tf.cast(w, v.dtype)
            out = tf.einsum("bkgtm,bkmd->bkgtd", w, v)

        # merge heads: (B,HKV,G,T,D)->(B,H,T,D)->(B,T,C)
        out = tf.reshape(out, [B, H, T, D])
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [B, T, C])

        out = tf.einsum("btc,cd->btd", out, cast_like(self.wo, out))
        if training and cfg.dropout > 0.0:
            out = tf.nn.dropout(out, rate=cfg.dropout)
        return out

    def step_ring(self, x_t, cache_entry, t):
        """
        Inference step with ring cache of length W=attn_window.

        cache_entry:
          - if RoPE:  (kflat, vflat)
          - if ALiBi: (kflat, vflat, pflat) where pflat stores absolute positions per slot

        Shapes:
          kflat,vflat: (B*HKV, W, D)
          pflat:       (B*HKV, W) int32 (ALiBi only), init to -1
        """
        cfg = self.cfg
        B = tf.shape(x_t)[0]
        C = cfg.n_embd
        H = cfg.n_head
        HKV = cfg.n_kv_head
        G = self.groups
        D = self.head_dim
        W = cfg.attn_window

        t = tf.cast(t, tf.int32)
        slot = tf.math.floormod(t, tf.cast(W, tf.int32))

        if cfg.pos_encoding == "alibi":
            kflat, vflat, pflat = cache_entry
        else:
            kflat, vflat = cache_entry
            pflat = None

        # fused projection: (B,C)->(B, C+2*HKV*D)
        if cfg.fuse_qkv:
            out_dim = C + 2 * HKV * D
            qkv = tf.einsum("bc,cd->bd", x_t, cast_like(self.wqkv, x_t))
            q = qkv[:, :C]                                  # (B,C)
            k = qkv[:, C:C + HKV * D]                       # (B,HKV*D)
            v = qkv[:, C + HKV * D:]                        # (B,HKV*D)
        else:
            q = tf.einsum("bc,cd->bd", x_t, cast_like(self.wq, x_t))
            k = tf.einsum("bc,cd->bd", x_t, cast_like(self.wk, x_t))
            v = tf.einsum("bc,cd->bd", x_t, cast_like(self.wv, x_t))

        q = tf.reshape(q, [B, H, D])                         # (B,H,D)
        q = tf.reshape(q, [B, HKV, G, D])                    # (B,HKV,G,D)
        k = tf.reshape(k, [B, HKV, D])                       # (B,HKV,D)
        v = tf.reshape(v, [B, HKV, D])                       # (B,HKV,D)

        # positional encoding for inference
        if cfg.pos_encoding == "rope":
            cos, sin = rope_cos_sin(tf.reshape(t, [1]), self.inv_freq, scale_factor=cfg.rope_scale_factor)
            cos = tf.cast(cos, q.dtype)
            sin = tf.cast(sin, q.dtype)
            q = apply_rope(q, cos[None, None, None, :], sin[None, None, None, :], D)
            k = apply_rope(k, cos[None, None, :],       sin[None, None, :],       D)

        # write KV into ring cache
        k_now = tf.reshape(k, [B * HKV, D])
        v_now = tf.reshape(v, [B * HKV, D])

        if k_now.dtype != kflat.dtype:
            k_now = tf.cast(k_now, kflat.dtype)
            v_now = tf.cast(v_now, vflat.dtype)

        BK = tf.shape(k_now)[0]
        idx = tf.stack([tf.range(BK, dtype=tf.int32), tf.fill([BK], slot)], axis=1)

        if isinstance(kflat, tf.Variable):
            kflat.scatter_nd_update(idx, k_now)
            vflat.scatter_nd_update(idx, v_now)
            Kflat, Vflat = kflat, vflat
        else:
            Kflat = tf.tensor_scatter_nd_update(kflat, idx, k_now)
            Vflat = tf.tensor_scatter_nd_update(vflat, idx, v_now)

        # ALiBi: update pos ring cache
        if cfg.pos_encoding == "alibi":
            tvals = tf.fill([BK], t)
            if isinstance(pflat, tf.Variable):
                pflat.scatter_nd_update(idx, tvals)
                Pflat = pflat
            else:
                Pflat = tf.tensor_scatter_nd_update(pflat, idx, tvals)
        else:
            Pflat = None

        # compute attention over ring cache
        K = tf.reshape(Kflat, [B, HKV, W, D])
        V = tf.reshape(Vflat, [B, HKV, W, D])

        scores = tf.einsum("bkgd,bkWd->bkgW", q, tf.cast(K, q.dtype)) * tf.cast(self.scale, q.dtype)  # (B,HKV,G,W)
        sf = tf.cast(scores, tf.float32)

        # mask valid slots:
        if cfg.pos_encoding == "alibi":
            P = tf.reshape(Pflat, [B, HKV, W])  # int32
            valid = P >= 0
            dist = tf.cast(t - P, tf.float32)   # (B,HKV,W)
            slopes = self.alibi_slopes_hg       # (HKV,G)
            bias = -slopes[None, :, :, None] * dist[:, :, None, :]  # (B,HKV,G,W)
            sf = sf + bias
            valid4 = valid[:, :, None, :]  # (B,HKV,1,W)
        else:
            filled = tf.minimum(t + 1, tf.cast(W, tf.int32))
            valid4 = (tf.range(W, dtype=tf.int32)[None, None, None, :] < filled)

        sf = tf.where(valid4, sf, tf.fill(tf.shape(sf), tf.constant(-1e9, tf.float32)))
        w = tf.nn.softmax(sf, axis=-1)
        w = tf.cast(w, tf.cast(V, compute_dtype).dtype)

        out = tf.einsum("bkgW,bkWd->bkgd", w, tf.cast(V, w.dtype))  # (B,HKV,G,D)
        out = tf.reshape(out, [B, H, D])
        out = tf.reshape(out, [B, C])
        out = tf.einsum("bc,cd->bd", out, cast_like(self.wo, out))

        if cfg.pos_encoding == "alibi":
            return out, (Kflat, Vflat, Pflat)
        else:
            return out, (Kflat, Vflat)


class Block(tf.keras.layers.Layer):
    def __init__(self, cfg: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.ln1 = RMSNorm(cfg.rmsnorm_eps, cfg.rmsnorm_scale)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.rmsnorm_eps, cfg.rmsnorm_scale)
        self.mlp = MLP(cfg)

    def call(self, x, training=False):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x

    def step_ring(self, x_t, cache_entry, t):
        a, cache_entry = self.attn.step_ring(self.ln1(x_t), cache_entry, t)
        a = tf.cast(a, x_t.dtype)
        x_t = x_t + a
        m = self.mlp.step(self.ln2(x_t))
        x_t = x_t + tf.cast(m, x_t.dtype)
        return x_t, cache_entry


class ExplicitGPT(tf.keras.Model):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.head_dim = cfg.n_embd // cfg.n_head

        self.wte = tf.keras.layers.Embedding(cfg.vocab_size, cfg.n_embd, embeddings_initializer=INIT, name="wte")
        self.emb_norm = RMSNorm(cfg.rmsnorm_eps, cfg.rmsnorm_scale)
        self.blocks = [Block(cfg, name=f"block{i}") for i in range(cfg.n_layer)]
        self.final_norm = RMSNorm(cfg.rmsnorm_eps, cfg.rmsnorm_scale)

        if not cfg.tie_embeddings:
            self.lm_head = self.add_weight(name="lm_head", shape=(cfg.vocab_size, cfg.n_embd),
                                           initializer=INIT, trainable=True)
        else:
            self.lm_head = None

    def _lm_weight(self):
        return self.wte.embeddings if self.cfg.tie_embeddings else self.lm_head

    def call(self, idx, training=False):
        x = self.wte(idx)                 # fp32
        x = tf.cast(x, compute_dtype)     # compute dtype
        x = self.emb_norm(x)

        for blk in self.blocks:
            x = blk(x, training=training)

        x = self.final_norm(x)
        W = self._lm_weight()
        logits = tf.einsum("btc,vc->btv", x, tf.cast(W, x.dtype))
        return logits

    def init_kv_cache(self, batch_size, as_variables=False):
        """
        Per layer cache:
          - RoPE:  (Kflat, Vflat)
          - ALiBi: (Kflat, Vflat, Pflat)
        """
        HKV = self.cfg.n_kv_head
        W = self.cfg.attn_window
        D = self.head_dim
        BK = batch_size * HKV

        zeros_kv = tf.zeros([BK, W, D], dtype=cache_dtype)

        cache = []
        for _ in range(self.cfg.n_layer):
            if as_variables:
                k = tf.Variable(zeros_kv, trainable=False)
                v = tf.Variable(zeros_kv, trainable=False)
            else:
                k = zeros_kv
                v = zeros_kv

            if self.cfg.pos_encoding == "alibi":
                init_p = tf.fill([BK, W], tf.constant(-1, tf.int32))
                if as_variables:
                    p = tf.Variable(init_p, trainable=False)
                else:
                    p = init_p
                cache.append((k, v, p))
            else:
                cache.append((k, v))
        return cache

    def reset_kv_cache(self, cache):
        for entry in cache:
            if self.cfg.pos_encoding == "alibi":
                k, v, p = entry
                if isinstance(k, tf.Variable): k.assign(tf.zeros_like(k))
                if isinstance(v, tf.Variable): v.assign(tf.zeros_like(v))
                if isinstance(p, tf.Variable): p.assign(tf.fill(tf.shape(p), tf.constant(-1, tf.int32)))
            else:
                k, v = entry
                if isinstance(k, tf.Variable): k.assign(tf.zeros_like(k))
                if isinstance(v, tf.Variable): v.assign(tf.zeros_like(v))

    def step_ring(self, token_id, t, cache):
        x_t = self.wte(token_id)
        x_t = tf.cast(x_t, compute_dtype)
        x_t = self.emb_norm(x_t)

        new_cache = []
        for li, blk in enumerate(self.blocks):
            x_t, entry = blk.step_ring(x_t, cache[li], t)
            new_cache.append(entry)

        x_t = self.final_norm(x_t)
        W = self._lm_weight()
        logits = tf.einsum("bc,vc->bv", x_t, tf.cast(W, x_t.dtype))
        return logits, new_cache


# ----------------------------
# LR schedule: warmup + cosine
# ----------------------------
def lr_schedule(update_step_int64):
    step = tf.cast(update_step_int64, tf.float32)
    warm = tf.cast(cfg.warmup_steps, tf.float32)
    total = tf.cast(cfg.num_updates, tf.float32)

    warm_lr = cfg.base_lr * (step / tf.maximum(1.0, warm))
    progress = (step - warm) / tf.maximum(1.0, total - warm)
    progress = tf.clip_by_value(progress, 0.0, 1.0)
    cosine = 0.5 * (1.0 + tf.cos(math.pi * progress))
    cos_lr = cfg.min_lr + (cfg.base_lr - cfg.min_lr) * cosine
    return tf.where(step < warm, warm_lr, cos_lr)


# ----------------------------
# Optimizer: Adam + explicit AdamW-style decay + grad clip
# ----------------------------
opt = tf.keras.optimizers.Adam(learning_rate=cfg.base_lr, beta_1=0.9, beta_2=0.95, epsilon=1e-8)

def apply_weight_decay(vars_, lr):
    wd = cfg.weight_decay
    if wd <= 0.0:
        return
    for v in vars_:
        r = v.shape.rank
        if r is not None and r >= 2:
            v.assign_sub(lr * wd * v)


# ----------------------------
# Build model + grad accum + checkpointing
# ----------------------------
model = ExplicitGPT(cfg)
_ = model(tf.zeros([cfg.batch_size, cfg.seq_len], dtype=tf.int32), training=False)

train_vars = model.trainable_variables
grad_accums = [tf.Variable(tf.zeros_like(v), trainable=False) for v in train_vars]
accum_count = tf.Variable(0, dtype=tf.int32, trainable=False)
update_step = tf.Variable(0, dtype=tf.int64, trainable=False)

ckpt = tf.train.Checkpoint(model=model, update_step=update_step)
manager = tf.train.CheckpointManager(ckpt, cfg.ckpt_dir, max_to_keep=3)

if manager.latest_checkpoint:
    ckpt.restore(manager.latest_checkpoint)
    print(f"restored checkpoint: {manager.latest_checkpoint} (update_step={int(update_step.numpy())})")
else:
    print("no checkpoint found, starting fresh")

param_count = sum(int(np.prod(v.shape)) for v in model.trainable_variables)
print(f"num params: {param_count:,}")


# ----------------------------
# One micro-step: accumulate grads; update every grad_accum_steps
# ----------------------------
@tf.function(jit_compile=cfg.jit_compile)
def train_micro_step(x, y):
    with tf.GradientTape() as tape:
        logits = model(x, training=True)          # compute dtype
        logits_f = tf.cast(logits, tf.float32)    # loss in fp32
        loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits_f)
        )

    grads = tape.gradient(loss, train_vars)

    for ga, g in zip(grad_accums, grads):
        if g is None:
            continue
        ga.assign_add(tf.where(tf.math.is_finite(g), g, tf.zeros_like(g)))

    accum_count.assign_add(1)
    did_update = tf.equal(accum_count, tf.cast(cfg.grad_accum_steps, tf.int32))

    def apply_update():
        lr = lr_schedule(update_step)
        opt.learning_rate.assign(lr)

        denom = tf.cast(cfg.grad_accum_steps, tf.float32)
        avg_grads_f32 = [tf.cast(ga, tf.float32) / denom for ga in grad_accums]
        avg_grads_f32, gnorm = tf.clip_by_global_norm(avg_grads_f32, cfg.grad_clip_norm)

        cast_grads = [tf.cast(g, v.dtype) for g, v in zip(avg_grads_f32, train_vars)]
        opt.apply_gradients(zip(cast_grads, train_vars))
        apply_weight_decay(train_vars, lr)

        for ga in grad_accums:
            ga.assign(tf.zeros_like(ga))
        accum_count.assign(0)

        update_step.assign_add(1)
        return gnorm, lr, tf.constant(True)

    def no_update():
        return tf.constant(0.0, tf.float32), tf.cast(opt.learning_rate, tf.float32), tf.constant(False)

    gnorm, lr, updated = tf.cond(did_update, apply_update, no_update)
    return loss, gnorm, lr, updated


# ----------------------------
# Sampling utilities
# ----------------------------
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

def generate_names(num=20, max_new_tokens=256, temperature=0.8, top_k=0, top_p=0.9):
    print("\n--- inference (new, hallucinated names) ---")
    cache = model.init_kv_cache(batch_size=1, as_variables=True)

    for i in range(num):
        model.reset_kv_cache(cache)
        token = BOS
        out = []

        for t in range(max_new_tokens):
            logits, cache = model.step_ring(tf.constant([token], tf.int32), t, cache)
            next_id = sample_from_logits(tf.cast(logits[0], tf.float32),
                                         temperature=temperature, top_k=top_k, top_p=top_p)
            if next_id == BOS:
                break
            out.append(itos[next_id])
            token = next_id

        print(f"sample {i+1:2d}: {''.join(out)}")


def generate_from_prompt(prompt, max_new_tokens=64, temperature=0.8, top_k=0, top_p=0.9):
    """
    Condition generation on a text prompt and return prompt+continuation.
    Unknown characters are dropped for this char-level tokenizer.
    """
    clean = "".join(ch for ch in prompt if ch in stoi)
    if not clean:
        return ""

    cache = model.init_kv_cache(batch_size=1, as_variables=True)
    model.reset_kv_cache(cache)

    token = BOS
    t = 0

    # Prime cache with BOS -> prompt chars.
    for ch in clean:
        _, cache = model.step_ring(tf.constant([token], tf.int32), t, cache)
        token = stoi[ch]
        t += 1

    # Feed last prompt token, then sample continuation.
    logits, cache = model.step_ring(tf.constant([token], tf.int32), t, cache)
    t += 1

    out = []
    for _ in range(max_new_tokens):
        next_id = sample_from_logits(tf.cast(logits[0], tf.float32),
                                     temperature=temperature, top_k=top_k, top_p=top_p)
        if next_id == BOS:
            break
        out.append(itos[next_id])
        logits, cache = model.step_ring(tf.constant([next_id], tf.int32), t, cache)
        t += 1

    return clean + "".join(out)


# ----------------------------
# Training loop
# ----------------------------
print("\n--- training ---")
micro = 0
t0 = time.time()
last_log_time = t0
last_log_updates = int(update_step.numpy())

while int(update_step.numpy()) < cfg.num_updates:
    x, y = next(it)
    loss, gnorm, lr, updated = train_micro_step(x, y)
    micro += 1

    upd = int(update_step.numpy())
    if upd == 0 and micro == 1:
        print(f"micro {micro:6d} | update {upd:5d}/{cfg.num_updates} | loss {loss.numpy():.4f}")

    if (upd > 0) and bool(updated.numpy()) and (upd % cfg.log_every == 0):
        now = time.time()
        dt = now - last_log_time
        du = upd - last_log_updates

        tok_per_update = cfg.batch_size * cfg.seq_len * cfg.grad_accum_steps
        toks = du * tok_per_update
        tps = toks / max(1e-9, dt)

        print(f"micro {micro:6d} | update {upd:5d}/{cfg.num_updates} | loss {loss.numpy():.4f} "
              f"| gnorm {gnorm.numpy():.3f} | lr {lr.numpy():.2e} | tok/s {tps:,.0f}")

        preview = generate_from_prompt("mar", max_new_tokens=24, temperature=0.7, top_p=0.9)
        if preview:
            print(f"prompt 'mar' -> {preview}")

        last_log_time = now
        last_log_updates = upd

    if cfg.save_every and bool(updated.numpy()) and (upd % cfg.save_every == 0):
        path = manager.save(checkpoint_number=upd)
        print(f"saved checkpoint: {path}")

path = manager.save(checkpoint_number=int(update_step.numpy()))
print(f"saved checkpoint: {path}")

generate_names(num=20, max_new_tokens=256, temperature=0.7, top_p=0.9)
