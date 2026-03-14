import math

import numpy as np
import tensorflow as tf

from teligence.config import GPTConfig


INIT = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.08)
COMPUTE_DTYPE = tf.float32
CACHE_DTYPE = tf.float32


def set_precision(use_bf16: bool):
    global COMPUTE_DTYPE, CACHE_DTYPE
    COMPUTE_DTYPE = tf.bfloat16 if use_bf16 else tf.float32
    CACHE_DTYPE = COMPUTE_DTYPE


def get_precision_dtypes():
    return COMPUTE_DTYPE, CACHE_DTYPE


def alibi_slopes(n_head: int) -> np.ndarray:
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


def make_inv_freq(head_dim, base=10000.0, dtype=tf.float32):
    assert head_dim % 2 == 0
    i = tf.range(0, head_dim, 2, dtype=dtype)
    return 1.0 / (base ** (i / tf.cast(head_dim, dtype)))


def rope_cos_sin(positions, inv_freq, scale_factor=1.0):
    pos = tf.cast(positions, inv_freq.dtype)
    if scale_factor != 1.0:
        pos = pos / tf.cast(scale_factor, inv_freq.dtype)
    freqs = tf.einsum("t,d->td", pos, inv_freq)
    return tf.cos(freqs), tf.sin(freqs)


def apply_rope(x, cos, sin, head_dim):
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos
    out = tf.stack([out_even, out_odd], axis=-1)
    out_shape = tf.concat([tf.shape(x)[:-1], [head_dim]], axis=0)
    return tf.reshape(out, out_shape)


def cast_like(w, x):
    return tf.cast(w, x.dtype) if w.dtype != x.dtype else w


def flash_attn_causal_gqa_while_core(
    q,
    k,
    v,
    q_pos_table,
    k_pos_table,
    k_start_table,
    k_end_table,
    scale,
    W,
    q_block,
    k_block,
    alibi_slopes_hg=None,
    parallel_iters=1,
    swap_memory=False,
):
    q_dtype = q.dtype
    neg = tf.constant(-1e9, tf.float32)
    eps = tf.constant(1e-9, tf.float32)

    b = tf.shape(q)[0]
    hkv = tf.shape(q)[1]
    g = tf.shape(q)[2]
    t = tf.shape(q)[3]
    d = tf.shape(q)[4]

    n_q = tf.shape(q_pos_table)[0]
    w_i = tf.cast(W, tf.int32)

    q6 = tf.reshape(q, [b, hkv, g, n_q, q_block, d])
    k5 = tf.reshape(k, [b, hkv, tf.shape(k_pos_table)[0], k_block, d])
    v5 = tf.reshape(v, [b, hkv, tf.shape(k_pos_table)[0], k_block, d])

    slopes = tf.convert_to_tensor(alibi_slopes_hg, dtype=tf.float32) if alibi_slopes_hg is not None else None

    out_ta = tf.TensorArray(
        dtype=tf.float32,
        size=n_q,
        element_shape=tf.TensorShape([None, None, None, q_block, None]),
        clear_after_read=False,
    )

    def q_cond(qi, out_ta_):
        return qi < n_q

    def q_body(qi, out_ta_):
        q_abs = tf.gather(q_pos_table, qi)
        q_blk = tf.gather(q6, qi, axis=3)
        lo = q_abs - (w_i - 1)

        m = tf.fill([b, hkv, g, q_block], neg)
        l = tf.zeros([b, hkv, g, q_block], tf.float32)
        o = tf.zeros([b, hkv, g, q_block, d], tf.float32)

        k_start = tf.gather(k_start_table, qi)
        k_end = tf.gather(k_end_table, qi)

        def k_cond(ki, m_, l_, o_):
            return ki < k_end

        def k_body(ki, m_, l_, o_):
            k_abs = tf.gather(k_pos_table, ki)
            if_skip = tf.less(k_abs[-1], lo[0])

            def skip():
                return ki + 1, m_, l_, o_

            def do_block():
                k_blk = tf.gather(k5, ki, axis=2)
                v_blk = tf.gather(v5, ki, axis=2)

                scores = tf.einsum("bkgqd,bkmd->bkgqm", q_blk, k_blk) * tf.cast(scale, q_dtype)
                scores_f = tf.cast(scores, tf.float32)

                mask = (k_abs[None, :] <= q_abs[:, None]) & (k_abs[None, :] >= lo[:, None])
                mask_b = mask[None, None, None, :, :]

                if slopes is not None:
                    dist = tf.cast(q_abs[:, None] - k_abs[None, :], tf.float32)
                    bias = -slopes[None, :, :, None, None] * dist[None, None, None, :, :]
                    scores_f = scores_f + bias

                scores_f = tf.where(mask_b, scores_f, tf.fill(tf.shape(scores_f), neg))
                block_max = tf.reduce_max(scores_f, axis=-1)
                m_new = tf.maximum(m_, block_max)

                exp_m = tf.exp(m_ - m_new)
                exp_b = tf.exp(block_max - m_new)

                exp_scores = tf.exp(scores_f - block_max[..., None])
                exp_scores = tf.where(mask_b, exp_scores, tf.zeros_like(exp_scores))
                block_sum = tf.reduce_sum(exp_scores, axis=-1)

                v_f = tf.cast(v_blk, tf.float32)
                block_out = tf.einsum("bkgqm,bkmd->bkgqd", exp_scores, v_f)

                l_new = exp_m * l_ + exp_b * block_sum
                o_new = exp_m[..., None] * o_ + exp_b[..., None] * block_out
                return ki + 1, m_new, l_new, o_new

            return tf.cond(if_skip, skip, do_block)

        ki0 = tf.cast(k_start, tf.int32)
        _, m, l, o = tf.while_loop(
            k_cond,
            k_body,
            loop_vars=[ki0, m, l, o],
            shape_invariants=[
                ki0.shape,
                tf.TensorShape([None, None, None, q_block]),
                tf.TensorShape([None, None, None, q_block]),
                tf.TensorShape([None, None, None, q_block, None]),
            ],
            parallel_iterations=parallel_iters,
            swap_memory=swap_memory,
        )

        out_blk = o / (l[..., None] + eps)
        out_ta_ = out_ta_.write(qi, out_blk)
        return qi + 1, out_ta_

    qi0 = tf.constant(0, tf.int32)
    _, out_ta = tf.while_loop(
        q_cond,
        q_body,
        loop_vars=[qi0, out_ta],
        parallel_iterations=parallel_iters,
        swap_memory=swap_memory,
    )

    out_stack = out_ta.stack()
    out = tf.transpose(out_stack, [1, 2, 3, 0, 4, 5])
    out = tf.reshape(out, [b, hkv, g, t, d])
    return tf.cast(out, q_dtype)


def flash_attn_causal_gqa(
    q,
    k,
    v,
    q_pos_table,
    k_pos_table,
    k_start_table,
    k_end_table,
    scale,
    W,
    q_block,
    k_block,
    alibi_slopes_hg=None,
    parallel_iters=1,
    swap_memory=False,
    recompute_grad=False,
):
    if not recompute_grad:
        return flash_attn_causal_gqa_while_core(
            q,
            k,
            v,
            q_pos_table,
            k_pos_table,
            k_start_table,
            k_end_table,
            scale,
            W,
            q_block,
            k_block,
            alibi_slopes_hg=alibi_slopes_hg,
            parallel_iters=parallel_iters,
            swap_memory=swap_memory,
        )

    @tf.custom_gradient
    def _flash(q_in, k_in, v_in):
        out = flash_attn_causal_gqa_while_core(
            q_in,
            k_in,
            v_in,
            q_pos_table,
            k_pos_table,
            k_start_table,
            k_end_table,
            scale,
            W,
            q_block,
            k_block,
            alibi_slopes_hg=alibi_slopes_hg,
            parallel_iters=parallel_iters,
            swap_memory=swap_memory,
        )

        def grad(dout):
            with tf.GradientTape() as tape:
                tape.watch([q_in, k_in, v_in])
                out2 = flash_attn_causal_gqa_while_core(
                    q_in,
                    k_in,
                    v_in,
                    q_pos_table,
                    k_pos_table,
                    k_start_table,
                    k_end_table,
                    scale,
                    W,
                    q_block,
                    k_block,
                    alibi_slopes_hg=alibi_slopes_hg,
                    parallel_iters=parallel_iters,
                    swap_memory=swap_memory,
                )
            dq, dk, dv = tape.gradient(out2, [q_in, k_in, v_in], output_gradients=tf.cast(dout, out2.dtype))
            return dq, dk, dv

        return out, grad

    return _flash(q, k, v)


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
        c = self.cfg.n_embd
        h = self.hidden
        if self.cfg.use_swiglu:
            self.w1 = self.add_weight(name="w1", shape=(c, h), initializer=INIT, trainable=True)
            self.w2 = self.add_weight(name="w2", shape=(c, h), initializer=INIT, trainable=True)
            self.w3 = self.add_weight(name="w3", shape=(h, c), initializer=INIT, trainable=True)
        else:
            self.w1 = self.add_weight(name="w1", shape=(c, h), initializer=INIT, trainable=True)
            self.w2 = self.add_weight(name="w2", shape=(h, c), initializer=INIT, trainable=True)

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
    def __init__(self, cfg: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.cfg = cfg
        self.head_dim = cfg.n_embd // cfg.n_head
        self.groups = cfg.n_head // cfg.n_kv_head
        self.scale = self.head_dim ** -0.5
        self.inv_freq = make_inv_freq(self.head_dim, base=cfg.rope_base, dtype=tf.float32)
        self.cos_cache = None
        self.sin_cache = None
        self.q_pos_table = None
        self.k_pos_table = None
        self.k_start_table = None
        self.k_end_table = None
        self.alibi_slopes_hg = None

    def build(self, input_shape):
        cfg = self.cfg
        c = cfg.n_embd
        d = self.head_dim
        hkv = cfg.n_kv_head

        if cfg.fuse_qkv:
            out_dim = c + 2 * hkv * d
            self.wqkv = self.add_weight(name="wqkv", shape=(c, out_dim), initializer=INIT, trainable=True)
        else:
            self.wq = self.add_weight(name="wq", shape=(c, c), initializer=INIT, trainable=True)
            self.wk = self.add_weight(name="wk", shape=(c, hkv * d), initializer=INIT, trainable=True)
            self.wv = self.add_weight(name="wv", shape=(c, hkv * d), initializer=INIT, trainable=True)

        self.wo = self.add_weight(name="wo", shape=(c, c), initializer=INIT, trainable=True)

        if cfg.pos_encoding == "rope":
            pos = tf.range(cfg.seq_len, dtype=tf.int32)
            cos, sin = rope_cos_sin(pos, self.inv_freq, scale_factor=cfg.rope_scale_factor)
            self.cos_cache = tf.constant(cos, dtype=tf.float32)
            self.sin_cache = tf.constant(sin, dtype=tf.float32)

        if cfg.pos_encoding == "alibi":
            slopes_h = alibi_slopes(cfg.n_head)
            slopes_hg = slopes_h.reshape(cfg.n_kv_head, self.groups)
            self.alibi_slopes_hg = tf.constant(slopes_hg, dtype=tf.float32)

        if cfg.use_flash_attn:
            t = cfg.seq_len
            qb = cfg.flash_q_block
            kb = cfg.flash_k_block
            n_q = t // qb
            n_k = t // kb

            self.q_pos_table = tf.reshape(tf.range(t, dtype=tf.int32), [n_q, qb])
            self.k_pos_table = tf.reshape(tf.range(t, dtype=tf.int32), [n_k, kb])

            w = cfg.attn_window
            k_start = []
            k_end = []
            for qi in range(n_q):
                q0 = qi * qb
                q1 = q0 + qb
                ke = min(n_k, (q1 + kb - 1) // kb)
                ks_pos = max(0, q0 - (w - 1))
                ks = ks_pos // kb
                k_start.append(ks)
                k_end.append(ke)
            self.k_start_table = tf.constant(k_start, dtype=tf.int32)
            self.k_end_table = tf.constant(k_end, dtype=tf.int32)

    def _project_qkv_train(self, x):
        cfg = self.cfg
        b = tf.shape(x)[0]
        t = cfg.seq_len
        c = cfg.n_embd
        hkv = cfg.n_kv_head
        d = self.head_dim
        h = cfg.n_head

        if cfg.fuse_qkv:
            qkv = tf.einsum("btc,cd->btd", x, cast_like(self.wqkv, x))
            q = qkv[:, :, :c]
            k = qkv[:, :, c : c + hkv * d]
            v = qkv[:, :, c + hkv * d :]
        else:
            q = tf.einsum("btc,cd->btd", x, cast_like(self.wq, x))
            k = tf.einsum("btc,cd->btd", x, cast_like(self.wk, x))
            v = tf.einsum("btc,cd->btd", x, cast_like(self.wv, x))

        q = tf.reshape(q, [b, t, h, d])
        q = tf.transpose(q, [0, 2, 1, 3])
        q = tf.reshape(q, [b, hkv, self.groups, t, d])

        k = tf.reshape(k, [b, t, hkv, d])
        k = tf.transpose(k, [0, 2, 1, 3])

        v = tf.reshape(v, [b, t, hkv, d])
        v = tf.transpose(v, [0, 2, 1, 3])
        return q, k, v

    def call(self, x, training=False):
        cfg = self.cfg
        b = tf.shape(x)[0]
        t = cfg.seq_len
        c = cfg.n_embd
        h = cfg.n_head
        d = self.head_dim

        q, k, v = self._project_qkv_train(x)

        if cfg.pos_encoding == "rope":
            cos = tf.cast(self.cos_cache, q.dtype)
            sin = tf.cast(self.sin_cache, q.dtype)
            q = apply_rope(q, cos[None, None, None, :, :], sin[None, None, None, :, :], d)
            k = apply_rope(k, cos[None, None, :, :], sin[None, None, :, :], d)

        if cfg.use_flash_attn:
            out = flash_attn_causal_gqa(
                q,
                k,
                v,
                self.q_pos_table,
                self.k_pos_table,
                self.k_start_table,
                self.k_end_table,
                scale=self.scale,
                W=cfg.attn_window,
                q_block=cfg.flash_q_block,
                k_block=cfg.flash_k_block,
                alibi_slopes_hg=(self.alibi_slopes_hg if cfg.pos_encoding == "alibi" else None),
                parallel_iters=cfg.flash_parallel_iterations,
                swap_memory=cfg.flash_swap_memory,
                recompute_grad=cfg.flash_recompute_grad,
            )
        else:
            scores = tf.einsum("bkgtd,bkmd->bkgtm", q, k) * tf.cast(self.scale, q.dtype)
            qp = tf.range(t, dtype=tf.int32)[:, None]
            kp = tf.range(t, dtype=tf.int32)[None, :]
            mask = (kp <= qp) & (kp >= (qp - (cfg.attn_window - 1)))
            mask = mask[None, None, None, :, :]
            sf = tf.cast(scores, tf.float32)
            if cfg.pos_encoding == "alibi":
                slopes = self.alibi_slopes_hg
                dist = tf.cast(qp - kp, tf.float32)
                bias = -slopes[None, :, :, None, None] * dist[None, None, None, :, :]
                sf = sf + bias
            sf = tf.where(mask, sf, tf.fill(tf.shape(sf), tf.constant(-1e9, tf.float32)))
            w = tf.nn.softmax(sf, axis=-1)
            w = tf.cast(w, v.dtype)
            out = tf.einsum("bkgtm,bkmd->bkgtd", w, v)

        out = tf.reshape(out, [b, h, t, d])
        out = tf.transpose(out, [0, 2, 1, 3])
        out = tf.reshape(out, [b, t, c])
        out = tf.einsum("btc,cd->btd", out, cast_like(self.wo, out))
        if training and cfg.dropout > 0.0:
            out = tf.nn.dropout(out, rate=cfg.dropout)
        return out

    def step_ring(self, x_t, cache_entry, t, return_attn=False):
        cfg = self.cfg
        b = tf.shape(x_t)[0]
        c = cfg.n_embd
        h = cfg.n_head
        hkv = cfg.n_kv_head
        g = self.groups
        d = self.head_dim
        w = cfg.attn_window

        t = tf.cast(t, tf.int32)
        slot = tf.math.floormod(t, tf.cast(w, tf.int32))

        if cfg.pos_encoding == "alibi":
            kflat, vflat, pflat = cache_entry
        else:
            kflat, vflat = cache_entry
            pflat = None

        if cfg.fuse_qkv:
            qkv = tf.einsum("bc,cd->bd", x_t, cast_like(self.wqkv, x_t))
            q = qkv[:, :c]
            k = qkv[:, c : c + hkv * d]
            v = qkv[:, c + hkv * d :]
        else:
            q = tf.einsum("bc,cd->bd", x_t, cast_like(self.wq, x_t))
            k = tf.einsum("bc,cd->bd", x_t, cast_like(self.wk, x_t))
            v = tf.einsum("bc,cd->bd", x_t, cast_like(self.wv, x_t))

        q = tf.reshape(q, [b, h, d])
        q = tf.reshape(q, [b, hkv, g, d])
        k = tf.reshape(k, [b, hkv, d])
        v = tf.reshape(v, [b, hkv, d])

        if cfg.pos_encoding == "rope":
            cos, sin = rope_cos_sin(tf.reshape(t, [1]), self.inv_freq, scale_factor=cfg.rope_scale_factor)
            cos = tf.cast(cos, q.dtype)
            sin = tf.cast(sin, q.dtype)
            q = apply_rope(q, cos[None, None, None, :], sin[None, None, None, :], d)
            k = apply_rope(k, cos[None, None, :], sin[None, None, :], d)

        k_now = tf.reshape(k, [b * hkv, d])
        v_now = tf.reshape(v, [b * hkv, d])
        if k_now.dtype != kflat.dtype:
            k_now = tf.cast(k_now, kflat.dtype)
            v_now = tf.cast(v_now, vflat.dtype)

        bk = tf.shape(k_now)[0]
        idx = tf.stack([tf.range(bk, dtype=tf.int32), tf.fill([bk], slot)], axis=1)

        if isinstance(kflat, tf.Variable):
            kflat.scatter_nd_update(idx, k_now)
            vflat.scatter_nd_update(idx, v_now)
            kflat_out, vflat_out = kflat, vflat
        else:
            kflat_out = tf.tensor_scatter_nd_update(kflat, idx, k_now)
            vflat_out = tf.tensor_scatter_nd_update(vflat, idx, v_now)

        if cfg.pos_encoding == "alibi":
            tvals = tf.fill([bk], t)
            if isinstance(pflat, tf.Variable):
                pflat.scatter_nd_update(idx, tvals)
                pflat_out = pflat
            else:
                pflat_out = tf.tensor_scatter_nd_update(pflat, idx, tvals)
        else:
            pflat_out = None

        k_cache = tf.reshape(kflat_out, [b, hkv, w, d])
        v_cache = tf.reshape(vflat_out, [b, hkv, w, d])

        scores = tf.einsum("bkgd,bkWd->bkgW", q, tf.cast(k_cache, q.dtype)) * tf.cast(self.scale, q.dtype)
        sf = tf.cast(scores, tf.float32)

        if cfg.pos_encoding == "alibi":
            p = tf.reshape(pflat_out, [b, hkv, w])
            valid = p >= 0
            dist = tf.cast(t - p, tf.float32)
            slopes = self.alibi_slopes_hg
            bias = -slopes[None, :, :, None] * dist[:, :, None, :]
            sf = sf + bias
            valid4 = valid[:, :, None, :]
        else:
            filled = tf.minimum(t + 1, tf.cast(w, tf.int32))
            valid4 = tf.range(w, dtype=tf.int32)[None, None, None, :] < filled

        sf = tf.where(valid4, sf, tf.fill(tf.shape(sf), tf.constant(-1e9, tf.float32)))
        wprob = tf.nn.softmax(sf, axis=-1)
        wprob = tf.cast(wprob, tf.cast(v_cache, COMPUTE_DTYPE).dtype)

        out = tf.einsum("bkgW,bkWd->bkgd", wprob, tf.cast(v_cache, wprob.dtype))
        out = tf.reshape(out, [b, h, d])
        out = tf.reshape(out, [b, c])
        out = tf.einsum("bc,cd->bd", out, cast_like(self.wo, out))

        attn_heads = tf.reshape(wprob, [b, h, w])

        if cfg.pos_encoding == "alibi":
            if return_attn:
                return out, (kflat_out, vflat_out, pflat_out), attn_heads
            return out, (kflat_out, vflat_out, pflat_out)

        if return_attn:
            return out, (kflat_out, vflat_out), attn_heads
        return out, (kflat_out, vflat_out)


class Block(tf.keras.layers.Layer):
    def __init__(self, cfg: GPTConfig, **kwargs):
        super().__init__(**kwargs)
        self.ln1 = RMSNorm(cfg.rmsnorm_eps, cfg.rmsnorm_scale)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = RMSNorm(cfg.rmsnorm_eps, cfg.rmsnorm_scale)
        self.mlp = MLP(cfg)

    def call(self, x, training=False):
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x), training=training)
        return x

    def step_ring(self, x_t, cache_entry, t, return_attn=False):
        if return_attn:
            a, cache_entry, attn_heads = self.attn.step_ring(self.ln1(x_t), cache_entry, t, return_attn=True)
        else:
            a, cache_entry = self.attn.step_ring(self.ln1(x_t), cache_entry, t, return_attn=False)
        a = tf.cast(a, x_t.dtype)
        x_t = x_t + a
        m = self.mlp.step(self.ln2(x_t))
        x_t = x_t + tf.cast(m, x_t.dtype)
        if return_attn:
            return x_t, cache_entry, attn_heads
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
            self.lm_head = self.add_weight(name="lm_head", shape=(cfg.vocab_size, cfg.n_embd), initializer=INIT, trainable=True)
        else:
            self.lm_head = None

    def _lm_weight(self):
        return self.wte.embeddings if self.cfg.tie_embeddings else self.lm_head

    def call(self, idx, training=False):
        x = self.wte(idx)
        x = tf.cast(x, COMPUTE_DTYPE)
        x = self.emb_norm(x)
        for blk in self.blocks:
            x = blk(x, training=training)
        x = self.final_norm(x)
        w = self._lm_weight()
        return tf.einsum("btc,vc->btv", x, tf.cast(w, x.dtype))

    def init_kv_cache(self, batch_size, as_variables=False):
        hkv = self.cfg.n_kv_head
        w = self.cfg.attn_window
        d = self.head_dim
        bk = batch_size * hkv
        zeros_kv = tf.zeros([bk, w, d], dtype=CACHE_DTYPE)
        cache = []
        for _ in range(self.cfg.n_layer):
            if as_variables:
                k = tf.Variable(zeros_kv, trainable=False)
                v = tf.Variable(zeros_kv, trainable=False)
            else:
                k = zeros_kv
                v = zeros_kv
            if self.cfg.pos_encoding == "alibi":
                init_p = tf.fill([bk, w], tf.constant(-1, tf.int32))
                p = tf.Variable(init_p, trainable=False) if as_variables else init_p
                cache.append((k, v, p))
            else:
                cache.append((k, v))
        return cache

    def reset_kv_cache(self, cache):
        for entry in cache:
            if self.cfg.pos_encoding == "alibi":
                k, v, p = entry
                if isinstance(k, tf.Variable):
                    k.assign(tf.zeros_like(k))
                if isinstance(v, tf.Variable):
                    v.assign(tf.zeros_like(v))
                if isinstance(p, tf.Variable):
                    p.assign(tf.fill(tf.shape(p), tf.constant(-1, tf.int32)))
            else:
                k, v = entry
                if isinstance(k, tf.Variable):
                    k.assign(tf.zeros_like(k))
                if isinstance(v, tf.Variable):
                    v.assign(tf.zeros_like(v))

    def step_ring(self, token_id, t, cache, return_attn=False):
        x_t = self.wte(token_id)
        x_t = tf.cast(x_t, COMPUTE_DTYPE)
        x_t = self.emb_norm(x_t)
        new_cache = []
        attn_by_layer = []
        for li, blk in enumerate(self.blocks):
            if return_attn:
                x_t, entry, attn_heads = blk.step_ring(x_t, cache[li], t, return_attn=True)
                attn_by_layer.append(attn_heads)
            else:
                x_t, entry = blk.step_ring(x_t, cache[li], t, return_attn=False)
            new_cache.append(entry)
        x_t = self.final_norm(x_t)
        w = self._lm_weight()
        logits = tf.einsum("bc,vc->bv", x_t, tf.cast(w, x_t.dtype))
        if return_attn:
            return logits, new_cache, attn_by_layer
        return logits, new_cache
