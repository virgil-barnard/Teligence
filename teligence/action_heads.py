import math
from typing import Optional

import tensorflow as tf

from teligence.config import GPTConfig
from teligence.modeling import ExplicitGPT


class ActionPointerValueModel(ExplicitGPT):
    def __init__(self, backbone_cfg: GPTConfig, pad_id: int, codec, value_loss_weight: float):
        super().__init__(backbone_cfg)
        self.pad_id = pad_id
        self.codec = codec
        self.value_loss_weight = value_loss_weight
        self.max_action_args = codec.max_args

        self.op_emb = tf.keras.layers.Embedding(codec.n_ops, backbone_cfg.n_embd)
        self.arg_embs = [tf.keras.layers.Embedding(codec.n_syms, backbone_cfg.n_embd) for _ in range(codec.max_args)]
        self.key_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.key_fc1 = tf.keras.layers.Dense(backbone_cfg.n_embd, activation=tf.nn.silu)
        self.key_fc2 = tf.keras.layers.Dense(backbone_cfg.n_embd)
        self.query = tf.keras.layers.Dense(backbone_cfg.n_embd, use_bias=False)
        self.value_head = tf.keras.layers.Dense(1)

    def encode_state(self, idx: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.encode(idx, training=training)
        not_pad = tf.cast(tf.not_equal(idx, self.pad_id), tf.int32)
        lengths = tf.reduce_sum(not_pad, axis=1)
        last = tf.maximum(lengths - 1, 0)
        b = tf.range(tf.shape(idx)[0], dtype=tf.int32)
        gather_idx = tf.stack([b, last], axis=1)
        return tf.gather_nd(tf.cast(x, tf.float32), gather_idx)

    def embed_action_structs(self, structs: tf.Tensor) -> tf.Tensor:
        op_ids = structs[:, :, 0]
        arg_ids = structs[:, :, 1:]
        key = self.op_emb(op_ids)
        for i in range(self.max_action_args):
            key = key + self.arg_embs[i](arg_ids[:, :, i])
        key = self.key_norm(key)
        return self.key_fc2(self.key_fc1(key))

    def forward_actions(
        self,
        idx: tf.Tensor,
        legal_structs: tf.Tensor,
        legal_mask: tf.Tensor,
        targets: Optional[tf.Tensor] = None,
        value_targets: Optional[tf.Tensor] = None,
        training: bool = False,
    ):
        s = self.encode_state(idx, training=training)
        q = self.query(s)
        keys = self.embed_action_structs(legal_structs)

        logits = tf.reduce_sum(keys * q[:, None, :], axis=-1) * (1.0 / math.sqrt(self.cfg.n_embd))
        neg = tf.fill(tf.shape(logits), tf.constant(-1e9, dtype=logits.dtype))
        logits = tf.where(legal_mask, logits, neg)

        value = tf.nn.softplus(self.value_head(s))[:, 0]

        policy_loss = None
        value_loss = None
        loss = None
        if targets is not None:
            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=tf.cast(logits, tf.float32))
            policy_loss = tf.reduce_mean(ce)
        if value_targets is not None:
            value_loss = tf.reduce_mean(tf.square(value - tf.cast(value_targets, tf.float32)))

        if policy_loss is not None and value_loss is not None:
            loss = policy_loss + self.value_loss_weight * value_loss
        elif policy_loss is not None:
            loss = policy_loss
        elif value_loss is not None:
            loss = value_loss
        return logits, value, loss, policy_loss, value_loss
