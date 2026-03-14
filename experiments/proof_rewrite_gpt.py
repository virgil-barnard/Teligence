
"""
proof_rewrite_gpt.py

A single-file, explicit, desktop-scale demo of:

1) A tiny GPT-style causal LM in TensorFlow with:
   - GQA (num_kv_heads < num_heads)
   - RoPE with YaRN (4k base -> 64x scaling), using the same formulas as HF transformers

2) A "Proof Gym" environment:
   - State is an equality goal:  lhs == rhs
   - Action is a tool call: apply a rewrite rule at a tree node index
   - Observation is the updated lhs, plus whether the tool succeeded

3) Training as an *interactive agent* (but without "boil the ocean" coding rollouts):
   - Generate expert trajectories by randomly applying valid rewrite rules
   - Train the LM via masked SFT so it learns to output ACTION tokens after an [ACT] prompt
   - Evaluate by rolling out the model in the environment and measuring solve rate

Run:
  python experiments/proof_rewrite_gpt.py --help

Notes:
- This is intentionally minimal and explicit (Karpathy-ish), not maximally optimized.
- The environment is fully synthetic + verifiable (because we generate the target by rewrites).
- You can later swap the environment for a real prover (Lean/Metamath) without changing the core agent loop.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import math
import os
import random
import sys
import time
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from teligence.config import GPTConfig, validate_config
from teligence.modeling import ExplicitGPT, set_precision

# -----------------------------------------------------------------------------
# Reproducibility

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -----------------------------------------------------------------------------
# Expression language (tiny algebra)

# Expr is an immutable binary tree: const/int, var/name, add(a,b), mul(a,b).
@dataclass(frozen=True)
class Expr:
    op: str                      # "const" | "var" | "add" | "mul"
    a: Optional[object] = None   # for const/var: value; for add/mul: left child
    b: Optional["Expr"] = None   # for add/mul: right child

    @staticmethod
    def Const(v: int) -> "Expr":
        return Expr("const", int(v), None)

    @staticmethod
    def Var(name: str) -> "Expr":
        return Expr("var", str(name), None)

    @staticmethod
    def Add(x: "Expr", y: "Expr") -> "Expr":
        return Expr("add", x, y)

    @staticmethod
    def Mul(x: "Expr", y: "Expr") -> "Expr":
        return Expr("mul", x, y)

    def is_leaf(self) -> bool:
        return self.op in ("const", "var")

    def size(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + self.a.size() + self.b.size()  # type: ignore[union-attr]

    def depth(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + max(self.a.depth(), self.b.depth())  # type: ignore[union-attr]

    def pretty(self) -> str:
        if self.op == "const":
            return str(self.a)
        if self.op == "var":
            return str(self.a)
        if self.op == "add":
            return f"({self.a.pretty()}+{self.b.pretty()})"  # type: ignore[union-attr]
        if self.op == "mul":
            return f"({self.a.pretty()}*{self.b.pretty()})"  # type: ignore[union-attr]
        raise ValueError(f"bad op: {self.op}")

    def to_tokens(self) -> List[str]:
        # Prefix, arity-2 ops => unambiguous
        if self.op == "const":
            return [str(self.a)]
        if self.op == "var":
            return [str(self.a)]
        if self.op == "add":
            return ["ADD"] + self.a.to_tokens() + self.b.to_tokens()  # type: ignore[union-attr]
        if self.op == "mul":
            return ["MUL"] + self.a.to_tokens() + self.b.to_tokens()  # type: ignore[union-attr]
        raise ValueError(f"bad op: {self.op}")

# Binary-heap node indices:
# root=1, left=2*i, right=2*i+1.
def _index_to_path(index: int) -> List[int]:
    if index < 1:
        raise ValueError("index must be >= 1")
    # Represent index in binary; drop leading '1'; remaining bits are directions (0=left,1=right).
    bits = bin(index)[3:]  # e.g. index=1 -> '' ; index=5 (0b101) -> '01'
    return [0 if c == "0" else 1 for c in bits]

def get_subexpr(root: Expr, index: int) -> Optional[Expr]:
    path = _index_to_path(index)
    cur = root
    for d in path:
        if cur.is_leaf():
            return None
        cur = cur.a if d == 0 else cur.b  # type: ignore[union-attr]
    return cur

def replace_subexpr(root: Expr, index: int, new_sub: Expr) -> Expr:
    path = _index_to_path(index)

    def rec(node: Expr, pi: int) -> Expr:
        if pi == len(path):
            return new_sub
        if node.is_leaf():
            return node  # target doesn't exist; no-op
        d = path[pi]
        if d == 0:
            return Expr(node.op, rec(node.a, pi + 1), node.b)  # type: ignore[arg-type]
        else:
            return Expr(node.op, node.a, rec(node.b, pi + 1))  # type: ignore[arg-type]

    return rec(root, 0)

def iter_nodes_with_indices(root: Expr) -> Iterable[Tuple[int, Expr]]:
    # DFS traversal, returning heap indices.
    stack: List[Tuple[int, Expr]] = [(1, root)]
    while stack:
        idx, node = stack.pop()
        yield idx, node
        if not node.is_leaf():
            # push right then left so left pops first (preorder-ish)
            stack.append((2 * idx + 1, node.b))  # type: ignore[arg-type]
            stack.append((2 * idx + 0, node.a))  # type: ignore[arg-type]

# -----------------------------------------------------------------------------
# Rewrite rules ("the tool")

@dataclass(frozen=True)
class Rule:
    name: str
    # Applies to a subexpression; returns rewritten expression or None.
    fn: callable

def _is_const(node: Expr, v: int) -> bool:
    return node.op == "const" and int(node.a) == v

def _same_expr(x: Expr, y: Expr) -> bool:
    # Structural equality (dataclass handles this), but keep explicit.
    return x == y

def make_rules() -> List[Rule]:
    rules: List[Rule] = []

    # Distribute:
    #   a*(b+c) -> a*b + a*c
    def dist_right(node: Expr) -> Optional[Expr]:
        if node.op != "mul":
            return None
        a, bc = node.a, node.b  # type: ignore[assignment]
        if bc.op != "add":
            return None
        b, c = bc.a, bc.b  # type: ignore[assignment]
        return Expr.Add(Expr.Mul(a, b), Expr.Mul(a, c))

    #   (a+b)*c -> a*c + b*c
    def dist_left(node: Expr) -> Optional[Expr]:
        if node.op != "mul":
            return None
        ab, c = node.a, node.b  # type: ignore[assignment]
        if ab.op != "add":
            return None
        a, b = ab.a, ab.b  # type: ignore[assignment]
        return Expr.Add(Expr.Mul(a, c), Expr.Mul(b, c))

    # Factor (inverse distribute):
    #   a*b + a*c -> a*(b+c)
    def factor_left(node: Expr) -> Optional[Expr]:
        if node.op != "add":
            return None
        x, y = node.a, node.b  # type: ignore[assignment]
        if x.op != "mul" or y.op != "mul":
            return None
        a1, b1 = x.a, x.b  # type: ignore[assignment]
        a2, b2 = y.a, y.b  # type: ignore[assignment]
        if not _same_expr(a1, a2):
            return None
        return Expr.Mul(a1, Expr.Add(b1, b2))

    #   b*a + c*a -> (b+c)*a
    def factor_right(node: Expr) -> Optional[Expr]:
        if node.op != "add":
            return None
        x, y = node.a, node.b  # type: ignore[assignment]
        if x.op != "mul" or y.op != "mul":
            return None
        b1, a1 = x.a, x.b  # type: ignore[assignment]
        b2, a2 = y.a, y.b  # type: ignore[assignment]
        if not _same_expr(a1, a2):
            return None
        return Expr.Mul(Expr.Add(b1, b2), a1)

    # Simplifications (kept tiny but useful):
    #   a + 0 -> a
    def add_zero_r(node: Expr) -> Optional[Expr]:
        if node.op == "add" and _is_const(node.b, 0):  # type: ignore[arg-type]
            return node.a  # type: ignore[return-value]
        return None

    #   0 + a -> a
    def add_zero_l(node: Expr) -> Optional[Expr]:
        if node.op == "add" and _is_const(node.a, 0):  # type: ignore[arg-type]
            return node.b  # type: ignore[return-value]
        return None

    #   a * 1 -> a
    def mul_one_r(node: Expr) -> Optional[Expr]:
        if node.op == "mul" and _is_const(node.b, 1):  # type: ignore[arg-type]
            return node.a  # type: ignore[return-value]
        return None

    #   1 * a -> a
    def mul_one_l(node: Expr) -> Optional[Expr]:
        if node.op == "mul" and _is_const(node.a, 1):  # type: ignore[arg-type]
            return node.b  # type: ignore[return-value]
        return None

    #   a * 0 -> 0
    def mul_zero_r(node: Expr) -> Optional[Expr]:
        if node.op == "mul" and _is_const(node.b, 0):  # type: ignore[arg-type]
            return Expr.Const(0)
        return None

    #   0 * a -> 0
    def mul_zero_l(node: Expr) -> Optional[Expr]:
        if node.op == "mul" and _is_const(node.a, 0):  # type: ignore[arg-type]
            return Expr.Const(0)
        return None

    rules.append(Rule("dist_left", dist_left))
    rules.append(Rule("dist_right", dist_right))
    rules.append(Rule("factor_left", factor_left))
    rules.append(Rule("factor_right", factor_right))
    rules.append(Rule("add0_l", add_zero_l))
    rules.append(Rule("add0_r", add_zero_r))
    rules.append(Rule("mul1_l", mul_one_l))
    rules.append(Rule("mul1_r", mul_one_r))
    rules.append(Rule("mul0_l", mul_zero_l))
    rules.append(Rule("mul0_r", mul_zero_r))
    return rules

def apply_rule_at(root: Expr, rule: Rule, index: int) -> Optional[Expr]:
    sub = get_subexpr(root, index)
    if sub is None:
        return None
    new_sub = rule.fn(sub)
    if new_sub is None:
        return None
    return replace_subexpr(root, index, new_sub)

# -----------------------------------------------------------------------------
# Random expression generator + expert trajectory synthesis

@dataclass
class ProofGymConfig:
    max_depth: int = 5
    max_nodes: int = 31            # corresponds to heap indices up to 31 (depth <= 5)
    max_steps: int = 8             # steps per trajectory
    min_trajectory_steps: int = 2
    prevent_expert_loops: bool = True
    vars: Tuple[str, ...] = ("x", "y", "z")
    consts: Tuple[int, ...] = (0, 1, 2, 3)

def random_expr(cfg: ProofGymConfig, depth: int = 0) -> Expr:
    # Simple grammar: with probability, emit leaf; else emit binary op.
    if depth >= cfg.max_depth - 1:
        if random.random() < 0.5:
            return Expr.Var(random.choice(cfg.vars))
        return Expr.Const(random.choice(cfg.consts))

    # bias to keep trees small
    p_leaf = 0.35 + 0.10 * depth
    if random.random() < p_leaf:
        if random.random() < 0.6:
            return Expr.Var(random.choice(cfg.vars))
        return Expr.Const(random.choice(cfg.consts))

    op = "add" if random.random() < 0.5 else "mul"
    a = random_expr(cfg, depth + 1)
    b = random_expr(cfg, depth + 1)
    return Expr.Add(a, b) if op == "add" else Expr.Mul(a, b)

def list_valid_actions(expr: Expr, rules: List[Rule], cfg: ProofGymConfig) -> List[Tuple[int, int, Expr]]:
    """
    Return a list of valid actions: (rule_id, node_index, new_expr)
    filtered by size/depth constraints.
    """
    out: List[Tuple[int, int, Expr]] = []
    for node_index, _node in iter_nodes_with_indices(expr):
        if node_index > cfg.max_nodes:
            continue
        for rid, rule in enumerate(rules):
            new_expr = apply_rule_at(expr, rule, node_index)
            if new_expr is None:
                continue
            if new_expr.depth() > cfg.max_depth:
                continue
            if new_expr.size() > cfg.max_nodes:  # crude but effective bound
                continue
            out.append((rid, node_index, new_expr))
    return out

@dataclass
class Trajectory:
    start: Expr
    goal: Expr
    states: List[Expr]                # includes start and intermediate states (lhs)
    actions: List[Tuple[int, int]]    # (rule_id, node_index)

def generate_trajectory(cfg: ProofGymConfig, rules: List[Rule], max_tries: int = 1000) -> Trajectory:
    """
    Generate an expert trajectory by randomly applying valid rewrites.
    """
    for _ in range(max_tries):
        start = random_expr(cfg)
        if start.size() > cfg.max_nodes or start.depth() > cfg.max_depth:
            continue
        states = [start]
        seen = {start}
        actions: List[Tuple[int, int]] = []
        cur = start
        for _step in range(cfg.max_steps):
            actions_list = list_valid_actions(cur, rules, cfg)
            if cfg.prevent_expert_loops:
                non_loop_actions = [item for item in actions_list if item[2] not in seen]
                if non_loop_actions:
                    actions_list = non_loop_actions
            if not actions_list:
                break
            rid, node_idx, nxt = random.choice(actions_list)
            actions.append((rid, node_idx))
            states.append(nxt)
            seen.add(nxt)
            cur = nxt
        required_steps = min(cfg.min_trajectory_steps, cfg.max_steps)
        if len(actions) < required_steps:
            continue
        goal = states[-1]
        if goal == start:
            continue
        return Trajectory(start=start, goal=goal, states=states, actions=actions)
    raise RuntimeError("failed to generate trajectory; relax constraints or increase max_tries")

# -----------------------------------------------------------------------------
# Tokenization / transcript format

@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]
    pad_id: int
    bos_id: int
    eos_id: int

    def encode(self, toks: Sequence[str]) -> List[int]:
        return [self.token_to_id[t] for t in toks]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.id_to_token[i] for i in ids]

def build_vocab(rules: List[Rule], cfg: ProofGymConfig) -> Tuple[Vocab, List[str]]:
    """
    Vocabulary contains:
    - special tokens: <pad>, <bos>, <eos>
    - control tokens: GOAL, STATE, ACT
    - expr tokens: ADD, MUL, variable names, constants
    - action tokens: A_{rule}_{node_index} for node_index=1..max_nodes
    """
    specials = ["<pad>", "<bos>", "<eos>"]
    controls = ["GOAL", "STATE", "ACT"]
    expr_toks = ["ADD", "MUL"] + list(cfg.vars) + [str(c) for c in cfg.consts]

    action_toks: List[str] = []
    for rid, rule in enumerate(rules):
        for node_idx in range(1, cfg.max_nodes + 1):
            action_toks.append(f"A_{rule.name}_{node_idx}")

    all_toks = specials + controls + expr_toks + action_toks
    token_to_id = {t: i for i, t in enumerate(all_toks)}
    vocab = Vocab(
        token_to_id=token_to_id,
        id_to_token=all_toks,
        pad_id=token_to_id["<pad>"],
        bos_id=token_to_id["<bos>"],
        eos_id=token_to_id["<eos>"],
    )
    return vocab, action_toks

def action_to_token(rules: List[Rule], rule_id: int, node_index: int) -> str:
    return f"A_{rules[rule_id].name}_{node_index}"

def token_to_action(rules: List[Rule], tok: str) -> Optional[Tuple[int, int]]:
    if not tok.startswith("A_"):
        return None
    # A_{rule.name}_{node}
    parts = tok.split("_")
    if len(parts) < 3:
        return None
    node_index = int(parts[-1])
    rule_name = "_".join(parts[1:-1])
    for rid, r in enumerate(rules):
        if r.name == rule_name:
            return rid, node_index
    return None

def encode_transcript(tr: Trajectory, vocab: Vocab, rules: List[Rule], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a single sequence:
      <bos> GOAL <goal_expr> STATE <state0> ACT <action0> STATE <state1> ACT <action1> ... <eos>

    We train with masked LM: only action tokens contribute to loss.
    Return:
      input_ids:  [seq_len]
      loss_mask: [seq_len] aligned with *labels* (i.e. next-token targets).
    """
    toks: List[str] = ["<bos>", "GOAL"] + tr.goal.to_tokens() + ["STATE"] + tr.start.to_tokens()

    # Each step: ACT action_token STATE next_state_tokens
    for (rid, node_idx), nxt_state in zip(tr.actions, tr.states[1:]):
        toks += ["ACT", action_to_token(rules, rid, node_idx), "STATE"] + nxt_state.to_tokens()

    toks += ["<eos>"]

    ids = vocab.encode(toks)
    if len(ids) > seq_len:
        # truncate from the left while keeping BOS, and always keep last tokens
        ids = ids[-seq_len:]
        ids[0] = vocab.bos_id

    # pad to seq_len
    pad = [vocab.pad_id] * (seq_len - len(ids))
    ids = ids + pad
    input_ids = np.array(ids, dtype=np.int32)

    # next-token labels are shifted left by 1
    labels = np.roll(input_ids, shift=-1)
    labels[-1] = vocab.pad_id

    # Loss mask on positions where the label is an action token id.
    # i.e. after the model reads "... ACT", it should predict the next token (the action token).
    loss_mask = np.zeros_like(input_ids, dtype=np.float32)
    for i in range(seq_len - 1):
        tok_id = labels[i]
        tok = vocab.id_to_token[int(tok_id)]
        if tok.startswith("A_"):
            loss_mask[i] = 1.0

    return input_ids, loss_mask

# -----------------------------------------------------------------------------
# Proof environment (interactive loop)

@dataclass
class ProofState:
    lhs: Expr
    rhs: Expr
    step: int = 0
    done: bool = False

class ProofEnv:
    def __init__(self, cfg: ProofGymConfig, rules: List[Rule]):
        self.cfg = cfg
        self.rules = rules
        self.state: Optional[ProofState] = None

    def reset(self, start: Expr, goal: Expr) -> ProofState:
        self.state = ProofState(lhs=start, rhs=goal, step=0, done=False)
        return self.state

    def is_solved(self) -> bool:
        assert self.state is not None
        return self.state.lhs == self.state.rhs

    def step(self, action: Tuple[int, int]) -> Tuple[ProofState, float, bool, Dict[str, object]]:
        assert self.state is not None
        if self.state.done:
            return self.state, 0.0, True, {"already_done": True}

        rid, node_idx = action
        info: Dict[str, object] = {"valid": False}
        reward = -0.01  # step penalty

        if rid < 0 or rid >= len(self.rules):
            reward -= 0.1
        else:
            new_lhs = apply_rule_at(self.state.lhs, self.rules[rid], node_idx)
            if new_lhs is None:
                reward -= 0.1
            else:
                # enforce bounds
                if new_lhs.depth() > self.cfg.max_depth or new_lhs.size() > self.cfg.max_nodes:
                    reward -= 0.1
                else:
                    self.state.lhs = new_lhs
                    info["valid"] = True

        self.state.step += 1
        done = self.is_solved() or (self.state.step >= self.cfg.max_steps)
        self.state.done = done
        if self.is_solved():
            reward += 1.0
            info["solved"] = True
        else:
            info["solved"] = False
        return self.state, reward, done, info

# -----------------------------------------------------------------------------
# GPT model notes

# This file originally carried a standalone GPT implementation.
# For repository consistency, training/eval now instantiate `ExplicitGPT`
# from `modeling.py` with `GPTConfig` from `config.py`.

@dataclass
class ModelConfig:
    vocab_size: int
    seq_len: int = 256
    n_layer: int = 4
    n_embd: int = 256
    n_head: int = 8
    n_kv_head: int = 2
    mlp_mult: int = 4
    dropout: float = 0.0

    # RoPE / YaRN (Devstral-aligned)
    rope_type: str = "yarn"
    rope_theta: float = 1_000_000.0
    original_max_pos: int = 4096
    yarn_factor: float = 64.0
    beta_fast: float = 4.0
    beta_slow: float = 1.0
    mscale: float = 1.0
    mscale_all_dim: float = 0.0  # keep 0.0 to match Devstral config behavior (=> use get_mscale(factor) only)

    @property
    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0
        return self.n_embd // self.n_head

    @property
    def q_per_kv(self) -> int:
        assert self.n_head % self.n_kv_head == 0
        return self.n_head // self.n_kv_head

class RMSNorm(tf.keras.layers.Layer):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = tf.Variable(tf.ones([dim]), trainable=True, name="rms_scale")

    def call(self, x: tf.Tensor) -> tf.Tensor:
        # x: [..., dim]
        x_f = tf.cast(x, tf.float32)
        ms = tf.reduce_mean(tf.square(x_f), axis=-1, keepdims=True)
        x_norm = x_f * tf.math.rsqrt(ms + self.eps)
        return tf.cast(x_norm, x.dtype) * self.scale

def _yarn_get_mscale(scale: float, mscale: float = 1.0) -> float:
    if scale <= 1.0:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0

def yarn_inv_freq_and_factor(cfg: ModelConfig) -> Tuple[tf.Tensor, float]:
    """
    Mirrors HF transformers' _compute_yarn_parameters (see modeling_rope_utils.py).
    Returns:
      inv_freq: [dim//2] float32
      attention_factor: float
    """
    dim = cfg.head_dim  # full rotary dim; we apply full-head RoPE
    base = float(cfg.rope_theta)
    factor = float(cfg.yarn_factor)

    attention_factor = None
    if attention_factor is None:
        # replicate HF logic: if mscale and mscale_all_dim: ratio; else get_mscale(factor)
        if cfg.mscale and cfg.mscale_all_dim:
            attention_factor = float(_yarn_get_mscale(factor, cfg.mscale) / _yarn_get_mscale(factor, cfg.mscale_all_dim))
        else:
            attention_factor = float(_yarn_get_mscale(factor))

    beta_fast = float(cfg.beta_fast) if cfg.beta_fast is not None else 32.0
    beta_slow = float(cfg.beta_slow) if cfg.beta_slow is not None else 1.0

    def find_correction_dim(num_rotations: float, dim_full: int, base: float, max_pos: int) -> float:
        return (dim_full * math.log(max_pos / (num_rotations * 2 * math.pi))) / (2 * math.log(base))

    def find_correction_range(low_rot: float, high_rot: float, dim_full: int, base: float, max_pos: int, truncate: bool) -> Tuple[float, float]:
        low = find_correction_dim(low_rot, dim_full, base, max_pos)
        high = find_correction_dim(high_rot, dim_full, base, max_pos)
        if truncate:
            low = math.floor(low)
            high = math.ceil(high)
        return max(low, 0.0), min(high, float(dim_full - 1))

    def linear_ramp_factor(minv: float, maxv: float, dim_half: int) -> tf.Tensor:
        # HF: if min==max, max += 0.001
        if minv == maxv:
            maxv += 0.001
        t = tf.cast(tf.range(dim_half), tf.float32)
        linear = (t - minv) / (maxv - minv)
        return tf.clip_by_value(linear, 0.0, 1.0)

    # pos_freqs = base ** (arange(0, dim, 2)/dim)
    dim_full = dim
    dim_half = dim_full // 2
    pos = tf.cast(tf.range(0, dim_full, delta=2), tf.float32) / tf.cast(dim_full, tf.float32)
    pos_freqs = tf.pow(tf.constant(base, tf.float32), pos)  # [dim_half]

    inv_extrap = 1.0 / pos_freqs
    inv_interp = 1.0 / (factor * pos_freqs)

    truncate = True
    low, high = find_correction_range(beta_fast, beta_slow, dim_full, base, cfg.original_max_pos, truncate)

    inv_extrap_factor = 1.0 - linear_ramp_factor(low, high, dim_half)  # [dim_half]
    inv = inv_interp * (1.0 - inv_extrap_factor) + inv_extrap * inv_extrap_factor
    return inv, float(attention_factor)

def apply_rope_yarn(x: tf.Tensor, positions: tf.Tensor, inv_freq: tf.Tensor, attn_factor: float) -> tf.Tensor:
    """
    Apply standard interleaved RoPE on last dimension, then scale by attention factor.
    x: [B, T, H, Dh]
    positions: [T]
    inv_freq: [Dh/2]
    """
    x_dtype = x.dtype
    x = tf.cast(x, tf.float32)
    Dh = x.shape[-1]
    half = Dh // 2

    angles = tf.einsum("t,d->td", tf.cast(positions, tf.float32), tf.cast(inv_freq, tf.float32))  # [T,half]
    cos = tf.cos(angles)[None, :, None, :]  # [1,T,1,half]
    sin = tf.sin(angles)[None, :, None, :]

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]

    out_even = x_even * cos - x_odd * sin
    out_odd = x_even * sin + x_odd * cos

    out = tf.stack([out_even, out_odd], axis=-1)           # [B,T,H,half,2]
    out = tf.reshape(out, tf.shape(x))                     # [B,T,H,Dh]
    out *= tf.cast(attn_factor, tf.float32)                # YaRN attention scaling (Eq. 15 in paper)
    return tf.cast(out, x_dtype)

class CausalSelfAttentionGQA(tf.keras.layers.Layer):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.n_head = cfg.n_head
        self.n_kv = cfg.n_kv_head
        self.dh = cfg.head_dim
        self.g = cfg.q_per_kv

        # Fused projection: q = n_head*dh, k = n_kv*dh, v = n_kv*dh
        self.w_qkv = tf.keras.layers.Dense(cfg.n_embd + 2 * cfg.n_kv_head * self.dh, use_bias=False)
        self.w_o = tf.keras.layers.Dense(cfg.n_embd, use_bias=False)

        # RoPE params
        self.inv_freq, self.attn_factor = yarn_inv_freq_and_factor(cfg)

        # Causal mask precomputed for max seq_len
        mask = np.tril(np.ones((cfg.seq_len, cfg.seq_len), dtype=np.float32))
        self.causal_mask = tf.constant(mask[None, None, None, :, :])  # [1,1,1,T,T]

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        # x: [B,T,C]
        B = tf.shape(x)[0]
        T = tf.shape(x)[1]

        qkv = self.w_qkv(x)  # [B,T, C + 2*n_kv*dh]
        # split
        q_raw = qkv[:, :, : self.cfg.n_embd]  # [B,T,C]
        kv_raw = qkv[:, :, self.cfg.n_embd :]  # [B,T, 2*n_kv*dh]
        k_raw = kv_raw[:, :, : self.n_kv * self.dh]
        v_raw = kv_raw[:, :, self.n_kv * self.dh :]

        # reshape
        q = tf.reshape(q_raw, [B, T, self.n_head, self.dh])          # [B,T,H,Dh]
        k = tf.reshape(k_raw, [B, T, self.n_kv, self.dh])            # [B,T,HKV,Dh]
        v = tf.reshape(v_raw, [B, T, self.n_kv, self.dh])            # [B,T,HKV,Dh]

        # RoPE on q and k (k has fewer heads, still same Dh)
        pos = tf.range(T, dtype=tf.int32)
        q = apply_rope_yarn(q, pos, self.inv_freq, self.attn_factor)
        k = apply_rope_yarn(k, pos, self.inv_freq, self.attn_factor)

        # reshape for GQA: group queries under each kv head
        # q: [B,T,H,Dh] -> [B,HKV,G,T,Dh]
        q = tf.reshape(q, [B, T, self.n_kv, self.g, self.dh])
        q = tf.transpose(q, [0, 2, 3, 1, 4])  # [B,HKV,G,T,Dh]
        k = tf.transpose(k, [0, 2, 1, 3])     # [B,HKV,T,Dh]
        v = tf.transpose(v, [0, 2, 1, 3])     # [B,HKV,T,Dh]

        # attn logits: [B,HKV,G,T,T]
        scale = tf.cast(1.0 / math.sqrt(self.dh), tf.float32)
        # NOTE: tf.einsum label space doesn't allow spaces; fix by explicit string:
        att = tf.einsum("bkgtd,bksd->bkgts", q, k) * scale

        # causal mask (slice to current T)
        m = self.causal_mask[:, :, :, :T, :T]
        att = tf.where(m > 0, att, tf.fill(tf.shape(att), tf.cast(-1e9, att.dtype)))

        # softmax
        w = tf.nn.softmax(att, axis=-1)

        # weighted sum: [B,HKV,G,T,Dh]
        y = tf.einsum("bkgts,bksd->bkgtd", w, v)

        # merge heads back: [B,T,HKV,G,Dh] -> [B,T,H,Dh] -> [B,T,C]
        y = tf.transpose(y, [0, 3, 1, 2, 4])              # [B,T,HKV,G,Dh]
        y = tf.reshape(y, [B, T, self.n_head, self.dh])   # [B,T,H,Dh]
        y = tf.reshape(y, [B, T, self.cfg.n_embd])        # [B,T,C]
        return self.w_o(y)

class MLP(tf.keras.layers.Layer):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = cfg.mlp_mult * cfg.n_embd
        # SwiGLU: (x W1) * silu(x W2), then W3
        self.w1 = tf.keras.layers.Dense(hidden, use_bias=False)
        self.w2 = tf.keras.layers.Dense(hidden, use_bias=False)
        self.w3 = tf.keras.layers.Dense(cfg.n_embd, use_bias=False)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        a = self.w1(x)
        b = self.w2(x)
        return self.w3(a * tf.nn.silu(b))

class Block(tf.keras.layers.Layer):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = RMSNorm(cfg.n_embd)
        self.attn = CausalSelfAttentionGQA(cfg)
        self.ln2 = RMSNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = x + self.attn(self.ln1(x), training=training)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT(tf.keras.Model):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.wte = tf.keras.layers.Embedding(cfg.vocab_size, cfg.n_embd)
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        self.ln_f = RMSNorm(cfg.n_embd)
        self.lm_head = tf.keras.layers.Dense(cfg.vocab_size, use_bias=False)

    def call(self, input_ids: tf.Tensor, training: bool = False) -> tf.Tensor:
        # [B,T] -> [B,T,C]
        x = self.wte(input_ids)
        for b in self.blocks:
            x = b(x, training=training)
        x = self.ln_f(x)
        return self.lm_head(x)  # [B,T,V]

# -----------------------------------------------------------------------------
# Training / evaluation

@dataclass
class TrainConfig:
    seq_len: int = 256
    batch_size: int = 32
    train_steps: int = 2000
    lr: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 100
    eval_every: int = 200
    eval_episodes: int = 200
    print_every: int = 50
    profile: bool = False
    trace_eval: bool = False
    eval_show_examples: int = 0
    eval_show_success_examples: int = 2
    eval_show_hardest_solved: bool = True
    curriculum_every: int = 0
    curriculum_max_depth: int = 7
    curriculum_max_nodes: int = 63
    curriculum_max_steps: int = 12
    curriculum_mix_levels: bool = True

def make_dataset(
    vocab: Vocab,
    rules: List[Rule],
    gym_cfg: ProofGymConfig,
    train_cfg: TrainConfig,
    steps: int,
) -> tf.data.Dataset:
    def gen():
        last_max_level = -1
        sample_i = 0
        while True:
            cur_cfg = gym_cfg
            if train_cfg.curriculum_every > 0:
                step_i = sample_i // max(1, train_cfg.batch_size)
                max_level = step_i // train_cfg.curriculum_every
                if train_cfg.curriculum_mix_levels and max_level > 0:
                    level = random.randint(0, max_level)
                else:
                    level = max_level
                depth = min(gym_cfg.max_depth + level, train_cfg.curriculum_max_depth)
                nodes = min(gym_cfg.max_nodes + (level * 8), train_cfg.curriculum_max_nodes)
                steps = min(gym_cfg.max_steps + level, train_cfg.curriculum_max_steps)
                cur_cfg = dataclasses.replace(gym_cfg, max_depth=depth, max_nodes=nodes, max_steps=steps)
                if max_level != last_max_level:
                    print(
                        f"[curriculum] max_level={max_level} sampled_level={level} depth={cur_cfg.max_depth} "
                        f"nodes={cur_cfg.max_nodes} steps={cur_cfg.max_steps}"
                    )
                    last_max_level = max_level

            try:
                tr = generate_trajectory(cur_cfg, rules, max_tries=1000)
            except RuntimeError:
                relaxed_cfg = dataclasses.replace(
                    cur_cfg,
                    min_trajectory_steps=max(1, cur_cfg.min_trajectory_steps - 1),
                    prevent_expert_loops=False,
                )
                print(
                    "[curriculum] generator fallback: relaxing constraints "
                    f"depth={relaxed_cfg.max_depth} nodes={relaxed_cfg.max_nodes} steps={relaxed_cfg.max_steps}"
                )
                tr = generate_trajectory(relaxed_cfg, rules, max_tries=2000)
            ids, mask = encode_transcript(tr, vocab, rules, train_cfg.seq_len)
            yield ids, mask
            sample_i += 1

    out_sig = (
        tf.TensorSpec(shape=(train_cfg.seq_len,), dtype=tf.int32),
        tf.TensorSpec(shape=(train_cfg.seq_len,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    ds = ds.batch(train_cfg.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds.take(steps)

def lr_schedule(step: int, cfg: TrainConfig) -> float:
    # warmup then cosine decay to non-zero floor
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    t = (step - cfg.warmup_steps) / max(1, (cfg.train_steps - cfg.warmup_steps))
    t = min(1.0, max(0.0, t))
    cosine = 0.5 * (1.0 + math.cos(math.pi * t))
    return cfg.min_lr + (cfg.lr - cfg.min_lr) * cosine

@tf.function(jit_compile=False)
def train_step(model: tf.keras.Model, opt: tf.keras.optimizers.Optimizer, x: tf.Tensor, loss_mask: tf.Tensor) -> tf.Tensor:
    # x: [B,T]
    # loss_mask: [B,T] aligned with label positions (next-token)
    with tf.GradientTape() as tape:
        logits = model(x, training=True)  # [B,T,V]
        # labels: next token
        labels = tf.concat([x[:, 1:], tf.fill([tf.shape(x)[0], 1], tf.cast(0, x.dtype))], axis=1)
        # cross-entropy per position
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)  # [B,T]
        ce = tf.cast(ce, tf.float32)
        # apply mask
        denom = tf.reduce_sum(loss_mask) + 1e-6
        loss = tf.reduce_sum(ce * loss_mask) / denom
    grads = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    opt.apply_gradients(grads_and_vars)
    return loss

def policy_rollout(
    model: tf.keras.Model,
    vocab: Vocab,
    rules: List[Rule],
    gym_cfg: ProofGymConfig,
    seq_len: int,
    greedy: bool = True,
    trace: bool = False,
    constrain_to_valid: bool = True,
    avoid_revisits: bool = True,
) -> Tuple[bool, int, List[str]]:
    """
    Roll out the model as an agent:
    - Provide: <bos> GOAL goal STATE start ACT
    - Repeatedly: model predicts next token => must be an action token; apply tool; append STATE newlhs ACT
    Stop if solved or max_steps.
    """
    tr = generate_trajectory(gym_cfg, rules)
    env = ProofEnv(gym_cfg, rules)
    st = env.reset(tr.start, tr.goal)
    visited_states = {st.lhs}
    trace_lines: List[str] = []
    if trace:
        trace_lines.append(f"goal={st.rhs.pretty()}")
        trace_lines.append(f"start={st.lhs.pretty()}")

    toks = ["<bos>", "GOAL"] + tr.goal.to_tokens() + ["STATE"] + st.lhs.to_tokens() + ["ACT"]
    ids = vocab.encode(toks)
    # left-pad/truncate context to seq_len
    def pack(ids_list: List[int]) -> np.ndarray:
        if len(ids_list) > seq_len:
            ids_list = ids_list[-seq_len:]
            ids_list[0] = vocab.bos_id
        pad = [vocab.pad_id] * (seq_len - len(ids_list))
        return np.array(ids_list + pad, dtype=np.int32)

    for step in range(gym_cfg.max_steps):
        x = pack(ids)[None, :]  # [1,T]
        logits = model(tf.constant(x), training=False)  # [1,T,V]
        # Predict next token after the last non-pad token. We know pad is on the right.
        last = min(len(ids) - 1, seq_len - 1)
        next_logits = logits[0, last, :].numpy()
        if constrain_to_valid:
            valid_actions = list_valid_actions(st.lhs, rules, gym_cfg)
            if avoid_revisits:
                non_loop_actions = [item for item in valid_actions if item[2] not in visited_states]
                if non_loop_actions:
                    valid_actions = non_loop_actions
                else:
                    if trace:
                        trace_lines.append(f"step={step+1} no_unseen_valid_actions=true")
                    return False, step + 1, trace_lines

            valid_token_ids = [vocab.token_to_id[action_to_token(rules, rid, node_idx)] for rid, node_idx, _ in valid_actions]
            if valid_token_ids:
                masked = np.full_like(next_logits, -1e9, dtype=np.float32)
                masked[np.array(valid_token_ids, dtype=np.int32)] = next_logits[np.array(valid_token_ids, dtype=np.int32)]
                next_logits = masked

        if greedy:
            next_id = int(np.argmax(next_logits))
        else:
            p = np.exp(next_logits - np.max(next_logits))
            p = p / np.sum(p)
            next_id = int(np.random.choice(len(p), p=p))
        next_tok = vocab.id_to_token[next_id]
        if trace:
            tail = vocab.decode(ids[-min(24, len(ids)):])
            trace_lines.append(f"step={step+1} input_tail={' '.join(tail)}")
            trace_lines.append(f"step={step+1} pred_token={next_tok}")
        act = token_to_action(rules, next_tok)
        if act is None:
            # invalid: model emitted a non-action token after ACT
            if trace:
                trace_lines.append(f"step={step+1} pred_invalid_action=true")
            return False, step + 1, trace_lines

        st, _r, done, info = env.step(act)
        visited_states.add(st.lhs)
        if trace:
            trace_lines.append(
                f"step={step+1} applied={rules[act[0]].name}@{act[1]} valid={bool(info.get('valid', False))} "
                f"lhs={st.lhs.pretty()} solved={env.is_solved()}"
            )
        ids.append(next_id)
        if env.is_solved():
            return True, step + 1, trace_lines
        if done:
            return False, step + 1, trace_lines

        # append new observation: STATE <lhs> ACT
        ids += vocab.encode(["STATE"] + st.lhs.to_tokens() + ["ACT"])

    return env.is_solved(), gym_cfg.max_steps, trace_lines

def evaluate(model: tf.keras.Model, vocab: Vocab, rules: List[Rule], gym_cfg: ProofGymConfig, train_cfg: TrainConfig) -> Dict[str, float]:
    solved = 0
    steps_sum = 0
    shown = 0
    shown_success = 0
    hardest_solved_steps = -1
    hardest_solved_trace: Optional[List[str]] = None
    hardest_solved_episode = -1
    eval_t0 = time.time()
    for i in range(train_cfg.eval_episodes):
        need_trace = (
            train_cfg.trace_eval
            or train_cfg.eval_show_examples > 0
            or train_cfg.eval_show_success_examples > 0
            or train_cfg.eval_show_hardest_solved
        )
        ok, nsteps, trace_lines = policy_rollout(
            model,
            vocab,
            rules,
            gym_cfg,
            train_cfg.seq_len,
            greedy=True,
            trace=need_trace,
            constrain_to_valid=True,
        )
        solved += int(ok)
        steps_sum += nsteps
        if ok and nsteps > hardest_solved_steps:
            hardest_solved_steps = nsteps
            hardest_solved_trace = trace_lines
            hardest_solved_episode = i + 1
        show_this = False
        if train_cfg.trace_eval:
            show_this = True
        elif train_cfg.eval_show_examples > 0 and shown < train_cfg.eval_show_examples:
            show_this = True
        elif ok and train_cfg.eval_show_success_examples > 0 and shown_success < train_cfg.eval_show_success_examples:
            show_this = True

        if show_this:
            print(f"[eval trace] episode {i+1}/{train_cfg.eval_episodes} solved={ok} steps={nsteps}")
            for line in trace_lines:
                print(f"[eval trace] {line}")
            shown += 1
            if ok:
                shown_success += 1
        if train_cfg.profile and (i + 1) % 25 == 0:
            dt = time.time() - eval_t0
            print(f"[profile] eval progress: {i+1}/{train_cfg.eval_episodes} episodes in {dt:.1f}s")

    if train_cfg.eval_show_hardest_solved and hardest_solved_trace is not None:
        print(
            f"[eval trace] hardest solved episode={hardest_solved_episode}/{train_cfg.eval_episodes} "
            f"steps={hardest_solved_steps}"
        )
        for line in hardest_solved_trace:
            print(f"[eval trace] {line}")

    total_eval_s = time.time() - eval_t0
    return {
        "solve_rate": solved / train_cfg.eval_episodes,
        "avg_steps": steps_sum / train_cfg.eval_episodes,
        "eval_seconds": total_eval_s,
    }

# -----------------------------------------------------------------------------
# Main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "sample"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="proof_rewrite_gpt")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--ckpt", type=str, default="")
    # quick knobs
    parser.add_argument("--train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_head", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--eval_every", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--trace_eval", action="store_true")
    parser.add_argument("--eval_show_examples", type=int, default=0)
    parser.add_argument("--eval_show_success_examples", type=int, default=2)
    parser.add_argument("--gym_max_depth", type=int, default=5)
    parser.add_argument("--gym_max_nodes", type=int, default=31)
    parser.add_argument("--gym_min_trajectory_steps", type=int, default=2)
    parser.add_argument("--allow_expert_loops", action="store_true")
    parser.add_argument("--curriculum_every", type=int, default=0)
    parser.add_argument("--curriculum_max_depth", type=int, default=7)
    parser.add_argument("--curriculum_max_nodes", type=int, default=63)
    parser.add_argument("--curriculum_max_steps", type=int, default=12)
    parser.add_argument("--curriculum_mix_levels", action="store_true")
    parser.add_argument("--no_curriculum_mix_levels", action="store_true")
    parser.add_argument("--eval_show_hardest_solved", action="store_true")
    parser.add_argument("--no_eval_show_hardest_solved", action="store_true")
    args = parser.parse_args()

    curriculum_mix_levels = True
    if args.curriculum_mix_levels:
        curriculum_mix_levels = True
    if args.no_curriculum_mix_levels:
        curriculum_mix_levels = False

    eval_show_hardest_solved = True
    if args.eval_show_hardest_solved:
        eval_show_hardest_solved = True
    if args.no_eval_show_hardest_solved:
        eval_show_hardest_solved = False

    set_seed(args.seed)

    gym_cfg = ProofGymConfig(
        max_depth=args.gym_max_depth,
        max_nodes=args.gym_max_nodes,
        max_steps=args.max_steps,
        min_trajectory_steps=args.gym_min_trajectory_steps,
        prevent_expert_loops=not args.allow_expert_loops,
    )
    rules = make_rules()
    vocab_max_nodes = max(args.gym_max_nodes, args.curriculum_max_nodes if args.curriculum_every > 0 else args.gym_max_nodes)
    vocab_cfg = dataclasses.replace(gym_cfg, max_nodes=vocab_max_nodes)
    vocab, _action_toks = build_vocab(rules, vocab_cfg)

    attn_window = min(256, args.seq_len)
    flash_block = 128
    while flash_block > 1 and (args.seq_len % flash_block != 0):
        flash_block //= 2
    run_dir = os.path.join(args.runs_dir, args.run_name)
    ckpt_dir = args.ckpt.strip() if args.ckpt.strip() else os.path.join(run_dir, "ckpt_last")

    cfg = GPTConfig(
        vocab_size=len(vocab.id_to_token),
        seq_len=args.seq_len,
        attn_window=attn_window,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        n_kv_head=args.n_kv_head,
        mlp_mult=4,
        dropout=0.0,
        use_bf16=True,
        use_flash_attn=False,
        flash_q_block=flash_block,
        flash_k_block=flash_block,
        batch_size=args.batch_size,
        num_updates=args.train_steps,
        log_every=50,
        eval_every=200,
        eval_tokens=1024,
        save_every=0,
        runs_dir=args.runs_dir,
    )
    validate_config(cfg)
    set_precision(cfg.use_bf16)

    model = ExplicitGPT(cfg)
    # Build model weights with a dummy call
    dummy = tf.zeros([1, cfg.seq_len], dtype=tf.int32)
    _ = model(dummy)

    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=3)

    if manager.latest_checkpoint:
        ckpt.restore(manager.latest_checkpoint)
        print(f"[loaded] {manager.latest_checkpoint}")

    train_cfg = TrainConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        print_every=args.print_every,
        profile=args.profile,
        trace_eval=args.trace_eval,
        eval_show_examples=args.eval_show_examples,
        eval_show_success_examples=args.eval_show_success_examples,
        curriculum_every=args.curriculum_every,
        curriculum_max_depth=args.curriculum_max_depth,
        curriculum_max_nodes=args.curriculum_max_nodes,
        curriculum_max_steps=args.curriculum_max_steps,
        curriculum_mix_levels=curriculum_mix_levels,
        eval_show_hardest_solved=eval_show_hardest_solved,
    )

    print(
        f"run_name={args.run_name} run_dir={run_dir} ckpt_dir={ckpt_dir} "
        f"proof run config | seq_len={train_cfg.seq_len} batch={train_cfg.batch_size} "
        f"steps={train_cfg.train_steps} eval_every={train_cfg.eval_every} "
        f"eval_episodes={train_cfg.eval_episodes} lr={train_cfg.lr:.2e} "
        f"min_lr={train_cfg.min_lr:.2e} flash_attn={cfg.use_flash_attn} "
        f"trace_eval={train_cfg.trace_eval} depth={gym_cfg.max_depth} nodes={gym_cfg.max_nodes} "
        f"max_steps={gym_cfg.max_steps} curriculum_every={train_cfg.curriculum_every} "
        f"curriculum_mix_levels={train_cfg.curriculum_mix_levels} "
        f"eval_show_hardest_solved={train_cfg.eval_show_hardest_solved} "
        f"min_traj_steps={gym_cfg.min_trajectory_steps} prevent_expert_loops={gym_cfg.prevent_expert_loops}"
    )

    if args.mode == "eval":
        stats = evaluate(model, vocab, rules, gym_cfg, train_cfg)
        print(stats)
        return

    if args.mode == "sample":
        # show one random expert trajectory + model rollout
        tr = generate_trajectory(gym_cfg, rules)
        print("=== Expert problem ===")
        print("start:", tr.start.pretty())
        print("goal :", tr.goal.pretty())
        print("actions:")
        for (rid, idx), st in zip(tr.actions, tr.states[1:]):
            print(f"  - {rules[rid].name}@{idx:>2} -> {st.pretty()}")
        ok, nsteps, trace_lines = policy_rollout(
            model,
            vocab,
            rules,
            gym_cfg,
            train_cfg.seq_len,
            greedy=True,
            trace=True,
            constrain_to_valid=True,
        )
        print(f"\n=== Model rollout ===\nsolved={ok} steps={nsteps}")
        for line in trace_lines:
            print(f"[sample trace] {line}")
        return

    # TRAIN
    opt = tf.keras.optimizers.Adam(learning_rate=train_cfg.lr, beta_1=0.9, beta_2=0.95, epsilon=1e-8)

    ds = make_dataset(vocab, rules, gym_cfg, train_cfg, steps=train_cfg.train_steps)

    t0 = time.time()
    for step, (x, mask) in enumerate(ds):
        step = int(step)
        lr_step = lr_schedule(step, train_cfg)
        if hasattr(opt.learning_rate, "assign"):
            opt.learning_rate.assign(lr_step)
        else:
            opt.learning_rate = lr_step

        loss = train_step(model, opt, x, mask)

        if (step + 1) % train_cfg.print_every == 0:
            dt = time.time() - t0
            lr_now = float(tf.convert_to_tensor(opt.learning_rate).numpy())
            print(f"step {step+1:5d}/{train_cfg.train_steps} loss {float(loss):.4f} lr {lr_now:.2e}  ({dt:.1f}s)")
            t0 = time.time()

        if (step + 1) % train_cfg.eval_every == 0 or (step + 1) == train_cfg.train_steps:
            eval_start = time.time()
            print(f"[eval] starting at step {step+1} ...")
            stats = evaluate(model, vocab, rules, gym_cfg, train_cfg)
            eval_wall = time.time() - eval_start
            print(
                f"[eval] step {step+1}: solve_rate={stats['solve_rate']:.3f} "
                f"avg_steps={stats['avg_steps']:.2f} "
                f"eval_s={stats['eval_seconds']:.1f} wall_s={eval_wall:.1f}"
            )
            manager.save()

    print("[done]")

if __name__ == "__main__":
    main()
