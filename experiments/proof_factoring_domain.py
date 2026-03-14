from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Expr:
    op: str
    val: Optional[int] = None
    name: Optional[str] = None
    left: Optional["Expr"] = None
    right: Optional["Expr"] = None

    @staticmethod
    def const(v: int) -> "Expr":
        return Expr(op="const", val=int(v))

    @staticmethod
    def var(name: str) -> "Expr":
        return Expr(op="var", name=name)

    @staticmethod
    def add(a: "Expr", b: "Expr") -> "Expr":
        return Expr(op="add", left=a, right=b)

    @staticmethod
    def mul(a: "Expr", b: "Expr") -> "Expr":
        return Expr(op="mul", left=a, right=b)

    def is_leaf(self) -> bool:
        return self.op in ("const", "var")

    def size(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + self.left.size() + self.right.size()

    def depth(self) -> int:
        if self.is_leaf():
            return 1
        return 1 + max(self.left.depth(), self.right.depth())

    def to_tokens(self) -> List[str]:
        if self.op == "const":
            return [str(self.val)]
        if self.op == "var":
            return [self.name]
        if self.op == "add":
            return ["ADD", *self.left.to_tokens(), *self.right.to_tokens()]
        if self.op == "mul":
            return ["MUL", *self.left.to_tokens(), *self.right.to_tokens()]
        raise ValueError(f"unknown op: {self.op}")

    def pretty(self) -> str:
        if self.op == "const":
            return str(self.val)
        if self.op == "var":
            return self.name
        if self.op == "add":
            return f"({self.left.pretty()}+{self.right.pretty()})"
        if self.op == "mul":
            return f"({self.left.pretty()}*{self.right.pretty()})"
        return "?"


def _index_to_path(index: int) -> List[int]:
    if index < 1:
        raise ValueError("heap index must be >= 1")
    bits = bin(index)[3:]
    return [0 if b == "0" else 1 for b in bits]


def get_subexpr(root: Expr, index: int) -> Optional[Expr]:
    cur = root
    for direction in _index_to_path(index):
        if cur is None or cur.is_leaf():
            return None
        cur = cur.left if direction == 0 else cur.right
    return cur


def replace_subexpr(root: Expr, index: int, new_sub: Expr) -> Expr:
    if index == 1:
        return new_sub
    path = _index_to_path(index)

    def rec(node: Expr, depth: int) -> Expr:
        if depth == len(path):
            return new_sub
        if node.is_leaf():
            return node
        if path[depth] == 0:
            return Expr(op=node.op, left=rec(node.left, depth + 1), right=node.right)
        return Expr(op=node.op, left=node.left, right=rec(node.right, depth + 1))

    return rec(root, 0)


def iter_nodes_with_indices(root: Expr) -> Iterable[Tuple[int, Expr]]:
    stack: List[Tuple[int, Expr]] = [(1, root)]
    while stack:
        idx, node = stack.pop()
        yield idx, node
        if not node.is_leaf():
            stack.append((idx * 2 + 1, node.right))
            stack.append((idx * 2, node.left))


@dataclass(frozen=True)
class Rule:
    name: str


def _is_const(node: Expr, v: int) -> bool:
    return node.op == "const" and node.val == v


def _same_expr(x: Expr, y: Expr) -> bool:
    return x == y


def make_rules() -> List[Rule]:
    return [
        Rule("add_zero_left"),
        Rule("add_zero_right"),
        Rule("mul_one_left"),
        Rule("mul_one_right"),
        Rule("mul_zero_left"),
        Rule("mul_zero_right"),
        Rule("comm_add"),
        Rule("comm_mul"),
        Rule("assoc_add_left"),
        Rule("assoc_add_right"),
        Rule("assoc_mul_left"),
        Rule("assoc_mul_right"),
        Rule("dist_left"),
        Rule("dist_right"),
        Rule("factor_left"),
        Rule("factor_right"),
    ]


def apply_rule_at(root: Expr, rule: Rule, index: int) -> Optional[Expr]:
    node = get_subexpr(root, index)
    if node is None:
        return None

    r = rule.name
    new_node = None

    if r == "add_zero_left" and node.op == "add" and _is_const(node.left, 0):
        new_node = node.right
    elif r == "add_zero_right" and node.op == "add" and _is_const(node.right, 0):
        new_node = node.left
    elif r == "mul_one_left" and node.op == "mul" and _is_const(node.left, 1):
        new_node = node.right
    elif r == "mul_one_right" and node.op == "mul" and _is_const(node.right, 1):
        new_node = node.left
    elif r == "mul_zero_left" and node.op == "mul" and _is_const(node.left, 0):
        new_node = Expr.const(0)
    elif r == "mul_zero_right" and node.op == "mul" and _is_const(node.right, 0):
        new_node = Expr.const(0)
    elif r == "comm_add" and node.op == "add":
        new_node = Expr.add(node.right, node.left)
    elif r == "comm_mul" and node.op == "mul":
        new_node = Expr.mul(node.right, node.left)
    elif r == "assoc_add_left" and node.op == "add" and node.left.op == "add":
        new_node = Expr.add(node.left.left, Expr.add(node.left.right, node.right))
    elif r == "assoc_add_right" and node.op == "add" and node.right.op == "add":
        new_node = Expr.add(Expr.add(node.left, node.right.left), node.right.right)
    elif r == "assoc_mul_left" and node.op == "mul" and node.left.op == "mul":
        new_node = Expr.mul(node.left.left, Expr.mul(node.left.right, node.right))
    elif r == "assoc_mul_right" and node.op == "mul" and node.right.op == "mul":
        new_node = Expr.mul(Expr.mul(node.left, node.right.left), node.right.right)
    elif r == "dist_left" and node.op == "mul" and node.left.op == "add":
        a, b, c = node.left.left, node.left.right, node.right
        new_node = Expr.add(Expr.mul(a, c), Expr.mul(b, c))
    elif r == "dist_right" and node.op == "mul" and node.right.op == "add":
        a, b, c = node.left, node.right.left, node.right.right
        new_node = Expr.add(Expr.mul(a, b), Expr.mul(a, c))
    elif r == "factor_left" and node.op == "add":
        l, rr = node.left, node.right
        if l.op == "mul" and rr.op == "mul" and _same_expr(l.left, rr.left):
            new_node = Expr.mul(l.left, Expr.add(l.right, rr.right))
    elif r == "factor_right" and node.op == "add":
        l, rr = node.left, node.right
        if l.op == "mul" and rr.op == "mul" and _same_expr(l.right, rr.right):
            new_node = Expr.mul(Expr.add(l.left, rr.left), l.right)

    if new_node is None:
        return None
    return replace_subexpr(root, index, new_node)


@dataclass
class ProofGymConfig:
    max_depth: int = 5
    max_nodes: int = 31
    max_steps: int = 8
    min_trajectory_steps: int = 2
    prevent_expert_loops: bool = True
    vars: Tuple[str, ...] = ("x", "y", "z")
    consts: Tuple[int, ...] = (0, 1, 2, 3)


def random_expr(cfg: ProofGymConfig, depth: int = 0) -> Expr:
    if depth >= cfg.max_depth - 1:
        return Expr.var(random.choice(cfg.vars)) if random.random() < 0.6 else Expr.const(random.choice(cfg.consts))
    if random.random() < 0.35:
        return Expr.var(random.choice(cfg.vars)) if random.random() < 0.6 else Expr.const(random.choice(cfg.consts))
    left = random_expr(cfg, depth + 1)
    right = random_expr(cfg, depth + 1)
    return Expr.add(left, right) if random.random() < 0.5 else Expr.mul(left, right)


def list_valid_actions(expr: Expr, rules: List[Rule], cfg: ProofGymConfig) -> List[Tuple[int, int, Expr]]:
    out: List[Tuple[int, int, Expr]] = []
    for idx, _ in iter_nodes_with_indices(expr):
        if idx > cfg.max_nodes:
            continue
        for rid, r in enumerate(rules):
            nxt = apply_rule_at(expr, r, idx)
            if nxt is None:
                continue
            if nxt.depth() > cfg.max_depth or nxt.size() > cfg.max_nodes:
                continue
            out.append((rid, idx, nxt))
    return out


@dataclass
class Trajectory:
    start: Expr
    goal: Expr
    states: List[Expr]
    actions: List[Tuple[int, int]]


def generate_trajectory(cfg: ProofGymConfig, rules: List[Rule], max_tries: int = 1000) -> Trajectory:
    for _ in range(max_tries):
        start = random_expr(cfg)
        if start.size() > cfg.max_nodes or start.depth() > cfg.max_depth:
            continue
        states = [start]
        actions: List[Tuple[int, int]] = []
        cur = start
        seen = {start}
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


@dataclass
class Vocab:
    token_to_id: Dict[str, int]
    id_to_token: List[str]

    def encode(self, toks: Sequence[str]) -> List[int]:
        return [self.token_to_id[t] for t in toks]

    def decode(self, ids: Sequence[int]) -> List[str]:
        return [self.id_to_token[i] for i in ids]


def build_vocab(rules: List[Rule], cfg: ProofGymConfig) -> Tuple[Vocab, List[str]]:
    specials = ["<pad>", "<bos>"]
    structural = ["GOAL", "STATE", "ACT"]
    ops = ["ADD", "MUL"]
    vars_ = list(cfg.vars)
    consts = [str(c) for c in cfg.consts]
    action_tokens = []
    for rid, r in enumerate(rules):
        for idx in range(1, cfg.max_nodes + 1):
            action_tokens.append(f"A_{r.name}_{idx}")
    all_toks = specials + structural + ops + vars_ + consts + action_tokens
    return Vocab({t: i for i, t in enumerate(all_toks)}, all_toks), action_tokens


def action_to_token(rules: List[Rule], rule_id: int, node_index: int) -> str:
    return f"A_{rules[rule_id].name}_{node_index}"


def token_to_action(rules: List[Rule], tok: str) -> Optional[Tuple[int, int]]:
    if not tok.startswith("A_"):
        return None
    body = tok[2:]
    if "_" not in body:
        return None
    name, idx_s = body.rsplit("_", 1)
    try:
        idx = int(idx_s)
    except ValueError:
        return None
    for rid, r in enumerate(rules):
        if r.name == name:
            return rid, idx
    return None


def encode_transcript(tr: Trajectory, vocab: Vocab, rules: List[Rule], seq_len: int) -> Tuple[np.ndarray, np.ndarray]:
    toks: List[str] = ["<bos>", "GOAL", *tr.goal.to_tokens(), "STATE", *tr.start.to_tokens(), "ACT"]
    action_pos: List[int] = []

    for (rid, idx), st in zip(tr.actions, tr.states[1:]):
        action_pos.append(len(toks))
        toks.append(action_to_token(rules, rid, idx))
        toks.extend(["STATE", *st.to_tokens(), "ACT"])

    ids = vocab.encode(toks)
    x = np.full((seq_len,), vocab.token_to_id["<pad>"], dtype=np.int32)
    y = np.full((seq_len,), vocab.token_to_id["<pad>"], dtype=np.int32)
    mask = np.zeros((seq_len,), dtype=np.float32)

    L = min(len(ids) - 1, seq_len)
    if L > 0:
        x[:L] = np.array(ids[:L], dtype=np.int32)
        y[:L] = np.array(ids[1 : 1 + L], dtype=np.int32)

    for p in action_pos:
        target_idx = p
        if 0 <= target_idx < seq_len:
            mask[target_idx] = 1.0

    return np.stack([x, y], axis=0), mask


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

    def reset(self, lhs: Expr, rhs: Expr) -> ProofState:
        self.state = ProofState(lhs=lhs, rhs=rhs, step=0, done=False)
        return self.state

    def is_solved(self) -> bool:
        return self.state.lhs == self.state.rhs

    def step_action(self, action: Tuple[int, int]) -> Tuple[ProofState, float, bool, Dict[str, object]]:
        if self.state is None:
            raise RuntimeError("env not reset")

        rid, node_idx = action
        info: Dict[str, object] = {"valid": False}
        reward = -0.01

        if rid < 0 or rid >= len(self.rules):
            reward -= 0.1
        else:
            new_lhs = apply_rule_at(self.state.lhs, self.rules[rid], node_idx)
            if new_lhs is None:
                reward -= 0.1
            else:
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
