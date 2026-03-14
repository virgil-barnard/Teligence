#!/usr/bin/env python3
"""
experiments/icarus_projective_actionptr_v2.py

TensorFlow + ExplicitGPT version of the projective action-pointer prototype.

- Finite projective/affine geometry verifier-backed environment
- Policy over legal action lists via compositional action structs
- Value head predicting steps-to-go
- Greedy + best-first rollout evaluation
"""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import itertools
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import tensorflow as tf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from teligence.config import GPTConfig, validate_config
from teligence.modeling import ExplicitGPT, set_precision


@dataclass
class Config:
    p: int = 5
    task_mix: Tuple[str, ...] = (
        "parallel_through",
        "meet_two_lines",
        "parcomp_opposite_sides",
        "parallelogram_diagonals",
    )

    train_instances: int = 12000
    val_instances: int = 1500
    seed: int = 1337

    batch_size: int = 64
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2
    n_embd: int = 192
    dropout: float = 0.1

    max_action_args: int = 4

    max_iters: int = 4000
    eval_interval: int = 200
    eval_batches: int = 50
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_steps: int = 200
    weight_decay: float = 1e-2
    grad_clip: float = 1.0

    value_loss_weight: float = 0.2

    rollout_max_steps_slack: int = 3
    search_max_expansions: int = 256
    search_topk_actions: int = 5
    search_value_weight: float = 1.0

    use_bf16: bool = True
    use_flash_attn: bool = False
    flash_q_block: int = 128
    flash_k_block: int = 128

    out_dir: str = "runs/icarus_projective_v2"
    checkpoint_name: str = "icarus_projective_actionptr_v2"


CFG = Config()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def render_progress(tag: str, current: int, total: int, start_time: float, done: bool = False) -> None:
    total = max(1, total)
    current = max(0, min(current, total))
    frac = current / total
    width = 24
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    msg = f"[{tag}] [{bar}] {current:>6d}/{total:<6d} ({100.0 * frac:5.1f}%) {elapsed:6.1f}s"
    # Use newline output for stable logs under Docker/compose capture.
    print(msg, flush=True)


class SimpleTokenizer:
    SPECIAL = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<NL>"]

    def __init__(self, texts: Sequence[str]):
        vocab = set()
        for text in texts:
            vocab.update(self._split(text))
        vocab = self.SPECIAL + sorted(tok for tok in vocab if tok not in self.SPECIAL)
        self.stoi = {tok: i for i, tok in enumerate(vocab)}
        self.itos = {i: tok for tok, i in self.stoi.items()}
        self.pad_id = self.stoi["<PAD>"]
        self.unk_id = self.stoi["<UNK>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]

    @classmethod
    def from_stoi(cls, stoi_map: Dict[str, int]) -> "SimpleTokenizer":
        obj = cls([])
        obj.stoi = dict(stoi_map)
        obj.itos = {i: tok for tok, i in stoi_map.items()}
        obj.pad_id = obj.stoi["<PAD>"]
        obj.unk_id = obj.stoi["<UNK>"]
        obj.bos_id = obj.stoi["<BOS>"]
        obj.eos_id = obj.stoi["<EOS>"]
        return obj

    @staticmethod
    def _split(text: str) -> List[str]:
        return text.replace("\n", " <NL> ").split()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        toks = self._split(text)
        ids = [self.stoi.get(tok, self.unk_id) for tok in toks]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)


PointH = Tuple[int, int, int]
LineH = Tuple[int, int, int]


class ProjectiveEngine:
    def __init__(self, p: int):
        if p < 3:
            raise ValueError("Use an odd prime >= 3 for this prototype.")
        self.p = p
        self.line_inf: LineH = (0, 0, 1)

    def modinv(self, a: int) -> int:
        a %= self.p
        if a == 0:
            raise ZeroDivisionError("no inverse for zero")
        return pow(a, self.p - 2, self.p)

    def _canon_vec3(self, v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        x, y, z = (v[0] % self.p, v[1] % self.p, v[2] % self.p)
        if x == 0 and y == 0 and z == 0:
            raise ValueError("zero vector cannot be canonicalized")
        if x != 0:
            s = self.modinv(x)
        elif y != 0:
            s = self.modinv(y)
        else:
            s = self.modinv(z)
        return ((x * s) % self.p, (y * s) % self.p, (z * s) % self.p)

    def point_affine(self, x: int, y: int) -> PointH:
        # Keep affine chart coordinates explicit as z=1.
        return (x % self.p, y % self.p, 1)

    def rand_affine_point(self) -> PointH:
        return self.point_affine(random.randrange(self.p), random.randrange(self.p))

    def sample_distinct_affine_points(self, n: int) -> List[PointH]:
        pts = set()
        while len(pts) < n:
            pts.add(self.rand_affine_point())
        return list(pts)

    def cross(self, u: Tuple[int, int, int], v: Tuple[int, int, int]) -> Tuple[int, int, int]:
        ux, uy, uz = u
        vx, vy, vz = v
        return (
            (uy * vz - uz * vy) % self.p,
            (uz * vx - ux * vz) % self.p,
            (ux * vy - uy * vx) % self.p,
        )

    def dot(self, u: Tuple[int, int, int], v: Tuple[int, int, int]) -> int:
        return (u[0] * v[0] + u[1] * v[1] + u[2] * v[2]) % self.p

    def line_through(self, p1: PointH, p2: PointH) -> LineH:
        if p1 == p2:
            raise ValueError("need distinct points for a line")
        return self._canon_vec3(self.cross(p1, p2))

    def meet(self, l1: LineH, l2: LineH) -> PointH:
        if l1 == l2:
            raise ValueError("identical lines do not determine a unique meet point")
        return self._canon_vec3(self.cross(l1, l2))

    def inc(self, pt: PointH, line: LineH) -> bool:
        return self.dot(line, pt) % self.p == 0

    def col(self, p1: PointH, p2: PointH, p3: PointH) -> bool:
        return self.dot(p1, self.cross(p2, p3)) % self.p == 0

    def is_infinite_point(self, pt: PointH) -> bool:
        return (pt[2] % self.p) == 0

    def parallel(self, l1: LineH, l2: LineH) -> bool:
        if l1 == l2:
            return False
        try:
            m = self.meet(l1, l2)
        except ValueError:
            return True
        return self.is_infinite_point(m)

    def parallel_through(self, pt: PointH, line: LineH) -> LineH:
        dir_pt = self.meet(line, self.line_inf)
        return self.line_through(pt, dir_pt)

    def sample_point_off_line(self, line: LineH) -> PointH:
        while True:
            pt = self.rand_affine_point()
            if not self.inc(pt, line):
                return pt

    def sample_noncollinear_triple(self) -> Tuple[PointH, PointH, PointH]:
        while True:
            a, b, c = self.sample_distinct_affine_points(3)
            if not self.col(a, b, c):
                return a, b, c

    def sample_nonparallel_quad(self) -> Tuple[PointH, PointH, PointH, PointH]:
        while True:
            a, b, c, d = self.sample_distinct_affine_points(4)
            try:
                l1 = self.line_through(a, b)
                l2 = self.line_through(c, d)
            except ValueError:
                continue
            if not self.parallel(l1, l2):
                return a, b, c, d

    def parcomp_affine(self, a: PointH, b: PointH, c: PointH) -> PointH:
        ax, ay, az = a
        bx, by, bz = b
        cx, cy, cz = c
        if az == 0 or bz == 0 or cz == 0:
            raise ValueError("parcomp requires affine points (z!=0)")
        dx = (ax + cx - bx) % self.p
        dy = (ay + cy - by) % self.p
        return self.point_affine(dx, dy)


@dataclass
class TaskInstance:
    name: str
    goal_text: str
    initial_points: Dict[str, PointH]
    teacher_actions: List[str]


class TaskFactory:
    def __init__(self, engine: ProjectiveEngine):
        self.engine = engine

    def sample(self, task_name: str) -> TaskInstance:
        return getattr(self, f"_make_{task_name}")()

    def _make_parallel_through(self) -> TaskInstance:
        a, b = self.engine.sample_distinct_affine_points(2)
        l0 = self.engine.line_through(a, b)
        p = self.engine.sample_point_off_line(l0)
        points = {"A": a, "B": b, "P": p}
        actions = [
            "CONSTRUCT_LINE A B L0",
            "CONSTRUCT_PARALLEL_THROUGH P L0 L1",
            "ASSERT_INC P L1",
            "ASSERT_PARALLEL L0 L1",
            "STOP",
        ]
        return TaskInstance("parallel_through", "construct line through P parallel to AB", points, actions)

    def _make_meet_two_lines(self) -> TaskInstance:
        a, b, c, d = self.engine.sample_nonparallel_quad()
        points = {"A": a, "B": b, "C": c, "D": d}
        actions = [
            "CONSTRUCT_LINE A B L0",
            "CONSTRUCT_LINE C D L1",
            "CONSTRUCT_MEET L0 L1 X",
            "ASSERT_INC X L0",
            "ASSERT_INC X L1",
            "STOP",
        ]
        return TaskInstance("meet_two_lines", "intersect AB and CD and certify incidence", points, actions)

    def _make_parcomp_opposite_sides(self) -> TaskInstance:
        a, b, c = self.engine.sample_noncollinear_triple()
        points = {"A": a, "B": b, "C": c}
        actions = [
            "CONSTRUCT_PARCOMP A B C D",
            "CONSTRUCT_LINE A B L0",
            "CONSTRUCT_LINE C D L1",
            "ASSERT_PARALLEL L0 L1",
            "CONSTRUCT_LINE B C L2",
            "CONSTRUCT_LINE A D L3",
            "ASSERT_PARALLEL L2 L3",
            "STOP",
        ]
        return TaskInstance("parcomp_opposite_sides", "complete a parallelogram and prove opposite sides are parallel", points, actions)

    def _make_parallelogram_diagonals(self) -> TaskInstance:
        a, b, c = self.engine.sample_noncollinear_triple()
        points = {"A": a, "B": b, "C": c}
        actions = [
            "CONSTRUCT_PARCOMP A B C D",
            "CONSTRUCT_LINE A C L0",
            "CONSTRUCT_LINE B D L1",
            "CONSTRUCT_MEET L0 L1 M",
            "ASSERT_COL A C M",
            "ASSERT_COL B D M",
            "STOP",
        ]
        return TaskInstance("parallelogram_diagonals", "complete a parallelogram and certify diagonal intersection", points, actions)


class ProofEnv:
    def __init__(self, engine: ProjectiveEngine, task: TaskInstance):
        self.engine = engine
        self.task = task
        self.points: Dict[str, PointH] = dict(task.initial_points)
        self.lines: Dict[str, LineH] = {}
        self.facts = set()
        self.history: List[str] = []
        self.done = False

    def clone(self) -> "ProofEnv":
        other = ProofEnv(self.engine, self.task)
        other.points = dict(self.points)
        other.lines = dict(self.lines)
        other.facts = set(self.facts)
        other.history = list(self.history)
        other.done = self.done
        return other

    @property
    def step(self) -> int:
        return len(self.history)

    @staticmethod
    def canon_action(action: str) -> str:
        toks = action.strip().split()
        if not toks:
            return action
        op = toks[0]
        if op == "ASSERT_PARALLEL" and len(toks) == 3:
            a, b = sorted([toks[1], toks[2]])
            return f"{op} {a} {b}"
        if op == "CONSTRUCT_MEET" and len(toks) == 4:
            l1, l2 = sorted([toks[1], toks[2]])
            return f"{op} {l1} {l2} {toks[3]}"
        if op == "CONSTRUCT_LINE" and len(toks) == 4:
            p1, p2 = sorted([toks[1], toks[2]])
            return f"{op} {p1} {p2} {toks[3]}"
        if op == "ASSERT_COL" and len(toks) == 4:
            p1, p2, p3 = sorted([toks[1], toks[2], toks[3]])
            return f"{op} {p1} {p2} {p3}"
        return action

    def _canon_parallel_fact(self, l1: str, l2: str):
        a, b = sorted([l1, l2])
        return ("PAR", a, b)

    def _canon_col_fact(self, p1: str, p2: str, p3: str):
        a, b, c = sorted([p1, p2, p3])
        return ("COL", a, b, c)

    def success(self) -> bool:
        t = self.task.name
        if t == "parallel_through":
            return ("INC", "P", "L1") in self.facts and self._canon_parallel_fact("L0", "L1") in self.facts
        if t == "meet_two_lines":
            return ("INC", "X", "L0") in self.facts and ("INC", "X", "L1") in self.facts
        if t == "parcomp_opposite_sides":
            return self._canon_parallel_fact("L0", "L1") in self.facts and self._canon_parallel_fact("L2", "L3") in self.facts
        if t == "parallelogram_diagonals":
            return self._canon_col_fact("A", "C", "M") in self.facts and self._canon_col_fact("B", "D", "M") in self.facts
        raise ValueError(f"unknown task {t}")

    def serialize(self) -> str:
        out: List[str] = []
        out.append(f"TASK {self.task.name}")
        out.append(f"FIELD {self.engine.p}")
        out.append("OBJECTS")
        for name in sorted(self.points):
            x, y, z = self.points[name]
            out.append(f"POINT {name} {x} {y} {z}")
        for name in sorted(self.lines):
            a, b, c = self.lines[name]
            out.append(f"LINE {name} {a} {b} {c}")
        out.append("FACTS")
        for fact in sorted(self.facts):
            out.append(" ".join(map(str, fact)))
        out.append(f"GOAL {self.task.goal_text}")
        out.append("HISTORY")
        for h in self.history:
            out.append(h)
        out.append("ACTION")
        return "\n".join(out)

    def execute(self, action: str) -> bool:
        action = self.canon_action(action)
        toks = action.strip().split()
        if not toks:
            return False
        op = toks[0]
        try:
            if op == "CONSTRUCT_LINE":
                _, p1, p2, lname = toks
                if lname in self.lines or p1 not in self.points or p2 not in self.points or p1 == p2:
                    return False
                line = self.engine.line_through(self.points[p1], self.points[p2])
                if line in self.lines.values():
                    return False
                self.lines[lname] = line
            elif op == "CONSTRUCT_PARALLEL_THROUGH":
                _, pname, lsrc, lnew = toks
                if lnew in self.lines or pname not in self.points or lsrc not in self.lines:
                    return False
                line = self.engine.parallel_through(self.points[pname], self.lines[lsrc])
                if line in self.lines.values():
                    return False
                self.lines[lnew] = line
            elif op == "CONSTRUCT_MEET":
                _, l1, l2, pname = toks
                if pname in self.points or l1 not in self.lines or l2 not in self.lines:
                    return False
                self.points[pname] = self.engine.meet(self.lines[l1], self.lines[l2])
            elif op == "CONSTRUCT_PARCOMP":
                _, a, b, c, outp = toks
                if outp in self.points or a not in self.points or b not in self.points or c not in self.points:
                    return False
                self.points[outp] = self.engine.parcomp_affine(self.points[a], self.points[b], self.points[c])
            elif op == "ASSERT_INC":
                _, pname, lname = toks
                if pname not in self.points or lname not in self.lines:
                    return False
                if not self.engine.inc(self.points[pname], self.lines[lname]):
                    return False
                self.facts.add(("INC", pname, lname))
            elif op == "ASSERT_PARALLEL":
                _, l1, l2 = toks
                if l1 not in self.lines or l2 not in self.lines:
                    return False
                if self.lines[l1] == self.lines[l2]:
                    return False
                if not self.engine.parallel(self.lines[l1], self.lines[l2]):
                    return False
                self.facts.add(self._canon_parallel_fact(l1, l2))
            elif op == "ASSERT_COL":
                _, p1, p2, p3 = toks
                if p1 not in self.points or p2 not in self.points or p3 not in self.points:
                    return False
                if not self.engine.col(self.points[p1], self.points[p2], self.points[p3]):
                    return False
                self.facts.add(self._canon_col_fact(p1, p2, p3))
            elif op == "STOP":
                if not self.success():
                    return False
                self.done = True
            else:
                return False
        except Exception:
            return False

        self.history.append(action)
        return True

    def legal_actions(self) -> List[str]:
        stage = self.step
        task = self.task.name
        if task == "parallel_through":
            c = self._legal_parallel_through(stage)
        elif task == "meet_two_lines":
            c = self._legal_meet_two_lines(stage)
        elif task == "parcomp_opposite_sides":
            c = self._legal_parcomp_opposite_sides(stage)
        elif task == "parallelogram_diagonals":
            c = self._legal_parallelogram_diagonals(stage)
        else:
            raise ValueError(task)
        if self.success():
            c = list(c) + ["STOP"]
        out = [self.canon_action(a) for a in c]
        return sorted(set(out))

    def _legal_parallel_through(self, stage: int) -> List[str]:
        pt_names = sorted(self.points)
        line_names = sorted(self.lines)
        cands: List[str] = []
        if stage == 0:
            for p1, p2 in itertools.combinations(pt_names, 2):
                cands.append(f"CONSTRUCT_LINE {p1} {p2} L0")
        elif stage == 1:
            for pname in pt_names:
                cands.append(f"CONSTRUCT_PARALLEL_THROUGH {pname} L0 L1")
        elif stage == 2:
            for pname in pt_names:
                for lname in line_names:
                    cands.append(f"ASSERT_INC {pname} {lname}")
        elif stage == 3:
            for l1, l2 in itertools.combinations(line_names, 2):
                cands.append(f"ASSERT_PARALLEL {l1} {l2}")
        return cands

    def _legal_meet_two_lines(self, stage: int) -> List[str]:
        pt_names = sorted(self.points)
        line_names = sorted(self.lines)
        cands: List[str] = []
        if stage == 0:
            for p1, p2 in itertools.combinations(pt_names, 2):
                cands.append(f"CONSTRUCT_LINE {p1} {p2} L0")
        elif stage == 1:
            for p1, p2 in itertools.combinations(pt_names, 2):
                cands.append(f"CONSTRUCT_LINE {p1} {p2} L1")
        elif stage == 2:
            for l1, l2 in itertools.combinations(line_names, 2):
                cands.append(f"CONSTRUCT_MEET {l1} {l2} X")
        elif stage in (3, 4):
            for pname in sorted(self.points):
                for lname in line_names:
                    cands.append(f"ASSERT_INC {pname} {lname}")
        return cands

    def _legal_parcomp_opposite_sides(self, stage: int) -> List[str]:
        line_names = sorted(self.lines)
        cands: List[str] = []
        if stage == 0:
            for a, b, c in itertools.permutations(["A", "B", "C"], 3):
                cands.append(f"CONSTRUCT_PARCOMP {a} {b} {c} D")
        elif stage == 1:
            for p1, p2 in itertools.combinations(sorted(self.points), 2):
                cands.append(f"CONSTRUCT_LINE {p1} {p2} L0")
        elif stage == 2:
            for p1, p2 in itertools.combinations(sorted(self.points), 2):
                cands.append(f"CONSTRUCT_LINE {p1} {p2} L1")
        elif stage == 3:
            for l1, l2 in itertools.combinations(line_names, 2):
                cands.append(f"ASSERT_PARALLEL {l1} {l2}")
        elif stage == 4:
            for p1, p2 in itertools.combinations(sorted(self.points), 2):
                cands.append(f"CONSTRUCT_LINE {p1} {p2} L2")
        elif stage == 5:
            for p1, p2 in itertools.combinations(sorted(self.points), 2):
                cands.append(f"CONSTRUCT_LINE {p1} {p2} L3")
        elif stage == 6:
            for l1, l2 in itertools.combinations(line_names, 2):
                cands.append(f"ASSERT_PARALLEL {l1} {l2}")
        return cands

    def _legal_parallelogram_diagonals(self, stage: int) -> List[str]:
        line_names = sorted(self.lines)
        cands: List[str] = []
        if stage == 0:
            for a, b, c in itertools.permutations(["A", "B", "C"], 3):
                cands.append(f"CONSTRUCT_PARCOMP {a} {b} {c} D")
        elif stage == 1:
            for p1, p2 in itertools.combinations(sorted(self.points), 2):
                cands.append(f"CONSTRUCT_LINE {p1} {p2} L0")
        elif stage == 2:
            for p1, p2 in itertools.combinations(sorted(self.points), 2):
                cands.append(f"CONSTRUCT_LINE {p1} {p2} L1")
        elif stage == 3:
            for l1, l2 in itertools.combinations(line_names, 2):
                cands.append(f"CONSTRUCT_MEET {l1} {l2} M")
        elif stage in (4, 5):
            for p1, p2, p3 in itertools.combinations(sorted(self.points), 3):
                cands.append(f"ASSERT_COL {p1} {p2} {p3}")
        return cands


@dataclass
class DecisionRecord:
    state_text: str
    legal_actions: List[str]
    teacher_action: str
    steps_to_goal: int


def parse_action(action: str) -> Tuple[str, List[str]]:
    toks = action.strip().split()
    if not toks:
        return "", []
    return toks[0], toks[1:]


class ActionCodec:
    def __init__(self, ops: List[str], syms: List[str], max_args: int):
        self.max_args = max_args
        self.op_stoi = {op: i for i, op in enumerate(ops)}
        self.op_itos = {i: op for op, i in self.op_stoi.items()}
        self.sym_stoi = {s: i for i, s in enumerate(syms)}
        self.sym_itos = {i: s for s, i in self.sym_stoi.items()}
        self.pad_op = self.op_stoi["<PAD_OP>"]
        self.unk_op = self.op_stoi["<UNK_OP>"]
        self.pad_sym = self.sym_stoi["<PAD_ARG>"]
        self.unk_sym = self.sym_stoi["<UNK_ARG>"]

    def encode(self, action: str) -> List[int]:
        op, args = parse_action(action)
        op_id = self.op_stoi.get(op, self.unk_op)
        out = [op_id]
        for i in range(self.max_args):
            out.append(self.sym_stoi.get(args[i], self.unk_sym) if i < len(args) else self.pad_sym)
        return out

    @property
    def n_ops(self) -> int:
        return len(self.op_stoi)

    @property
    def n_syms(self) -> int:
        return len(self.sym_stoi)


class DecisionDataset:
    def __init__(self, states, legal_structs, legal_mask, targets, value_targets):
        self.states = states
        self.legal_structs = legal_structs
        self.legal_mask = legal_mask
        self.targets = targets
        self.value_targets = value_targets

    def sample_batch(self, batch_size: int):
        n = self.states.shape[0]
        idx = np.random.randint(0, n, size=(batch_size,))
        return (
            tf.constant(self.states[idx], dtype=tf.int32),
            tf.constant(self.legal_structs[idx], dtype=tf.int32),
            tf.constant(self.legal_mask[idx], dtype=tf.bool),
            tf.constant(self.targets[idx], dtype=tf.int32),
            tf.constant(self.value_targets[idx], dtype=tf.float32),
        )


def collect_decisions(engine: ProjectiveEngine, factory: TaskFactory, task_mix: Sequence[str], n_instances: int) -> List[DecisionRecord]:
    records: List[DecisionRecord] = []
    t0 = time.time()
    last_bucket = -1
    for inst_i in range(n_instances):
        task_name = random.choice(task_mix)
        task = factory.sample(task_name)
        env = ProofEnv(engine, task)
        n = len(task.teacher_actions)
        for i, act in enumerate(task.teacher_actions):
            act = env.canon_action(act)
            state = env.serialize()
            legal = [env.canon_action(a) for a in env.legal_actions()]
            if act not in legal:
                raise RuntimeError(f"Teacher action not in legal list. TASK={task.name} ACT={act}")
            records.append(DecisionRecord(state, legal, act, n - i))
            if not env.execute(act):
                raise RuntimeError(
                    "Teacher action rejected unexpectedly: "
                    f"task={task.name} step={i} action={act} state={state!r}"
                )
        bucket = int(((inst_i + 1) * 20) / max(1, n_instances))
        if bucket != last_bucket:
            render_progress("data_gen", inst_i + 1, n_instances, t0, done=(inst_i + 1) == n_instances)
            last_bucket = bucket
    return records


def build_datasets(cfg: Config):
    engine = ProjectiveEngine(cfg.p)
    factory = TaskFactory(engine)

    train_recs = collect_decisions(engine, factory, cfg.task_mix, cfg.train_instances)
    val_recs = collect_decisions(engine, factory, cfg.task_mix, cfg.val_instances)

    tokenizer = SimpleTokenizer([r.state_text for r in (train_recs + val_recs)])

    op_set = set(["STOP"])
    sym_set = set()
    for r in (train_recs + val_recs):
        for a in r.legal_actions:
            op, args = parse_action(a)
            op_set.add(op)
            sym_set.update(args)
    ops = ["<PAD_OP>", "<UNK_OP>"] + sorted(op_set)
    syms = ["<PAD_ARG>", "<UNK_ARG>"] + sorted(sym_set)
    codec = ActionCodec(ops=ops, syms=syms, max_args=cfg.max_action_args)

    all_states = [tokenizer.encode(r.state_text, add_bos=True) for r in (train_recs + val_recs)]
    max_T = max(len(x) for x in all_states)
    max_A = max(len(r.legal_actions) for r in (train_recs + val_recs))
    pad_struct = [codec.pad_op] + [codec.pad_sym] * cfg.max_action_args

    def encode_split(records: List[DecisionRecord]):
        xs: List[List[int]] = []
        legal_s: List[List[List[int]]] = []
        mask_s: List[List[bool]] = []
        tgt_s: List[int] = []
        val_s: List[float] = []
        t0 = time.time()
        last_bucket = -1
        total = len(records)
        for i, r in enumerate(records):
            x = tokenizer.encode(r.state_text, add_bos=True)
            x = x + [tokenizer.pad_id] * (max_T - len(x))

            leg = [codec.encode(a) for a in r.legal_actions]
            m = [True] * len(leg)
            if len(leg) < max_A:
                leg = leg + [pad_struct] * (max_A - len(leg))
                m = m + [False] * (max_A - len(m))

            t = r.legal_actions.index(r.teacher_action)

            xs.append(x)
            legal_s.append(leg)
            mask_s.append(m)
            tgt_s.append(t)
            val_s.append(float(r.steps_to_goal))

            bucket = int(((i + 1) * 20) / max(1, total))
            if bucket != last_bucket:
                render_progress("encode", i + 1, total, t0, done=(i + 1) == total)
                last_bucket = bucket

        return (
            np.asarray(xs, dtype=np.int32),
            np.asarray(legal_s, dtype=np.int32),
            np.asarray(mask_s, dtype=np.bool_),
            np.asarray(tgt_s, dtype=np.int32),
            np.asarray(val_s, dtype=np.float32),
        )

    tr = encode_split(train_recs)
    va = encode_split(val_recs)

    train_ds = DecisionDataset(*tr)
    val_ds = DecisionDataset(*va)
    meta = {
        "tokenizer": tokenizer,
        "codec": codec,
        "max_T": max_T,
        "max_A": max_A,
        "engine": engine,
        "factory": factory,
    }
    return train_ds, val_ds, meta


def make_backbone_cfg(cfg: Config, vocab_size: int, seq_len: int) -> GPTConfig:
    qblk = cfg.flash_q_block
    kblk = cfg.flash_k_block
    while qblk > 1 and seq_len % qblk != 0:
        qblk //= 2
    while kblk > 1 and seq_len % kblk != 0:
        kblk //= 2
    gcfg = GPTConfig(
        vocab_size=vocab_size,
        n_layer=cfg.n_layer,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        mlp_mult=4,
        seq_len=seq_len,
        attn_window=min(seq_len, 256),
        dropout=cfg.dropout,
        use_bf16=cfg.use_bf16,
        use_flash_attn=cfg.use_flash_attn,
        flash_q_block=qblk,
        flash_k_block=kblk,
        batch_size=cfg.batch_size,
        num_updates=cfg.max_iters,
        base_lr=cfg.learning_rate,
        min_lr=cfg.min_lr,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
    )
    validate_config(gcfg)
    return gcfg


class ActionPointerValueModel(tf.Module):
    def __init__(self, backbone_cfg: GPTConfig, pad_id: int, codec: ActionCodec, value_loss_weight: float):
        super().__init__()
        self.backbone_cfg = backbone_cfg
        self.pad_id = pad_id
        self.codec = codec
        self.value_loss_weight = value_loss_weight
        self.max_action_args = codec.max_args

        self.backbone = ExplicitGPT(backbone_cfg)
        self.op_emb = tf.keras.layers.Embedding(codec.n_ops, backbone_cfg.n_embd)
        self.arg_embs = [tf.keras.layers.Embedding(codec.n_syms, backbone_cfg.n_embd) for _ in range(codec.max_args)]
        self.key_norm = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.key_fc1 = tf.keras.layers.Dense(backbone_cfg.n_embd, activation=tf.nn.silu)
        self.key_fc2 = tf.keras.layers.Dense(backbone_cfg.n_embd)
        self.query = tf.keras.layers.Dense(backbone_cfg.n_embd, use_bias=False)
        self.value_head = tf.keras.layers.Dense(1)

    def encode_state(self, idx: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.backbone.wte(idx)
        x = tf.cast(x, tf.bfloat16 if self.backbone_cfg.use_bf16 else tf.float32)
        x = self.backbone.emb_norm(x)
        for blk in self.backbone.blocks:
            x = blk(x, training=training)
        x = self.backbone.final_norm(x)

        not_pad = tf.cast(tf.not_equal(idx, self.pad_id), tf.int32)
        lengths = tf.reduce_sum(not_pad, axis=1)
        last = tf.maximum(lengths - 1, 0)
        b = tf.range(tf.shape(idx)[0], dtype=tf.int32)
        gather_idx = tf.stack([b, last], axis=1)
        s = tf.gather_nd(tf.cast(x, tf.float32), gather_idx)
        return s

    def embed_action_structs(self, structs: tf.Tensor) -> tf.Tensor:
        op_ids = structs[:, :, 0]
        arg_ids = structs[:, :, 1:]
        key = self.op_emb(op_ids)
        for i in range(self.max_action_args):
            key = key + self.arg_embs[i](arg_ids[:, :, i])
        key = self.key_norm(key)
        key = self.key_fc2(self.key_fc1(key))
        return key

    def __call__(
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

        logits = tf.reduce_sum(keys * q[:, None, :], axis=-1) * (1.0 / math.sqrt(self.backbone_cfg.n_embd))
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


def lr_schedule(step: int, cfg: Config) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    t = (step - cfg.warmup_steps) / max(1, (cfg.max_iters - cfg.warmup_steps))
    t = min(1.0, max(0.0, t))
    cos = 0.5 * (1.0 + math.cos(math.pi * t))
    return cfg.min_lr + (cfg.learning_rate - cfg.min_lr) * cos


def apply_weight_decay(vars_: Sequence[tf.Variable], lr: float, wd: float) -> None:
    if wd <= 0.0:
        return
    for v in vars_:
        if v.shape.rank is not None and v.shape.rank >= 2:
            v.assign_sub(tf.cast(lr * wd, v.dtype) * v)


@tf.function(jit_compile=False, reduce_retracing=True)
def train_step(model, opt, x, ls, lm, y, v, grad_clip):
    with tf.GradientTape() as tape:
        _, _, loss, _, _ = model(x, ls, lm, targets=y, value_targets=v, training=True)
    grads = tape.gradient(loss, model.trainable_variables)
    grads_vars = [(g, var) for g, var in zip(grads, model.trainable_variables) if g is not None]
    if grads_vars:
        gvals = [g for g, _ in grads_vars]
        gvals, _ = tf.clip_by_global_norm(gvals, grad_clip)
        opt.apply_gradients(zip(gvals, [var for _, var in grads_vars]))
    return loss


@tf.function(jit_compile=False, reduce_retracing=True)
def eval_step(model, x, ls, lm, y, v):
    logits, value, _, pl, vl = model(x, ls, lm, targets=y, value_targets=v, training=False)
    pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))
    return pl, vl, acc


def estimate_metrics(model: ActionPointerValueModel, train_ds: DecisionDataset, val_ds: DecisionDataset, cfg: Config):
    out = {}
    for split, ds in [("train", train_ds), ("val", val_ds)]:
        policy_losses, value_losses, accs = [], [], []
        t0 = time.time()
        for i in range(cfg.eval_batches):
            x, ls, lm, y, v = ds.sample_batch(cfg.batch_size)
            pl, vl, acc = eval_step(model, x, ls, lm, y, v)
            policy_losses.append(float(pl.numpy()))
            value_losses.append(float(vl.numpy()))
            accs.append(float(acc.numpy()))
            if (i + 1) % max(1, cfg.eval_batches // 10) == 0 or (i + 1) == cfg.eval_batches:
                render_progress(f"eval:{split}", i + 1, cfg.eval_batches, t0, done=(i + 1) == cfg.eval_batches)
        out[split] = {
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "value_mse": sum(value_losses) / len(value_losses),
            "acc": sum(accs) / len(accs),
        }
    return out


def _encode_state_for_infer(tok: SimpleTokenizer, state_text: str, max_T: int) -> List[int]:
    ids = tok.encode(state_text, add_bos=True)
    ids = ids[:max_T]
    if len(ids) < max_T:
        ids = ids + [tok.pad_id] * (max_T - len(ids))
    return ids


def _encode_legal_for_infer(codec: ActionCodec, legal_actions: List[str], max_A: int, max_args: int):
    pad_struct = [codec.pad_op] + [codec.pad_sym] * max_args
    structs = [codec.encode(a) for a in legal_actions]
    mask = [True] * len(structs)
    if len(structs) < max_A:
        structs += [pad_struct] * (max_A - len(structs))
        mask += [False] * (max_A - len(mask))
    else:
        structs = structs[:max_A]
        mask = mask[:max_A]
    return structs, mask


def policy_value(model, tok, codec, env, max_T, max_A):
    legal = env.legal_actions()
    x = _encode_state_for_infer(tok, env.serialize(), max_T)
    structs, mask = _encode_legal_for_infer(codec, legal, max_A, model.max_action_args)

    X = tf.constant([x], dtype=tf.int32)
    LS = tf.constant([structs], dtype=tf.int32)
    LM = tf.constant([mask], dtype=tf.bool)
    logits, value, _, _, _ = model(X, LS, LM, training=False)
    n = min(len(legal), max_A)
    return legal[:n], tf.cast(logits[0, :n], tf.float32), float(value.numpy()[0])


def rollout_greedy(model, tok, codec, engine, task, max_T, max_A, verbose):
    env = ProofEnv(engine, task)
    max_steps = len(task.teacher_actions) + CFG.rollout_max_steps_slack
    for _ in range(max_steps):
        legal, logits, v = policy_value(model, tok, codec, env, max_T, max_A)
        probs = tf.nn.softmax(logits)
        idx = int(tf.argmax(probs).numpy())
        action = legal[idx]
        ok = env.execute(action)
        if verbose:
            print(f"V~{v:.2f} | CHOSEN: {action}")
        if not ok:
            return False
        if env.done and env.success():
            return True
    return False


def rollout_greedy_trace(model, tok, codec, engine, task, max_T, max_A, verbose):
    env = ProofEnv(engine, task)
    max_steps = len(task.teacher_actions) + CFG.rollout_max_steps_slack
    for _ in range(max_steps):
        legal, logits, v = policy_value(model, tok, codec, env, max_T, max_A)
        probs = tf.nn.softmax(logits)
        idx = int(tf.argmax(probs).numpy())
        action = legal[idx]
        ok = env.execute(action)
        if verbose:
            print(f"V~{v:.2f} | CHOSEN: {action}")
        if not ok:
            return False, env
        if env.done and env.success():
            return True, env
    return False, env


@dataclass(order=True)
class _Node:
    priority: float
    neglogp: float
    depth: int
    env: ProofEnv


def rollout_best_first(model, tok, codec, engine, task, max_T, max_A, cfg: Config, verbose):
    root = ProofEnv(engine, task)
    max_depth = len(task.teacher_actions) + cfg.rollout_max_steps_slack

    _, _, v0 = policy_value(model, tok, codec, root, max_T, max_A)
    heap: List[_Node] = []
    heapq.heappush(heap, _Node(priority=cfg.search_value_weight * v0, neglogp=0.0, depth=0, env=root))
    expansions = 0

    while heap and expansions < cfg.search_max_expansions:
        node = heapq.heappop(heap)
        env = node.env
        if env.done and env.success():
            return True
        if node.depth >= max_depth:
            continue

        legal, logits, v = policy_value(model, tok, codec, env, max_T, max_A)
        logp = tf.nn.log_softmax(logits)
        k = min(cfg.search_topk_actions, int(logp.shape[0]))
        topi = tf.math.top_k(logp, k=k).indices.numpy().tolist()
        topv = tf.math.top_k(logp, k=k).values.numpy().tolist()
        expansions += 1
        if verbose and expansions % 25 == 0:
            print(f"exp {expansions:4d} depth {node.depth:2d} neglogp {node.neglogp:.2f} V~{v:.2f} heap={len(heap)}")

        child_h = max(0.0, v - 1.0)
        for lp, idx in zip(topv, topi):
            action = legal[int(idx)]
            child = env.clone()
            if not child.execute(action):
                continue
            neglogp = node.neglogp + float(-lp)
            priority = neglogp + cfg.search_value_weight * child_h
            heapq.heappush(heap, _Node(priority=priority, neglogp=neglogp, depth=node.depth + 1, env=child))
    return False


def rollout_best_first_trace(model, tok, codec, engine, task, max_T, max_A, cfg: Config, verbose):
    root = ProofEnv(engine, task)
    max_depth = len(task.teacher_actions) + cfg.rollout_max_steps_slack

    _, _, v0 = policy_value(model, tok, codec, root, max_T, max_A)
    heap: List[_Node] = []
    heapq.heappush(heap, _Node(priority=cfg.search_value_weight * v0, neglogp=0.0, depth=0, env=root))
    expansions = 0
    best_env = root
    best_depth = 0

    while heap and expansions < cfg.search_max_expansions:
        node = heapq.heappop(heap)
        env = node.env
        depth_here = len(env.history)
        if depth_here > best_depth:
            best_depth = depth_here
            best_env = env
        if env.done and env.success():
            return True, env
        if node.depth >= max_depth:
            continue

        legal, logits, v = policy_value(model, tok, codec, env, max_T, max_A)
        logp = tf.nn.log_softmax(logits)
        k = min(cfg.search_topk_actions, int(logp.shape[0]))
        topi = tf.math.top_k(logp, k=k).indices.numpy().tolist()
        topv = tf.math.top_k(logp, k=k).values.numpy().tolist()
        expansions += 1
        if verbose and expansions % 25 == 0:
            print(f"exp {expansions:4d} depth {node.depth:2d} neglogp {node.neglogp:.2f} V~{v:.2f} heap={len(heap)}")

        child_h = max(0.0, v - 1.0)
        for lp, idx in zip(topv, topi):
            action = legal[int(idx)]
            child = env.clone()
            if not child.execute(action):
                continue
            child_depth = len(child.history)
            if child_depth > best_depth:
                best_depth = child_depth
                best_env = child
            neglogp = node.neglogp + float(-lp)
            priority = neglogp + cfg.search_value_weight * child_h
            heapq.heappush(heap, _Node(priority=priority, neglogp=neglogp, depth=node.depth + 1, env=child))
    return False, best_env


def train(cfg: Config, resume_checkpoint: str = ""):
    set_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)

    train_ds, val_ds, meta = build_datasets(cfg)
    tok: SimpleTokenizer = meta["tokenizer"]
    codec: ActionCodec = meta["codec"]
    max_T: int = meta["max_T"]
    max_A: int = meta["max_A"]
    engine: ProjectiveEngine = meta["engine"]
    factory: TaskFactory = meta["factory"]

    print(f"vocab_size={tok.vocab_size} max_T={max_T} max_A={max_A} ops={codec.n_ops} syms={codec.n_syms}")
    print(f"train_decisions={train_ds.states.shape[0]} val_decisions={val_ds.states.shape[0]}")

    backbone_cfg = make_backbone_cfg(cfg, tok.vocab_size, max_T)
    set_precision(backbone_cfg.use_bf16)
    model = ActionPointerValueModel(backbone_cfg, tok.pad_id, codec, cfg.value_loss_weight)

    dummy_x = tf.zeros([1, max_T], dtype=tf.int32)
    dummy_ls = tf.zeros([1, max_A, 1 + cfg.max_action_args], dtype=tf.int32)
    dummy_lm = tf.zeros([1, max_A], dtype=tf.bool)
    _ = model(dummy_x, dummy_ls, dummy_lm, training=False)

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate, beta_1=0.9, beta_2=0.95, epsilon=1e-8)
    step_var = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False)
    ckpt = tf.train.Checkpoint(model=model, step=step_var)
    last_mgr = tf.train.CheckpointManager(ckpt, os.path.join(cfg.out_dir, "ckpt_last"), max_to_keep=3)
    best_mgr = tf.train.CheckpointManager(ckpt, os.path.join(cfg.out_dir, "ckpt_best"), max_to_keep=1)

    restore_path = resume_checkpoint.strip() if resume_checkpoint else ""
    if not restore_path:
        restore_path = last_mgr.latest_checkpoint or ""
    start_it = 0
    if restore_path:
        ckpt.restore(restore_path).expect_partial()
        start_it = int(step_var.numpy()) + 1
        print(f"restored checkpoint: {restore_path} (step={int(step_var.numpy())})")
        if start_it > cfg.max_iters:
            print(f"checkpoint step {start_it-1} already >= max_iters {cfg.max_iters}; nothing to train")
    else:
        print("no checkpoint found, starting fresh")

    best_val = float("inf")
    t0 = time.time()
    grad_clip_t = tf.constant(cfg.grad_clip, dtype=tf.float32)
    for it in range(start_it, cfg.max_iters + 1):
        step_var.assign(it)
        if it % cfg.eval_interval == 0:
            stats = estimate_metrics(model, train_ds, val_ds, cfg)
            dt = time.time() - t0
            tr = stats["train"]
            va = stats["val"]
            print(
                f"step {it:5d} | train pl {tr['policy_loss']:.4f} acc {tr['acc']:.3f} vMSE {tr['value_mse']:.3f} | "
                f"val pl {va['policy_loss']:.4f} acc {va['acc']:.3f} vMSE {va['value_mse']:.3f} | {dt:.1f}s"
            )
            if va["policy_loss"] < best_val:
                best_val = va["policy_loss"]
                path = best_mgr.save(checkpoint_number=it)
                print(f"saved best checkpoint: {path}")
            last_mgr.save(checkpoint_number=it)

        x, ls, lm, y, v = train_ds.sample_batch(cfg.batch_size)
        lr = lr_schedule(it, cfg)
        if isinstance(opt.learning_rate, tf.Variable):
            opt.learning_rate.assign(lr)
        else:
            opt.learning_rate = lr
        loss = train_step(model, opt, x, ls, lm, y, v, grad_clip_t)
        apply_weight_decay(model.trainable_variables, lr, cfg.weight_decay)
        if it % 100 == 0:
            print(f"iter {it:5d} | loss {float(loss.numpy()):.4f} | lr {lr:.2e}")
        if (it + 1) % max(1, cfg.max_iters // 20) == 0 or it == cfg.max_iters:
            render_progress("train", it + 1, cfg.max_iters + 1, t0, done=(it == cfg.max_iters))

    meta_path = os.path.join(cfg.out_dir, f"{cfg.checkpoint_name}.meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "cfg": dataclasses.asdict(cfg),
                "tokenizer_stoi": tok.stoi,
                "codec_ops": list(codec.op_stoi.keys()),
                "codec_syms": list(codec.sym_stoi.keys()),
                "max_T": max_T,
                "max_A": max_A,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    print(f"wrote meta: {meta_path}")

    print("\n--- quick rollouts after training ---")
    successes = 0
    hardest_task = None
    hardest_env = None
    hardest_steps = -1
    task_stats = {name: {"attempts": 0, "solved": 0, "steps_all": 0, "steps_solved": 0} for name in cfg.task_mix}
    quick_rollouts = 10
    quick_t0 = time.time()
    for i in range(quick_rollouts):
        task = factory.sample(random.choice(cfg.task_mix))
        ok, env = rollout_best_first_trace(model, tok, codec, engine, task, max_T, max_A, cfg, verbose=False)
        successes += int(ok)
        steps = len(env.history) if env is not None else 0
        rec = task_stats[task.name]
        rec["attempts"] += 1
        rec["steps_all"] += steps
        if ok:
            rec["solved"] += 1
            rec["steps_solved"] += steps
        print(f"[quick rollout] {i+1}/{quick_rollouts} task={task.name} solved={ok} steps={steps}")
        render_progress("quick_rollout", i + 1, quick_rollouts, quick_t0, done=(i + 1) == quick_rollouts)
        if ok and env is not None and len(env.history) > hardest_steps:
            hardest_steps = len(env.history)
            hardest_task = task
            hardest_env = env
    print(f"search success@10 = {successes}/10")

    print("\n--- quick rollout synopsis by task ---")
    for task_name in sorted(task_stats):
        rec = task_stats[task_name]
        attempts = rec["attempts"]
        if attempts == 0:
            continue
        solved = rec["solved"]
        solve_rate = solved / attempts
        avg_steps_all = rec["steps_all"] / attempts
        avg_steps_solved = (rec["steps_solved"] / solved) if solved > 0 else float("nan")
        solved_steps = f"{avg_steps_solved:.2f}" if solved > 0 else "-"
        print(
            f"{task_name:26s} attempts={attempts:2d} solved={solved:2d} "
            f"solve_rate={solve_rate:.2f} avg_steps_all={avg_steps_all:.2f} avg_steps_solved={solved_steps}"
        )

    if hardest_task is not None and hardest_env is not None:
        print("\n--- hardest solved rollout ---")
        print(f"task: {hardest_task.name} | steps: {hardest_steps}")
        print(f"goal: {hardest_task.goal_text}")
        print("initial points:")
        for name in sorted(hardest_task.initial_points):
            x, y, z = hardest_task.initial_points[name]
            print(f"  {name}: ({x},{y},{z})")
        print("actions:")
        for i, act in enumerate(hardest_env.history, start=1):
            print(f"  {i:2d}. {act}")

    return best_mgr.latest_checkpoint or last_mgr.latest_checkpoint, meta_path


def load_checkpoint(out_dir: str, checkpoint: str, device: str):
    del device
    meta_path = os.path.join(out_dir, f"{CFG.checkpoint_name}.meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    cfg = Config(**payload["cfg"])
    tok = SimpleTokenizer.from_stoi(payload["tokenizer_stoi"])
    codec = ActionCodec(payload["codec_ops"], payload["codec_syms"], cfg.max_action_args)
    max_T = int(payload["max_T"])
    max_A = int(payload["max_A"])

    backbone_cfg = make_backbone_cfg(cfg, tok.vocab_size, max_T)
    set_precision(backbone_cfg.use_bf16)
    model = ActionPointerValueModel(backbone_cfg, tok.pad_id, codec, cfg.value_loss_weight)

    dummy_x = tf.zeros([1, max_T], dtype=tf.int32)
    dummy_ls = tf.zeros([1, max_A, 1 + cfg.max_action_args], dtype=tf.int32)
    dummy_lm = tf.zeros([1, max_A], dtype=tf.bool)
    _ = model(dummy_x, dummy_ls, dummy_lm, training=False)

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate)
    step_var = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False)
    ckpt = tf.train.Checkpoint(model=model, step=step_var)
    ckpt.restore(checkpoint).expect_partial()

    engine = ProjectiveEngine(cfg.p)
    factory = TaskFactory(engine)
    return cfg, tok, codec, model, engine, factory, max_T, max_A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "rollout"], default="train")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--rollout_mode", choices=["greedy", "search"], default="search")
    p.add_argument("--demo_tasks", type=int, default=5)
    p.add_argument("--train_instances", type=int, default=CFG.train_instances)
    p.add_argument("--val_instances", type=int, default=CFG.val_instances)
    p.add_argument("--max_iters", type=int, default=CFG.max_iters)
    p.add_argument("--eval_interval", type=int, default=CFG.eval_interval)
    p.add_argument("--batch_size", type=int, default=CFG.batch_size)
    p.add_argument("--eval_batches", type=int, default=CFG.eval_batches)
    p.add_argument("--search_max_expansions", type=int, default=CFG.search_max_expansions)
    p.add_argument("--search_topk_actions", type=int, default=CFG.search_topk_actions)
    p.add_argument("--search_value_weight", type=float, default=CFG.search_value_weight)
    p.add_argument("--out_dir", type=str, default=CFG.out_dir)
    p.add_argument("--resume_checkpoint", type=str, default="")
    args = p.parse_args()

    if args.quick:
        # Apply fast sanity-run defaults when corresponding flags are untouched.
        if args.train_instances == CFG.train_instances:
            args.train_instances = 1000
        if args.val_instances == CFG.val_instances:
            args.val_instances = 200
        if args.max_iters == CFG.max_iters:
            args.max_iters = 200
        if args.eval_interval == CFG.eval_interval:
            args.eval_interval = 100
        if args.eval_batches == CFG.eval_batches:
            args.eval_batches = 10
        if args.batch_size == CFG.batch_size:
            args.batch_size = 32
        if args.demo_tasks == 5:
            args.demo_tasks = 3
        if args.search_max_expansions == CFG.search_max_expansions:
            args.search_max_expansions = 128
        if args.search_topk_actions == CFG.search_topk_actions:
            args.search_topk_actions = 3
        print("[quick] enabled fast sanity defaults")

    cfg = Config()
    cfg.train_instances = args.train_instances
    cfg.val_instances = args.val_instances
    cfg.max_iters = args.max_iters
    cfg.eval_interval = args.eval_interval
    cfg.batch_size = args.batch_size
    cfg.eval_batches = args.eval_batches
    cfg.search_max_expansions = args.search_max_expansions
    cfg.search_topk_actions = args.search_topk_actions
    cfg.search_value_weight = args.search_value_weight
    cfg.out_dir = args.out_dir

    if args.mode == "train":
        ckpt, meta = train(cfg, resume_checkpoint=args.resume_checkpoint)
        print(f"checkpoint: {ckpt}")
        print(f"meta: {meta}")
    else:
        checkpoint = args.checkpoint
        if not checkpoint:
            mgr = tf.train.CheckpointManager(tf.train.Checkpoint(), os.path.join(cfg.out_dir, "ckpt_best"), max_to_keep=1)
            checkpoint = mgr.latest_checkpoint
            if checkpoint is None:
                mgr = tf.train.CheckpointManager(tf.train.Checkpoint(), os.path.join(cfg.out_dir, "ckpt_last"), max_to_keep=1)
                checkpoint = mgr.latest_checkpoint
        if checkpoint is None:
            raise ValueError("No checkpoint found. Provide --checkpoint or train first.")

        cfg2, tok, codec, model, engine, factory, max_T, max_A = load_checkpoint(cfg.out_dir, checkpoint, "cpu")

        successes = 0
        task_stats = {name: {"attempts": 0, "solved": 0} for name in cfg2.task_mix}
        for _ in range(args.demo_tasks):
            task = factory.sample(random.choice(cfg2.task_mix))
            if args.rollout_mode == "greedy":
                ok = rollout_greedy(model, tok, codec, engine, task, max_T, max_A, verbose=True)
            else:
                ok = rollout_best_first(model, tok, codec, engine, task, max_T, max_A, cfg2, verbose=True)
            successes += int(ok)
            rec = task_stats[task.name]
            rec["attempts"] += 1
            rec["solved"] += int(ok)
        print(f"{args.rollout_mode} success rate: {successes}/{args.demo_tasks}")
        print("rollout synopsis by task:")
        for task_name in sorted(task_stats):
            rec = task_stats[task_name]
            if rec["attempts"] == 0:
                continue
            print(
                f"  {task_name:26s} solved={rec['solved']:2d}/{rec['attempts']:2d} "
                f"({rec['solved'] / rec['attempts']:.2f})"
            )


if __name__ == "__main__":
    main()
