import itertools
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import tensorflow as tf


def render_progress(tag: str, current: int, total: int, start_time: float, done: bool = False) -> None:
    del done
    total = max(1, total)
    current = max(0, min(current, total))
    frac = current / total
    width = 24
    filled = int(width * frac)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = time.time() - start_time
    print(f"[{tag}] [{bar}] {current:>6d}/{total:<6d} ({100.0 * frac:5.1f}%) {elapsed:6.1f}s", flush=True)


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
        ids = [self.stoi.get(tok, self.unk_id) for tok in self._split(text)]
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
        return ((uy * vz - uz * vy) % self.p, (uz * vx - ux * vz) % self.p, (ux * vy - uy * vx) % self.p)

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
        return self.line_through(pt, self.meet(line, self.line_inf))

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
        return self.point_affine((ax + cx - bx) % self.p, (ay + cy - by) % self.p)


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
        return TaskInstance("parallel_through", "construct line through P parallel to AB", {"A": a, "B": b, "P": p}, [
            "CONSTRUCT_LINE A B L0", "CONSTRUCT_PARALLEL_THROUGH P L0 L1", "ASSERT_INC P L1", "ASSERT_PARALLEL L0 L1", "STOP",
        ])

    def _make_meet_two_lines(self) -> TaskInstance:
        a, b, c, d = self.engine.sample_nonparallel_quad()
        return TaskInstance("meet_two_lines", "intersect AB and CD and certify incidence", {"A": a, "B": b, "C": c, "D": d}, [
            "CONSTRUCT_LINE A B L0", "CONSTRUCT_LINE C D L1", "CONSTRUCT_MEET L0 L1 X", "ASSERT_INC X L0", "ASSERT_INC X L1", "STOP",
        ])

    def _make_parcomp_opposite_sides(self) -> TaskInstance:
        a, b, c = self.engine.sample_noncollinear_triple()
        return TaskInstance("parcomp_opposite_sides", "complete a parallelogram and prove opposite sides are parallel", {"A": a, "B": b, "C": c}, [
            "CONSTRUCT_PARCOMP A B C D", "CONSTRUCT_LINE A B L0", "CONSTRUCT_LINE C D L1", "ASSERT_PARALLEL L0 L1", "CONSTRUCT_LINE B C L2", "CONSTRUCT_LINE A D L3", "ASSERT_PARALLEL L2 L3", "STOP",
        ])

    def _make_parallelogram_diagonals(self) -> TaskInstance:
        a, b, c = self.engine.sample_noncollinear_triple()
        return TaskInstance("parallelogram_diagonals", "complete a parallelogram and certify diagonal intersection", {"A": a, "B": b, "C": c}, [
            "CONSTRUCT_PARCOMP A B C D", "CONSTRUCT_LINE A C L0", "CONSTRUCT_LINE B D L1", "CONSTRUCT_MEET L0 L1 M", "ASSERT_COL A C M", "ASSERT_COL B D M", "STOP",
        ])


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
        out: List[str] = [f"TASK {self.task.name}", f"FIELD {self.engine.p}", "OBJECTS"]
        for name in sorted(self.points):
            x, y, z = self.points[name]
            out.append(f"POINT {name} {x} {y} {z}")
        for name in sorted(self.lines):
            a, b, c = self.lines[name]
            out.append(f"LINE {name} {a} {b} {c}")
        out.append("FACTS")
        for fact in sorted(self.facts):
            out.append(" ".join(map(str, fact)))
        out.extend([f"GOAL {self.task.goal_text}", "HISTORY", *self.history, "ACTION"])
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
                if pname not in self.points or lname not in self.lines or not self.engine.inc(self.points[pname], self.lines[lname]):
                    return False
                self.facts.add(("INC", pname, lname))
            elif op == "ASSERT_PARALLEL":
                _, l1, l2 = toks
                if l1 not in self.lines or l2 not in self.lines or self.lines[l1] == self.lines[l2]:
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
        return sorted(set(self.canon_action(a) for a in c))

    def _legal_parallel_through(self, stage: int) -> List[str]:
        pt_names = sorted(self.points)
        line_names = sorted(self.lines)
        if stage == 0:
            return [f"CONSTRUCT_LINE {p1} {p2} L0" for p1, p2 in itertools.combinations(pt_names, 2)]
        if stage == 1:
            return [f"CONSTRUCT_PARALLEL_THROUGH {pname} L0 L1" for pname in pt_names]
        if stage == 2:
            return [f"ASSERT_INC {pname} {lname}" for pname in pt_names for lname in line_names]
        if stage == 3:
            return [f"ASSERT_PARALLEL {l1} {l2}" for l1, l2 in itertools.combinations(line_names, 2)]
        return []

    def _legal_meet_two_lines(self, stage: int) -> List[str]:
        pt_names = sorted(self.points)
        line_names = sorted(self.lines)
        if stage == 0:
            return [f"CONSTRUCT_LINE {p1} {p2} L0" for p1, p2 in itertools.combinations(pt_names, 2)]
        if stage == 1:
            return [f"CONSTRUCT_LINE {p1} {p2} L1" for p1, p2 in itertools.combinations(pt_names, 2)]
        if stage == 2:
            return [f"CONSTRUCT_MEET {l1} {l2} X" for l1, l2 in itertools.combinations(line_names, 2)]
        if stage in (3, 4):
            return [f"ASSERT_INC {pname} {lname}" for pname in sorted(self.points) for lname in line_names]
        return []

    def _legal_parcomp_opposite_sides(self, stage: int) -> List[str]:
        line_names = sorted(self.lines)
        if stage == 0:
            return [f"CONSTRUCT_PARCOMP {a} {b} {c} D" for a, b, c in itertools.permutations(["A", "B", "C"], 3)]
        if stage == 1:
            return [f"CONSTRUCT_LINE {p1} {p2} L0" for p1, p2 in itertools.combinations(sorted(self.points), 2)]
        if stage == 2:
            return [f"CONSTRUCT_LINE {p1} {p2} L1" for p1, p2 in itertools.combinations(sorted(self.points), 2)]
        if stage == 3:
            return [f"ASSERT_PARALLEL {l1} {l2}" for l1, l2 in itertools.combinations(line_names, 2)]
        if stage == 4:
            return [f"CONSTRUCT_LINE {p1} {p2} L2" for p1, p2 in itertools.combinations(sorted(self.points), 2)]
        if stage == 5:
            return [f"CONSTRUCT_LINE {p1} {p2} L3" for p1, p2 in itertools.combinations(sorted(self.points), 2)]
        if stage == 6:
            return [f"ASSERT_PARALLEL {l1} {l2}" for l1, l2 in itertools.combinations(line_names, 2)]
        return []

    def _legal_parallelogram_diagonals(self, stage: int) -> List[str]:
        line_names = sorted(self.lines)
        if stage == 0:
            return [f"CONSTRUCT_PARCOMP {a} {b} {c} D" for a, b, c in itertools.permutations(["A", "B", "C"], 3)]
        if stage == 1:
            return [f"CONSTRUCT_LINE {p1} {p2} L0" for p1, p2 in itertools.combinations(sorted(self.points), 2)]
        if stage == 2:
            return [f"CONSTRUCT_LINE {p1} {p2} L1" for p1, p2 in itertools.combinations(sorted(self.points), 2)]
        if stage == 3:
            return [f"CONSTRUCT_MEET {l1} {l2} M" for l1, l2 in itertools.combinations(line_names, 2)]
        if stage in (4, 5):
            return [f"ASSERT_COL {p1} {p2} {p3}" for p1, p2, p3 in itertools.combinations(sorted(self.points), 3)]
        return []


@dataclass
class DecisionRecord:
    state_text: str
    legal_actions: List[str]
    teacher_action: str
    steps_to_goal: int


def parse_action(action: str) -> Tuple[str, List[str]]:
    toks = action.strip().split()
    return ("", []) if not toks else (toks[0], toks[1:])


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
        out = [self.op_stoi.get(op, self.unk_op)]
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
        task = factory.sample(random.choice(task_mix))
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
                raise RuntimeError(f"Teacher action rejected unexpectedly: task={task.name} step={i} action={act}")
        bucket = int(((inst_i + 1) * 20) / max(1, n_instances))
        if bucket != last_bucket:
            render_progress("data_gen", inst_i + 1, n_instances, t0)
            last_bucket = bucket
    return records


def build_datasets(cfg):
    engine = ProjectiveEngine(cfg.p)
    factory = TaskFactory(engine)

    train_recs = collect_decisions(engine, factory, cfg.task_mix, cfg.train_instances)
    val_recs = collect_decisions(engine, factory, cfg.task_mix, cfg.val_instances)

    tokenizer = SimpleTokenizer([r.state_text for r in (train_recs + val_recs)])

    op_set = {"STOP"}
    sym_set = set()
    for r in (train_recs + val_recs):
        for a in r.legal_actions:
            op, args = parse_action(a)
            op_set.add(op)
            sym_set.update(args)
    codec = ActionCodec(["<PAD_OP>", "<UNK_OP>"] + sorted(op_set), ["<PAD_ARG>", "<UNK_ARG>"] + sorted(sym_set), cfg.max_action_args)

    all_states = [tokenizer.encode(r.state_text, add_bos=True) for r in (train_recs + val_recs)]
    max_T = max(len(x) for x in all_states)
    max_A = max(len(r.legal_actions) for r in (train_recs + val_recs))
    pad_struct = [codec.pad_op] + [codec.pad_sym] * cfg.max_action_args

    def encode_split(records: List[DecisionRecord]):
        xs, legal_s, mask_s, tgt_s, val_s = [], [], [], [], []
        t0 = time.time()
        last_bucket = -1
        total = len(records)
        for i, r in enumerate(records):
            x = tokenizer.encode(r.state_text, add_bos=True)
            x = x + [tokenizer.pad_id] * (max_T - len(x))
            leg = [codec.encode(a) for a in r.legal_actions]
            m = [True] * len(leg)
            if len(leg) < max_A:
                leg += [pad_struct] * (max_A - len(leg))
                m += [False] * (max_A - len(m))
            xs.append(x)
            legal_s.append(leg)
            mask_s.append(m)
            tgt_s.append(r.legal_actions.index(r.teacher_action))
            val_s.append(float(r.steps_to_goal))

            bucket = int(((i + 1) * 20) / max(1, total))
            if bucket != last_bucket:
                render_progress("encode", i + 1, total, t0)
                last_bucket = bucket

        return (
            np.asarray(xs, dtype=np.int32),
            np.asarray(legal_s, dtype=np.int32),
            np.asarray(mask_s, dtype=np.bool_),
            np.asarray(tgt_s, dtype=np.int32),
            np.asarray(val_s, dtype=np.float32),
        )

    train_ds = DecisionDataset(*encode_split(train_recs))
    val_ds = DecisionDataset(*encode_split(val_recs))
    return train_ds, val_ds, {
        "tokenizer": tokenizer,
        "codec": codec,
        "max_T": max_T,
        "max_A": max_A,
        "engine": engine,
        "factory": factory,
    }
