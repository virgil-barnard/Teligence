#!/usr/bin/env python3
"""Training/eval entrypoint for affine/projective proof GPT experiment."""

from __future__ import annotations

import argparse
import dataclasses
import heapq
import json
import os
import random
import time
from dataclasses import dataclass
from typing import List

import tensorflow as tf

from experiments.proof_affine_domain import ProofEnv, TaskFactory, build_datasets, render_progress
from teligence.action_heads import ActionPointerValueModel
from teligence.experiment_utils import (
    apply_weight_decay,
    assign_optimizer_lr,
    build_backbone_cfg,
    cosine_lr,
    ExperimentLogger,
    maybe_print_train_step,
    set_global_seed,
    should_eval,
)
from teligence.modeling import set_precision


@dataclass
class Config:
    p: int = 5
    task_mix: tuple[str, ...] = (
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
    out_dir: str = "runs/proof_affine_gpt"
    checkpoint_name: str = "proof_affine_gpt"


CFG = Config()


def make_backbone_cfg(cfg: Config, vocab_size: int, seq_len: int):
    return build_backbone_cfg(
        vocab_size=vocab_size,
        seq_len=seq_len,
        n_layer=cfg.n_layer,
        n_embd=cfg.n_embd,
        n_head=cfg.n_head,
        n_kv_head=cfg.n_kv_head,
        dropout=cfg.dropout,
        batch_size=cfg.batch_size,
        num_updates=cfg.max_iters,
        base_lr=cfg.learning_rate,
        min_lr=cfg.min_lr,
        warmup_steps=cfg.warmup_steps,
        weight_decay=cfg.weight_decay,
        use_bf16=cfg.use_bf16,
        use_flash_attn=cfg.use_flash_attn,
        flash_q_block=cfg.flash_q_block,
        flash_k_block=cfg.flash_k_block,
    )


def lr_schedule(step: int, cfg: Config) -> float:
    return cosine_lr(step, cfg.learning_rate, cfg.min_lr, cfg.warmup_steps, cfg.max_iters)


@tf.function(jit_compile=False, reduce_retracing=True)
def train_step(model, opt, x, ls, lm, y, v, grad_clip):
    with tf.GradientTape() as tape:
        _, _, loss, _, _ = model.forward_actions(x, ls, lm, targets=y, value_targets=v, training=True)
    grads = tape.gradient(loss, model.trainable_variables)
    grads_vars = [(g, var) for g, var in zip(grads, model.trainable_variables) if g is not None]
    if grads_vars:
        gvals = [g for g, _ in grads_vars]
        gvals, _ = tf.clip_by_global_norm(gvals, grad_clip)
        opt.apply_gradients(zip(gvals, [var for _, var in grads_vars]))
    return loss


@tf.function(jit_compile=False, reduce_retracing=True)
def eval_step(model, x, ls, lm, y, v):
    logits, _, _, pl, vl = model.forward_actions(x, ls, lm, targets=y, value_targets=v, training=False)
    pred = tf.argmax(logits, axis=-1, output_type=tf.int32)
    acc = tf.reduce_mean(tf.cast(tf.equal(pred, y), tf.float32))
    return pl, vl, acc


def estimate_metrics(model, train_ds, val_ds, cfg: Config):
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
                render_progress(f"eval:{split}", i + 1, cfg.eval_batches, t0)
        out[split] = {
            "policy_loss": sum(policy_losses) / len(policy_losses),
            "value_mse": sum(value_losses) / len(value_losses),
            "acc": sum(accs) / len(accs),
        }
    return out


def _encode_state_for_infer(tok, state_text: str, max_T: int):
    ids = tok.encode(state_text, add_bos=True)[:max_T]
    return ids + [tok.pad_id] * (max_T - len(ids))


def _encode_legal_for_infer(codec, legal_actions: List[str], max_A: int, max_args: int):
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
    logits, value, _, _, _ = model.forward_actions(
        tf.constant([x], dtype=tf.int32),
        tf.constant([structs], dtype=tf.int32),
        tf.constant([mask], dtype=tf.bool),
        training=False,
    )
    n = min(len(legal), max_A)
    return legal[:n], tf.cast(logits[0, :n], tf.float32), float(value.numpy()[0])


@dataclass(order=True)
class _Node:
    priority: float
    neglogp: float
    depth: int
    env: ProofEnv


def rollout_best_first_trace(model, tok, codec, engine, task, max_T, max_A, cfg: Config, verbose):
    root = ProofEnv(engine, task)
    max_depth = len(task.teacher_actions) + cfg.rollout_max_steps_slack
    _, _, v0 = policy_value(model, tok, codec, root, max_T, max_A)
    heap: List[_Node] = []
    heapq.heappush(heap, _Node(priority=cfg.search_value_weight * v0, neglogp=0.0, depth=0, env=root))
    best_env = root
    best_depth = 0
    expansions = 0
    while heap and expansions < cfg.search_max_expansions:
        node = heapq.heappop(heap)
        env = node.env
        if len(env.history) > best_depth:
            best_depth = len(env.history)
            best_env = env
        if env.done and env.success():
            return True, env
        if node.depth >= max_depth:
            continue
        legal, logits, v = policy_value(model, tok, codec, env, max_T, max_A)
        logp = tf.nn.log_softmax(logits)
        k = min(cfg.search_topk_actions, int(logp.shape[0]))
        top = tf.math.top_k(logp, k=k)
        expansions += 1
        if verbose and expansions % 25 == 0:
            print(f"exp {expansions:4d} depth {node.depth:2d} neglogp {node.neglogp:.2f} V~{v:.2f} heap={len(heap)}")
        child_h = max(0.0, v - 1.0)
        for lp, idx in zip(top.values.numpy().tolist(), top.indices.numpy().tolist()):
            child = env.clone()
            if not child.execute(legal[int(idx)]):
                continue
            neglogp = node.neglogp + float(-lp)
            priority = neglogp + cfg.search_value_weight * child_h
            heapq.heappush(heap, _Node(priority=priority, neglogp=neglogp, depth=node.depth + 1, env=child))
    return False, best_env


def train(cfg: Config, resume_checkpoint: str = ""):
    set_global_seed(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    run_id = os.path.basename(os.path.normpath(cfg.out_dir)) or cfg.checkpoint_name
    logger = ExperimentLogger(run_dir=cfg.out_dir, dataset_name="proof_affine", run_id=run_id)
    train_ds, val_ds, meta = build_datasets(cfg)
    tok = meta["tokenizer"]
    codec = meta["codec"]
    max_T = meta["max_T"]
    max_A = meta["max_A"]
    engine = meta["engine"]
    factory: TaskFactory = meta["factory"]

    print(f"vocab_size={tok.vocab_size} max_T={max_T} max_A={max_A} ops={codec.n_ops} syms={codec.n_syms}")
    print(f"train_decisions={train_ds.states.shape[0]} val_decisions={val_ds.states.shape[0]}")
    logger.log_event(
        {
            "event": "run_start",
            "max_iters": cfg.max_iters,
            "eval_interval": cfg.eval_interval,
            "eval_batches": cfg.eval_batches,
            "batch_size": cfg.batch_size,
            "out_dir": cfg.out_dir,
        }
    )

    backbone_cfg = make_backbone_cfg(cfg, tok.vocab_size, max_T)
    set_precision(backbone_cfg.use_bf16)
    model = ActionPointerValueModel(backbone_cfg, tok.pad_id, codec, cfg.value_loss_weight)
    _ = model.forward_actions(
        tf.zeros([1, max_T], dtype=tf.int32),
        tf.zeros([1, max_A, 1 + cfg.max_action_args], dtype=tf.int32),
        tf.zeros([1, max_A], dtype=tf.bool),
        training=False,
    )

    opt = tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate, beta_1=0.9, beta_2=0.95, epsilon=1e-8)
    step_var = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False)
    ckpt = tf.train.Checkpoint(model=model, step=step_var)
    last_mgr = tf.train.CheckpointManager(ckpt, os.path.join(cfg.out_dir, "ckpt_last"), max_to_keep=3)
    best_mgr = tf.train.CheckpointManager(ckpt, os.path.join(cfg.out_dir, "ckpt_best"), max_to_keep=1)

    restore_path = resume_checkpoint.strip() if resume_checkpoint else (last_mgr.latest_checkpoint or "")
    start_it = 0
    if restore_path:
        ckpt.restore(restore_path).expect_partial()
        start_it = int(step_var.numpy()) + 1
        print(f"restored checkpoint: {restore_path} (step={int(step_var.numpy())})")
    else:
        print("no checkpoint found, starting fresh")

    best_val = float("inf")
    t0 = time.time()
    timer_state = {"t0": time.time()}
    grad_clip_t = tf.constant(cfg.grad_clip, dtype=tf.float32)
    for it in range(start_it, cfg.max_iters + 1):
        step_var.assign(it)
        if should_eval(it, cfg.max_iters, cfg.eval_interval):
            eval_start = time.time()
            stats = estimate_metrics(model, train_ds, val_ds, cfg)
            tr = stats["train"]
            va = stats["val"]
            eval_wall = time.time() - eval_start
            print(
                f"step {it:5d} | train pl {tr['policy_loss']:.4f} acc {tr['acc']:.3f} vMSE {tr['value_mse']:.3f} | "
                f"val pl {va['policy_loss']:.4f} acc {va['acc']:.3f} vMSE {va['value_mse']:.3f} | {time.time()-t0:.1f}s"
            )
            if va["policy_loss"] < best_val:
                best_val = va["policy_loss"]
                print(f"saved best checkpoint: {best_mgr.save(checkpoint_number=it)}")
            last_mgr.save(checkpoint_number=it)
            logger.log_event(
                {
                    "event": "eval",
                    "update": it,
                    "num_updates": cfg.max_iters,
                    "primary_metric_name": "val_policy_loss",
                    "primary_metric": float(va["policy_loss"]),
                    "secondary_metric_name": "val_acc",
                    "secondary_metric": float(va["acc"]),
                    "eval_wall_s": float(eval_wall),
                    "train_policy_loss": float(tr["policy_loss"]),
                    "train_acc": float(tr["acc"]),
                    "val_policy_loss": float(va["policy_loss"]),
                    "val_acc": float(va["acc"]),
                    "val_value_mse": float(va["value_mse"]),
                }
            )

        x, ls, lm, y, v = train_ds.sample_batch(cfg.batch_size)
        lr = lr_schedule(it, cfg)
        assign_optimizer_lr(opt, lr)
        loss = train_step(model, opt, x, ls, lm, y, v, grad_clip_t)
        apply_weight_decay(model.trainable_variables, lr, cfg.weight_decay)
        maybe_print_train_step(
            step_value=it,
            end_step_value=cfg.max_iters,
            print_every=100,
            loss_value=float(loss.numpy()),
            lr_value=lr,
            timer_state=timer_state,
            prefix="update",
        )
        if it % 100 == 0:
            logger.log_event(
                {
                    "event": "train",
                    "update": it,
                    "num_updates": cfg.max_iters,
                    "loss": float(loss.numpy()),
                    "lr": float(lr),
                }
            )
        if (it + 1) % max(1, cfg.max_iters // 20) == 0 or it == cfg.max_iters:
            render_progress("train", it + 1, cfg.max_iters + 1, t0)

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
        steps = len(env.history) if env is not None else 0
        print(f"[quick rollout] {i+1}/{quick_rollouts} task={task.name} solved={ok} steps={steps}")
        render_progress("quick_rollout", i + 1, quick_rollouts, quick_t0)
        rec = task_stats[task.name]
        rec["attempts"] += 1
        rec["steps_all"] += steps
        if ok:
            successes += 1
            rec["solved"] += 1
            rec["steps_solved"] += steps
            if steps > hardest_steps:
                hardest_steps = steps
                hardest_task = task
                hardest_env = env

    print(f"search success@10 = {successes}/10")
    print("\n--- quick rollout synopsis by task ---")
    for task_name in sorted(task_stats):
        rec = task_stats[task_name]
        if rec["attempts"] == 0:
            continue
        solved_steps = "-" if rec["solved"] == 0 else f"{(rec['steps_solved']/rec['solved']):.2f}"
        print(
            f"{task_name:26s} attempts={rec['attempts']:2d} solved={rec['solved']:2d} "
            f"solve_rate={rec['solved']/rec['attempts']:.2f} avg_steps_all={rec['steps_all']/rec['attempts']:.2f} avg_steps_solved={solved_steps}"
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

    logger.write_summary(
        {
            "num_updates": cfg.max_iters,
            "best_primary_metric_name": "val_policy_loss",
            "best_primary_metric": float(best_val if best_val < 1e30 else 0.0),
            "quick_rollout_success": float(successes / max(1, quick_rollouts)),
            "hardest_solved_task": (hardest_task.name if hardest_task is not None else "none"),
            "hardest_solved_steps": int(max(0, hardest_steps)),
        }
    )
    print(f"wrote run summary: {logger.summary_json}")

    return best_mgr.latest_checkpoint or last_mgr.latest_checkpoint, meta_path


def load_checkpoint(out_dir: str, checkpoint: str):
    meta_path = os.path.join(out_dir, f"{CFG.checkpoint_name}.meta.json")
    with open(meta_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    cfg = Config(**payload["cfg"])
    from experiments.proof_affine_domain import ActionCodec, ProjectiveEngine, SimpleTokenizer

    tok = SimpleTokenizer.from_stoi(payload["tokenizer_stoi"])
    codec = ActionCodec(payload["codec_ops"], payload["codec_syms"], cfg.max_action_args)
    max_T = int(payload["max_T"])
    max_A = int(payload["max_A"])
    backbone_cfg = make_backbone_cfg(cfg, tok.vocab_size, max_T)
    set_precision(backbone_cfg.use_bf16)
    model = ActionPointerValueModel(backbone_cfg, tok.pad_id, codec, cfg.value_loss_weight)
    _ = model.forward_actions(
        tf.zeros([1, max_T], dtype=tf.int32),
        tf.zeros([1, max_A, 1 + cfg.max_action_args], dtype=tf.int32),
        tf.zeros([1, max_A], dtype=tf.bool),
        training=False,
    )
    step_var = tf.Variable(tf.constant(0, dtype=tf.int64), trainable=False)
    tf.train.Checkpoint(model=model, step=step_var).restore(checkpoint).expect_partial()
    engine = ProjectiveEngine(cfg.p)
    factory = TaskFactory(engine)
    return cfg, tok, codec, model, engine, factory, max_T, max_A


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["train", "rollout"], default="train")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--checkpoint", type=str, default="")
    p.add_argument("--train_instances", type=int, default=CFG.train_instances)
    p.add_argument("--val_instances", type=int, default=CFG.val_instances)
    p.add_argument("--max_iters", "--train_steps", dest="max_iters", type=int, default=CFG.max_iters)
    p.add_argument("--eval_interval", "--eval_every", dest="eval_interval", type=int, default=CFG.eval_interval)
    p.add_argument("--batch_size", type=int, default=CFG.batch_size)
    p.add_argument("--eval_batches", type=int, default=CFG.eval_batches)
    p.add_argument("--search_max_expansions", type=int, default=CFG.search_max_expansions)
    p.add_argument("--search_topk_actions", type=int, default=CFG.search_topk_actions)
    p.add_argument("--search_value_weight", type=float, default=CFG.search_value_weight)
    p.add_argument("--out_dir", type=str, default=CFG.out_dir)
    p.add_argument("--resume_checkpoint", type=str, default="")
    args = p.parse_args()

    if args.quick:
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
        if args.search_max_expansions == CFG.search_max_expansions:
            args.search_max_expansions = 128
        if args.search_topk_actions == CFG.search_topk_actions:
            args.search_topk_actions = 3
        print("[quick] enabled fast sanity defaults")

    cfg = Config(
        train_instances=args.train_instances,
        val_instances=args.val_instances,
        max_iters=args.max_iters,
        eval_interval=args.eval_interval,
        batch_size=args.batch_size,
        eval_batches=args.eval_batches,
        search_max_expansions=args.search_max_expansions,
        search_topk_actions=args.search_topk_actions,
        search_value_weight=args.search_value_weight,
        out_dir=args.out_dir,
    )

    if args.mode == "train":
        ckpt, meta = train(cfg, resume_checkpoint=args.resume_checkpoint)
        print(f"checkpoint: {ckpt}")
        print(f"meta: {meta}")
        return

    checkpoint = args.checkpoint
    if not checkpoint:
        mgr = tf.train.CheckpointManager(tf.train.Checkpoint(), os.path.join(cfg.out_dir, "ckpt_best"), max_to_keep=1)
        checkpoint = mgr.latest_checkpoint or tf.train.CheckpointManager(
            tf.train.Checkpoint(), os.path.join(cfg.out_dir, "ckpt_last"), max_to_keep=1
        ).latest_checkpoint
    if checkpoint is None:
        raise ValueError("No checkpoint found. Provide --checkpoint or train first.")

    cfg2, tok, codec, model, engine, factory, max_T, max_A = load_checkpoint(cfg.out_dir, checkpoint)
    successes = 0
    task_stats = {name: {"attempts": 0, "solved": 0} for name in cfg2.task_mix}
    for _ in range(5):
        task = factory.sample(random.choice(cfg2.task_mix))
        ok, _ = rollout_best_first_trace(model, tok, codec, engine, task, max_T, max_A, cfg2, verbose=True)
        successes += int(ok)
        task_stats[task.name]["attempts"] += 1
        task_stats[task.name]["solved"] += int(ok)
    print(f"search success rate: {successes}/5")
    print("rollout synopsis by task:")
    for task_name in sorted(task_stats):
        rec = task_stats[task_name]
        if rec["attempts"] > 0:
            print(f"  {task_name:26s} solved={rec['solved']:2d}/{rec['attempts']:2d} ({rec['solved']/rec['attempts']:.2f})")


if __name__ == "__main__":
    main()
