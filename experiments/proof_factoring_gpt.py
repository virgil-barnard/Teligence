"""
experiments/proof_factoring_gpt.py

Proof-rewrite training script using shared `ExplicitGPT` backbone.
Domain/data generation lives in `experiments/proof_factoring_domain.py`.
"""

from __future__ import annotations

import argparse
import dataclasses
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import tensorflow as tf

from experiments.proof_factoring_domain import (
    ProofEnv,
    ProofGymConfig,
    Vocab,
    action_to_token,
    build_vocab,
    encode_transcript,
    generate_trajectory,
    list_valid_actions,
    make_rules,
    token_to_action,
)
from teligence.config import GPTConfig, validate_config
from teligence.experiment_utils import (
    assign_optimizer_lr,
    cosine_lr,
    ExperimentLogger,
    maybe_print_train_step,
    set_global_seed,
    should_eval,
)
from teligence.modeling import ExplicitGPT, set_precision


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


def make_dataset(vocab: Vocab, rules, gym_cfg: ProofGymConfig, train_cfg: TrainConfig) -> tf.data.Dataset:
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
        tf.TensorSpec(shape=(2, train_cfg.seq_len), dtype=tf.int32),
        tf.TensorSpec(shape=(train_cfg.seq_len,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)
    return ds.batch(train_cfg.batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)


def lr_schedule(step: int, cfg: TrainConfig) -> float:
    return cosine_lr(step, cfg.lr, cfg.min_lr, cfg.warmup_steps, cfg.train_steps)


@tf.function(jit_compile=False)
def train_step(model: tf.keras.Model, opt: tf.keras.optimizers.Optimizer, x: tf.Tensor, loss_mask: tf.Tensor) -> tf.Tensor:
    with tf.GradientTape() as tape:
        logits = model(x, training=True)
        y = x[:, 1:]
        logits = logits[:, :-1, :]
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=tf.cast(logits, tf.float32))
        lm = loss_mask[:, :-1]
        loss = tf.reduce_sum(ce * lm) / tf.maximum(1.0, tf.reduce_sum(lm))

    grads = tape.gradient(loss, model.trainable_variables)
    grads_and_vars = [(g, v) for g, v in zip(grads, model.trainable_variables) if g is not None]
    opt.apply_gradients(grads_and_vars)
    return loss


def policy_rollout(
    model: tf.keras.Model,
    vocab: Vocab,
    rules,
    gym_cfg: ProofGymConfig,
    seq_len: int,
    greedy: bool = True,
    trace: bool = False,
    constrain_to_valid: bool = True,
    avoid_revisits: bool = True,
):
    def pack(ids: List[int]) -> np.ndarray:
        x = np.full((seq_len,), vocab.token_to_id["<pad>"], dtype=np.int32)
        tail = ids[-seq_len:]
        x[-len(tail) :] = np.array(tail, dtype=np.int32)
        return x

    tr = generate_trajectory(gym_cfg, rules)
    env = ProofEnv(gym_cfg, rules)
    st = env.reset(tr.start, tr.goal)
    visited_states = {st.lhs}

    trace_lines: List[str] = []
    if trace:
        trace_lines.append(f"goal={st.rhs.pretty()}")
        trace_lines.append(f"start={st.lhs.pretty()}")

    toks = ["<bos>", "GOAL", *tr.goal.to_tokens(), "STATE", *st.lhs.to_tokens(), "ACT"]
    ids = vocab.encode(toks)

    for step in range(gym_cfg.max_steps):
        x = pack(ids)[None, :]
        logits = model(tf.constant(x), training=False)
        last = seq_len - 1
        next_logits = logits[0, last, :].numpy()

        if constrain_to_valid:
            valid_actions = list_valid_actions(st.lhs, rules, gym_cfg)
            if avoid_revisits:
                non_loop = [item for item in valid_actions if item[2] not in visited_states]
                if non_loop:
                    valid_actions = non_loop
                else:
                    if trace:
                        trace_lines.append(f"step={step+1} no_unseen_valid_actions=true")
                    return False, step + 1, trace_lines

            valid_token_ids = [vocab.token_to_id[action_to_token(rules, rid, node_idx)] for rid, node_idx, _ in valid_actions]
            if valid_token_ids:
                masked = np.full_like(next_logits, -1e9, dtype=np.float32)
                vti = np.array(valid_token_ids, dtype=np.int32)
                masked[vti] = next_logits[vti]
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
            if trace:
                trace_lines.append(f"step={step+1} pred_invalid_action=true")
            return False, step + 1, trace_lines

        st, _r, done, info = env.step_action(act)
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

        ids.extend(vocab.encode(["STATE", *st.lhs.to_tokens(), "ACT"]))

    return env.is_solved(), gym_cfg.max_steps, trace_lines


def evaluate(model: tf.keras.Model, vocab: Vocab, rules, gym_cfg: ProofGymConfig, train_cfg: TrainConfig) -> Dict[str, float]:
    solved = 0
    steps_sum = 0
    shown = 0
    shown_success = 0
    hardest_solved_steps = -1
    hardest_solved_trace = None
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
            avoid_revisits=True,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="train", choices=["train", "eval", "sample"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="proof_factoring_gpt")
    parser.add_argument("--runs_dir", type=str, default="runs")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--no_resume", action="store_true")
    parser.add_argument("--train_steps", "--max_iters", dest="train_steps", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=256)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--n_kv_head", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=8)
    parser.add_argument("--eval_every", "--eval_interval", dest="eval_every", type=int, default=200)
    parser.add_argument("--eval_episodes", type=int, default=200)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
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

    curriculum_mix_levels = not args.no_curriculum_mix_levels
    eval_show_hardest_solved = not args.no_eval_show_hardest_solved

    set_global_seed(args.seed)
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
    vocab, _ = build_vocab(rules, vocab_cfg)

    run_dir = os.path.join(args.runs_dir, args.run_name)
    ckpt_dir = args.ckpt.strip() if args.ckpt.strip() else os.path.join(run_dir, "ckpt_last")
    logger = ExperimentLogger(run_dir=run_dir, dataset_name="proof_factoring", run_id=args.run_name)

    attn_window = min(256, args.seq_len)
    flash_block = 128
    while flash_block > 1 and (args.seq_len % flash_block != 0):
        flash_block //= 2

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
        runs_dir=args.runs_dir,
    )
    validate_config(cfg)
    set_precision(cfg.use_bf16)

    model = ExplicitGPT(cfg)
    _ = model(tf.zeros([1, cfg.seq_len], dtype=tf.int32))

    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=3)
    if manager.latest_checkpoint and not args.no_resume:
        restore_path = manager.latest_checkpoint
        try:
            ckpt.restore(restore_path).expect_partial()
            print(f"[loaded] {restore_path}")
        except Exception as e:
            print(
                f"[warn] failed to restore checkpoint {restore_path}: {e}\n"
                "[warn] starting fresh model state. Use a new --run_name or --ckpt directory to avoid mixing vocab/config variants."
            )
            model = ExplicitGPT(cfg)
            _ = model(tf.zeros([1, cfg.seq_len], dtype=tf.int32))
            ckpt = tf.train.Checkpoint(model=model)
            manager = tf.train.CheckpointManager(ckpt, directory=ckpt_dir, max_to_keep=3)

    train_cfg = TrainConfig(
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        lr=args.lr,
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
    logger.log_event(
        {
            "event": "run_start",
            "seq_len": train_cfg.seq_len,
            "batch_size": train_cfg.batch_size,
            "train_steps": train_cfg.train_steps,
            "eval_every": train_cfg.eval_every,
            "eval_episodes": train_cfg.eval_episodes,
            "ckpt_dir": ckpt_dir,
        }
    )

    if args.mode == "eval":
        print(evaluate(model, vocab, rules, gym_cfg, train_cfg))
        return

    if args.mode == "sample":
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

    opt = tf.keras.optimizers.Adam(learning_rate=train_cfg.lr, beta_1=0.9, beta_2=0.95, epsilon=1e-8)
    ds = make_dataset(vocab, rules, gym_cfg, train_cfg)
    it = iter(ds)
    timer_state = {"t0": time.time()}

    best_solve_rate = float("-inf")
    for step in range(train_cfg.train_steps):
        step_value = step + 1
        batch_ids, batch_mask = next(it)
        x = batch_ids[:, 0, :]
        lm = batch_mask

        lr_step = lr_schedule(step, train_cfg)
        assign_optimizer_lr(opt, lr_step)

        loss = train_step(model, opt, x, lm)

        lr_now = float(tf.convert_to_tensor(opt.learning_rate).numpy())
        maybe_print_train_step(
            step_value=step_value,
            end_step_value=train_cfg.train_steps,
            print_every=train_cfg.print_every,
            loss_value=float(loss),
            lr_value=lr_now,
            timer_state=timer_state,
            prefix="update",
        )
        if train_cfg.print_every > 0 and step_value % train_cfg.print_every == 0:
            logger.log_event(
                {
                    "event": "train",
                    "update": step_value,
                    "num_updates": train_cfg.train_steps,
                    "loss": float(loss),
                    "lr": lr_now,
                }
            )

        if should_eval(step_value, train_cfg.train_steps, train_cfg.eval_every):
            eval_start = time.time()
            print(f"[eval] starting at step {step_value} ...")
            stats = evaluate(model, vocab, rules, gym_cfg, train_cfg)
            eval_wall = time.time() - eval_start
            best_solve_rate = max(best_solve_rate, float(stats["solve_rate"]))
            print(
                f"[eval] step {step_value}: solve_rate={stats['solve_rate']:.3f} "
                f"avg_steps={stats['avg_steps']:.2f} "
                f"eval_s={stats['eval_seconds']:.1f} wall_s={eval_wall:.1f}"
            )
            logger.log_event(
                {
                    "event": "eval",
                    "update": step_value,
                    "num_updates": train_cfg.train_steps,
                    "primary_metric_name": "solve_rate",
                    "primary_metric": float(stats["solve_rate"]),
                    "secondary_metric_name": "avg_steps",
                    "secondary_metric": float(stats["avg_steps"]),
                    "eval_wall_s": float(eval_wall),
                    "solve_rate": float(stats["solve_rate"]),
                    "avg_steps": float(stats["avg_steps"]),
                    "eval_s": float(stats["eval_seconds"]),
                }
            )
            manager.save()

    manager.save()
    logger.write_summary(
        {
            "num_updates": train_cfg.train_steps,
            "best_primary_metric_name": "solve_rate",
            "best_primary_metric": float(best_solve_rate if best_solve_rate > -1e9 else 0.0),
            "last_checkpoint_dir": ckpt_dir,
        }
    )
    print(f"wrote run summary: {logger.summary_json}")
    print("[done]")


if __name__ == "__main__":
    main()
