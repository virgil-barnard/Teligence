import csv
import json
import os
import subprocess
import time


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, str(default)))


def _build_profiles():
    profiles = {
        "enwik8": {
            "domain": "web_text_byte",
            "env": {"DATASET": "enwik8", "TOKENIZER": "byte", "PREVIEW_PROMPT": "The "},
        },
        "tinyshakespeare": {
            "domain": "natural_language_char",
            "env": {"DATASET": "tinyshakespeare", "TOKENIZER": "char", "PREVIEW_PROMPT": "ROMEO: "},
        },
        "names": {
            "domain": "entity_names_char",
            "env": {"DATASET": "names", "TOKENIZER": "char", "PREVIEW_PROMPT": "mar"},
        },
    }
    return profiles


def _run_trial(run_name: str, trial_env: dict, use_docker: bool):
    env = os.environ.copy()
    env.update(trial_env)

    if use_docker:
        cmd = ["docker", "compose", "run", "--rm"]
        for k in sorted(trial_env.keys()):
            cmd.extend(["-e", f"{k}={trial_env[k]}"])
        cmd.append("gpt")
    else:
        cmd = ["python", "scripts/gpt_text.py"]

    print(f"\n=== Benchmark {run_name} ===")
    print("Command:", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    dt = time.time() - t0
    print(f"Exit code: {rc} | wall_time_s={dt:.1f}")

    summary_path = os.path.join(trial_env.get("RUNS_DIR", "./runs"), run_name, "summary.json")
    if rc == 0 and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            s = json.load(f)
        s["wall_time_s"] = dt
        return s

    return {"run_id": run_name, "failed": True, "exit_code": rc, "wall_time_s": dt}


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    matrix_name = os.environ.get("BENCH_NAME", f"matrix_{stamp}")
    runs_dir = os.environ.get("RUNS_DIR", "./runs")
    use_docker = os.environ.get("BENCH_USE_DOCKER", "1") == "1"

    profiles = _build_profiles()
    selected = os.environ.get("BENCH_PROFILES", "enwik8,tinyshakespeare,names").split(",")
    selected = [p.strip() for p in selected if p.strip()]

    base = {
        "RUNS_DIR": runs_dir,
        "NUM_UPDATES": str(_env_int("BENCH_NUM_UPDATES", 300)),
        "LOG_EVERY": str(_env_int("BENCH_LOG_EVERY", 50)),
        "EVAL_EVERY": str(_env_int("BENCH_EVAL_EVERY", 50)),
        "EVAL_TOKENS": str(_env_int("BENCH_EVAL_TOKENS", 262144)),
        "SAVE_EVERY": str(_env_int("BENCH_SAVE_EVERY", 0)),
        "VISUALIZE": os.environ.get("BENCH_VISUALIZE", "0"),
    }

    results = []
    for p in selected:
        if p not in profiles:
            print(f"Skipping unknown profile: {p}")
            continue
        run_name = f"{matrix_name}_{p}"
        trial_env = dict(base)
        trial_env.update(profiles[p]["env"])
        trial_env["RUN_NAME"] = run_name

        out = _run_trial(run_name, trial_env, use_docker=use_docker)
        out["profile"] = p
        out["domain"] = profiles[p]["domain"]
        results.append(out)

    ok = [r for r in results if not r.get("failed")]
    ok.sort(key=lambda x: x.get("best_val_bpc", float("inf")))

    os.makedirs(runs_dir, exist_ok=True)
    report_json = os.path.join(runs_dir, f"{matrix_name}_report.json")
    report_csv = os.path.join(runs_dir, f"{matrix_name}_report.csv")

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    with open(report_csv, "w", encoding="utf-8", newline="") as f:
        wr = csv.DictWriter(
            f,
            fieldnames=[
                "profile",
                "domain",
                "run_id",
                "best_val_bpc",
                "best_val_step",
                "test_bpc",
                "test_nll",
                "test_ppl",
                "num_updates",
                "wall_time_s",
                "failed",
                "exit_code",
            ],
        )
        wr.writeheader()
        for r in results:
            wr.writerow(
                {
                    "profile": r.get("profile"),
                    "domain": r.get("domain"),
                    "run_id": r.get("run_id"),
                    "best_val_bpc": r.get("best_val_bpc"),
                    "best_val_step": r.get("best_val_step"),
                    "test_bpc": r.get("test_bpc"),
                    "test_nll": r.get("test_nll"),
                    "test_ppl": r.get("test_ppl"),
                    "num_updates": r.get("num_updates"),
                    "wall_time_s": r.get("wall_time_s"),
                    "failed": r.get("failed", False),
                    "exit_code": r.get("exit_code", 0),
                }
            )

    print("\n=== Benchmark Matrix Results ===")
    if ok:
        for r in ok:
            print(
                f"{r['profile']:16s} | best_val_bpc={r.get('best_val_bpc', float('nan')):.4f} "
                f"| test_bpc={r.get('test_bpc', float('nan')):.4f} "
                f"| wall_time_s={r.get('wall_time_s', float('nan')):.1f}"
            )
        print(f"Best profile by val_bpc: {ok[0]['profile']} ({ok[0]['run_id']})")
    else:
        print("No successful profiles.")
    print(f"Wrote report JSON: {report_json}")
    print(f"Wrote report CSV:  {report_csv}")


if __name__ == "__main__":
    main()
