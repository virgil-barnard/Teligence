import json
import os
import subprocess
import time


def run_trial(trial_env):
    env = os.environ.copy()
    env.update(trial_env)

    use_docker = env.get("SWEEP_USE_DOCKER", "1") == "1"
    if use_docker:
        cmd = ["docker", "compose", "run", "--rm"]
        for k in sorted(trial_env.keys()):
            cmd.extend(["-e", f"{k}={trial_env[k]}"])
        cmd.append("gpt")
    else:
        cmd = ["python", "gpt.py"]

    print("\n=== Trial", trial_env["RUN_NAME"], "===")
    print("Command:", " ".join(cmd))
    t0 = time.time()
    rc = subprocess.call(cmd, env=env)
    dt = time.time() - t0
    print(f"Exit code: {rc} | wall_time_s={dt:.1f}")

    summary_path = os.path.join(env.get("RUNS_DIR", "./runs"), trial_env["RUN_NAME"], "summary.json")
    if rc == 0 and os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"run_id": trial_env["RUN_NAME"], "failed": True, "exit_code": rc}


def main():
    stamp = time.strftime("%Y%m%d_%H%M%S")
    prefix = os.environ.get("SWEEP_NAME", f"sweep_{stamp}")

    base = {
        "DATASET": os.environ.get("DATASET", "enwik8"),
        "NUM_UPDATES": os.environ.get("NUM_UPDATES", "400"),
        "LOG_EVERY": os.environ.get("LOG_EVERY", "50"),
        "EVAL_EVERY": os.environ.get("EVAL_EVERY", "50"),
        "EVAL_TOKENS": os.environ.get("EVAL_TOKENS", "262144"),
        "SAVE_EVERY": os.environ.get("SAVE_EVERY", "0"),
        "RUNS_DIR": os.environ.get("RUNS_DIR", "./runs"),
    }

    grid = [
        {"BASE_LR": "3e-4", "WEIGHT_DECAY": "0.10", "DROPOUT": "0.10"},
        {"BASE_LR": "2e-4", "WEIGHT_DECAY": "0.10", "DROPOUT": "0.10"},
        {"BASE_LR": "4e-4", "WEIGHT_DECAY": "0.08", "DROPOUT": "0.10"},
        {"BASE_LR": "3e-4", "WEIGHT_DECAY": "0.05", "DROPOUT": "0.05"},
    ]

    results = []
    for i, cfg in enumerate(grid, start=1):
        trial_env = dict(base)
        trial_env.update(cfg)
        trial_env["RUN_NAME"] = f"{prefix}_t{i:02d}"
        out = run_trial(trial_env)
        out["trial_cfg"] = cfg
        results.append(out)

    ok = [r for r in results if not r.get("failed")]
    ok.sort(key=lambda x: x.get("best_val_bpc", float("inf")))

    report_path = os.path.join(base["RUNS_DIR"], f"{prefix}_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    print("\n=== Sweep Results ===")
    if ok:
        for r in ok:
            print(
                f"{r['run_id']}: best_val_bpc={r.get('best_val_bpc', float('nan')):.4f} "
                f"test_bpc={r.get('test_bpc', float('nan')):.4f} cfg={r.get('trial_cfg')}"
            )
        print("Best run:", ok[0]["run_id"])
    else:
        print("No successful trials.")
    print("Wrote sweep report:", report_path)


if __name__ == "__main__":
    main()
