from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.score import score_summary


VALID_TRACKS = ("gpt_text", "proof_factoring", "proof_affine")
RESULTS_HEADER = [
    "timestamp",
    "track",
    "branch",
    "commit",
    "status",
    "primary_metric_name",
    "primary_metric_value",
    "score",
    "run_dir",
    "summary_path",
    "description",
]


def _repo_root() -> Path:
    return REPO_ROOT


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _run(cmd: str, timeout_sec: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout_sec)


def _default_command_and_summary(track: str, run_name: str, dataset: str, extra_args: str) -> tuple[str, Path]:
    if track == "gpt_text":
        cmd = (
            "docker compose run --rm "
            f"-e APP_ENTRY=gpt_text -e DATASET={dataset} -e RUN_NAME={run_name} gpt"
        )
        return cmd, Path(f"runs/{run_name}/summary.json")

    if track == "proof_factoring":
        app_args = f"--run_name {run_name}".strip()
        if extra_args.strip():
            app_args = f"{app_args} {extra_args.strip()}"
        cmd = (
            "docker compose run --rm "
            f"-e APP_ENTRY=proof_factoring -e APP_ARGS=\"{app_args}\" gpt"
        )
        return cmd, Path(f"runs/{run_name}/summary.json")

    if track == "proof_affine":
        app_args = f"--out_dir runs/{run_name}".strip()
        if extra_args.strip():
            app_args = f"{app_args} {extra_args.strip()}"
        cmd = (
            "docker compose run --rm "
            f"-e APP_ENTRY=proof_affine -e APP_ARGS=\"{app_args}\" gpt"
        )
        return cmd, Path(f"runs/{run_name}/summary.json")

    raise ValueError(f"Unknown track: {track}")


def _git_branch() -> str:
    out = _run("git rev-parse --abbrev-ref HEAD")
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _git_commit() -> str:
    out = _run("git rev-parse --short HEAD")
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip() or "unknown"


def _ensure_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writeheader()


def _read_summary(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _best_score_for_track(results_path: Path, track: str) -> float | None:
    if not results_path.exists():
        return None
    best = None
    with results_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("track") != track:
                continue
            if row.get("status") not in {"keep", "discard"}:
                continue
            try:
                score = float(row.get("score", ""))
            except ValueError:
                continue
            if best is None or score > best:
                best = score
    return best


def _append_result(results_path: Path, row: dict) -> None:
    with results_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULTS_HEADER, delimiter="\t")
        writer.writerow(row)


def _load_results_rows(results_path: Path) -> list[dict]:
    if not results_path.exists():
        return []
    with results_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _short(text: str, n: int) -> str:
    if len(text) <= n:
        return text
    return text[: max(0, n - 3)] + "..."


def _as_float(s: str, default: float | None = None) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return default


def _print_row(columns: list[tuple[str, int]]) -> None:
    print("  ".join(f"{_short(str(val), width):<{width}}" for val, width in columns))


def _write_log(log_path: Path, command: str, run_result: subprocess.CompletedProcess | None, timeout: bool, timeout_sec: int | None) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"timestamp: {_now()}\n")
        f.write(f"command: {command}\n")
        f.write(f"timeout_sec: {timeout_sec}\n")
        f.write(f"timed_out: {timeout}\n")
        if run_result is None:
            return
        f.write(f"returncode: {run_result.returncode}\n")
        f.write("\n--- stdout ---\n")
        f.write(run_result.stdout)
        f.write("\n\n--- stderr ---\n")
        f.write(run_result.stderr)


def command_init(args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    _ensure_results_file(results_path)
    print(f"initialized results file: {results_path}")
    return 0


def command_run(args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    _ensure_results_file(results_path)

    if args.track not in VALID_TRACKS:
        raise ValueError(f"track must be one of {VALID_TRACKS}")

    command = args.command.strip()
    summary_path = Path(args.summary_path) if args.summary_path else None

    if not command:
        if not args.run_name.strip():
            raise ValueError("Provide either --command or --run-name")
        command, summary_path = _default_command_and_summary(args.track, args.run_name.strip(), args.dataset.strip(), args.extra_args)

    if summary_path is None:
        raise ValueError("Provide --summary-path when using --command")

    run_result = None
    timed_out = False
    try:
        run_result = _run(command, timeout_sec=args.timeout_sec)
    except subprocess.TimeoutExpired:
        timed_out = True

    log_path = Path(args.log_path) if args.log_path else _repo_root() / "research" / "logs" / f"{args.track}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _write_log(log_path, command, run_result, timed_out, args.timeout_sec)

    branch = _git_branch()
    commit = _git_commit()
    timestamp = _now()
    run_dir = str(summary_path.parent).replace("\\", "/")
    summary_path_str = str(summary_path).replace("\\", "/")

    if timed_out:
        row = {
            "timestamp": timestamp,
            "track": args.track,
            "branch": branch,
            "commit": commit,
            "status": "timeout",
            "primary_metric_name": "",
            "primary_metric_value": "",
            "score": "",
            "run_dir": run_dir,
            "summary_path": summary_path_str,
            "description": args.description,
        }
        _append_result(results_path, row)
        print(f"run timed out, recorded timeout row. log: {log_path}")
        return 124

    if run_result is None or run_result.returncode != 0:
        row = {
            "timestamp": timestamp,
            "track": args.track,
            "branch": branch,
            "commit": commit,
            "status": "crash",
            "primary_metric_name": "",
            "primary_metric_value": "",
            "score": "",
            "run_dir": run_dir,
            "summary_path": summary_path_str,
            "description": args.description,
        }
        _append_result(results_path, row)
        print(f"run crashed, recorded crash row. log: {log_path}")
        return 1

    if not summary_path.exists():
        row = {
            "timestamp": timestamp,
            "track": args.track,
            "branch": branch,
            "commit": commit,
            "status": "crash",
            "primary_metric_name": "missing_summary",
            "primary_metric_value": "",
            "score": "",
            "run_dir": run_dir,
            "summary_path": summary_path_str,
            "description": args.description,
        }
        _append_result(results_path, row)
        print(f"summary missing, recorded crash row. log: {log_path}")
        return 2

    summary = _read_summary(summary_path)
    score_rec = score_summary(args.track, summary)
    metric_name = score_rec.primary_metric_name
    metric_value = score_rec.primary_metric_value
    score = score_rec.score
    best_before = _best_score_for_track(results_path, args.track)
    keep = best_before is None or score > (best_before + args.min_delta)
    status = "keep" if keep else "discard"

    row = {
        "timestamp": timestamp,
        "track": args.track,
        "branch": branch,
        "commit": commit,
        "status": status,
        "primary_metric_name": metric_name,
        "primary_metric_value": f"{metric_value:.12g}",
        "score": f"{score:.12g}",
        "run_dir": run_dir,
        "summary_path": summary_path_str,
        "description": args.description,
    }
    _append_result(results_path, row)

    print(f"run completed: status={status} metric={metric_name} value={metric_value:.8f} score={score:.8f}")
    print(f"results: {results_path}")
    print(f"log: {log_path}")
    return 0


def command_leaderboard(args: argparse.Namespace) -> int:
    results_path = Path(args.results)
    rows = _load_results_rows(results_path)
    if not rows:
        print(f"no results found at: {results_path}")
        return 0

    scored_rows: list[dict] = []
    for row in rows:
        scored = dict(row)
        scored["score_num"] = _as_float(row.get("score", ""))
        scored["metric_num"] = _as_float(row.get("primary_metric_value", ""))
        scored_rows.append(scored)

    print(f"results file: {results_path}")
    print(f"total rows: {len(scored_rows)}")
    print()

    print("best per track")
    _print_row(
        [
            ("track", 16),
            ("status", 8),
            ("metric", 18),
            ("value", 12),
            ("score", 12),
            ("commit", 8),
            ("timestamp", 20),
            ("description", 40),
        ]
    )
    _print_row([("-" * 16, 16), ("-" * 8, 8), ("-" * 18, 18), ("-" * 12, 12), ("-" * 12, 12), ("-" * 8, 8), ("-" * 20, 20), ("-" * 40, 40)])

    for track in VALID_TRACKS:
        candidates = [
            r
            for r in scored_rows
            if r.get("track") == track and r.get("status") in {"keep", "discard"} and r.get("score_num") is not None
        ]
        if not candidates:
            _print_row([(track, 16), ("-", 8), ("-", 18), ("-", 12), ("-", 12), ("-", 8), ("-", 20), ("-", 40)])
            continue
        best = max(candidates, key=lambda r: float(r["score_num"]))
        _print_row(
            [
                (track, 16),
                (best.get("status", ""), 8),
                (best.get("primary_metric_name", ""), 18),
                (best.get("primary_metric_value", ""), 12),
                (best.get("score", ""), 12),
                (best.get("commit", ""), 8),
                (best.get("timestamp", ""), 20),
                (best.get("description", ""), 40),
            ]
        )

    print()
    print(f"recent runs (last {args.limit})")
    _print_row(
        [
            ("timestamp", 20),
            ("track", 16),
            ("status", 8),
            ("metric", 18),
            ("value", 12),
            ("score", 12),
            ("commit", 8),
            ("description", 40),
        ]
    )
    _print_row([("-" * 20, 20), ("-" * 16, 16), ("-" * 8, 8), ("-" * 18, 18), ("-" * 12, 12), ("-" * 12, 12), ("-" * 8, 8), ("-" * 40, 40)])

    for row in scored_rows[-max(1, args.limit) :]:
        _print_row(
            [
                (row.get("timestamp", ""), 20),
                (row.get("track", ""), 16),
                (row.get("status", ""), 8),
                (row.get("primary_metric_name", ""), 18),
                (row.get("primary_metric_value", ""), 12),
                (row.get("score", ""), 12),
                (row.get("commit", ""), 8),
                (row.get("description", ""), 40),
            ]
        )

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Autoresearch helper loop for Teligence")
    sub = p.add_subparsers(dest="subcommand", required=True)

    p_init = sub.add_parser("init", help="Initialize research/results.tsv")
    p_init.add_argument("--results", default="research/results.tsv")
    p_init.set_defaults(func=command_init)

    p_run = sub.add_parser("run", help="Run one experiment and record result row")
    p_run.add_argument("--track", required=True, choices=VALID_TRACKS)
    p_run.add_argument("--command", default="", help="Shell command for one experiment")
    p_run.add_argument("--summary-path", default="", help="Path to summary.json for this run")
    p_run.add_argument("--run-name", default="", help="Auto-build command/summary path if --command is omitted")
    p_run.add_argument("--dataset", default="tinyshakespeare", help="Dataset for auto-built gpt_text command")
    p_run.add_argument("--extra-args", default="", help="Extra APP_ARGS for proof tracks in auto-built command")
    p_run.add_argument("--description", required=True, help="Short experiment description")
    p_run.add_argument("--timeout-sec", type=int, default=1800)
    p_run.add_argument("--min-delta", type=float, default=1e-9)
    p_run.add_argument("--results", default="research/results.tsv")
    p_run.add_argument("--log-path", default="")
    p_run.set_defaults(func=command_run)

    p_leader = sub.add_parser("leaderboard", help="Show best and recent research runs")
    p_leader.add_argument("--results", default="research/results.tsv")
    p_leader.add_argument("--limit", type=int, default=10)
    p_leader.set_defaults(func=command_leaderboard)

    return p


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
