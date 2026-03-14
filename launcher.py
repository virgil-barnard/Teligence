import os
import shlex
import subprocess
import sys


ENTRYPOINTS = {
    "gpt": "gpt.py",
    "proof_agent": "proof_gpt_agent_onefile.py",
    "icarus_projective_v2": "icarus_projective_actionptr_v2.py",
}


def main() -> int:
    entry = os.environ.get("APP_ENTRY", "gpt").strip().lower()
    if entry not in ENTRYPOINTS:
        valid = ", ".join(sorted(ENTRYPOINTS))
        print(f"Invalid APP_ENTRY='{entry}'. Expected one of: {valid}")
        return 2

    script = ENTRYPOINTS[entry]
    app_args_raw = os.environ.get("APP_ARGS", "").strip()
    app_args = shlex.split(app_args_raw) if app_args_raw else []

    command = [sys.executable, script, *app_args, *sys.argv[1:]]
    print(f"launcher: APP_ENTRY={entry} -> {script}")
    if app_args:
        print(f"launcher: APP_ARGS={app_args_raw}")

    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
