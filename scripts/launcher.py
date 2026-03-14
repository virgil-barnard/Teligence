import os
import shlex
import subprocess
import sys


ENTRYPOINTS = {
    "gpt": "scripts.gpt_text",
    "gpt_text": "scripts.gpt_text",
    "proof_factoring": "experiments.proof_factoring_gpt",
    "proof_factoring_gpt": "experiments.proof_factoring_gpt",
    "proof_affine": "experiments.proof_affine_gpt",
    "proof_affine_gpt": "experiments.proof_affine_gpt",
    # Legacy aliases
    "proof_math": "experiments.proof_factoring_gpt",
    "proof_agent": "experiments.proof_factoring_gpt",
    "icarus_affine": "experiments.proof_affine_gpt",
    "icarus_projective_v2": "experiments.proof_affine_gpt",
}


def main() -> int:
    entry = os.environ.get("APP_ENTRY", "gpt").strip().lower()
    if entry not in ENTRYPOINTS:
        valid = ", ".join(sorted(ENTRYPOINTS))
        print(f"Invalid APP_ENTRY='{entry}'. Expected one of: {valid}")
        return 2

    module_name = ENTRYPOINTS[entry]
    app_args_raw = os.environ.get("APP_ARGS", "").strip()
    app_args = shlex.split(app_args_raw) if app_args_raw else []

    command = [sys.executable, "-m", module_name, *app_args, *sys.argv[1:]]
    print(f"launcher: APP_ENTRY={entry} -> {module_name}")
    if app_args:
        print(f"launcher: APP_ARGS={app_args_raw}")

    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
