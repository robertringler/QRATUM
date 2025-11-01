"""CLI wrapper to execute the QuASIM micro-benchmark."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_SCRIPT = REPO_ROOT / "benchmarks" / "quasim_bench.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the QuASIM benchmark harness")
    parser.add_argument("--batches", type=int, default=32)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--dimension", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = dict(**os.environ)
    python_path = env.get("PYTHONPATH", "")
    runtime_paths = [str(REPO_ROOT / "runtime" / "python"), str(REPO_ROOT / "quantum")]
    env["PYTHONPATH"] = ":".join(runtime_paths + ([python_path] if python_path else []))

    cmd = [sys.executable, str(BENCH_SCRIPT)]
    cmd.extend(["--batches", str(args.batches)])
    cmd.extend(["--rank", str(args.rank)])
    cmd.extend(["--dimension", str(args.dimension)])
    cmd.extend(["--repeat", str(args.repeat)])
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
