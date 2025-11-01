"""Run static analysis passes for firmware and runtime sources."""
from __future__ import annotations

import pathlib
import subprocess
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def run_clang_format() -> None:
    sources = [
        *REPO_ROOT.glob("fw/**/*.c"),
        *REPO_ROOT.glob("drivers/**/*.c"),
        *REPO_ROOT.glob("runtime/libquasim/src/**/*.cpp"),
        *REPO_ROOT.glob("runtime/libquasim/include/**/*.hpp"),
    ]
    sources = [str(path) for path in sources if path.is_file()]
    if not sources:
        print("No sources discovered for clang-format linting.")
        return
    cmd = ["clang-format", "--dry-run", "--Werror", *sources]
    print("Running", " ".join(cmd))
    subprocess.check_call(cmd)


def run_python_style() -> None:
    cmd = [sys.executable, "-m", "py_compile", *map(str, REPO_ROOT.glob("**/*.py"))]
    print("Running", " ".join(cmd[:4]), "... python bytecode check")
    subprocess.check_call(cmd)


def main() -> None:
    run_python_style()
    try:
        run_clang_format()
    except FileNotFoundError:
        print("clang-format not found; skipping C/C++ style enforcement", file=sys.stderr)


if __name__ == "__main__":
    main()
