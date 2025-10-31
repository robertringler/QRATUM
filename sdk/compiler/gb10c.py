"""Synthetic CUDA-like compiler frontend producing PTX-style IR."""
from __future__ import annotations

import argparse
import json
import pathlib
from dataclasses import dataclass


@dataclass
class CompileResult:
    entry: str
    ir: str


def compile_kernel(source: str, entry: str) -> CompileResult:
    ir = f"// GB10 PTX\n.entry {entry}() {{\n  // {len(source.splitlines())} lines\n}}\n"
    return CompileResult(entry=entry, ir=ir)


def main() -> None:
    parser = argparse.ArgumentParser(description="GB10 CUDA-compatible compiler")
    parser.add_argument("source", type=pathlib.Path)
    parser.add_argument("--entry", default="main")
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("a.ptx"))
    args = parser.parse_args()

    result = compile_kernel(args.source.read_text(), args.entry)
    args.out.write_text(result.ir)
    print(json.dumps({"entry": result.entry, "output": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
