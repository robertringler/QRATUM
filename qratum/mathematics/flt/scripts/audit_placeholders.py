#!/usr/bin/env python3
"""Fail if Lean source contains common proof placeholders."""

from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1] / "lean"
FORBIDDEN = {
    "sorry": re.compile(r"\bsorry\b"),
    "admit": re.compile(r"\badmit\b"),
    "axiom": re.compile(r"^\s*axiom\b", re.MULTILINE),
    "unsafe theorem": re.compile(r"\bunsafe\s+theorem\b"),
}


def main() -> int:
    failures: list[str] = []
    for path in sorted(ROOT.rglob("*.lean")):
        text = path.read_text(encoding="utf-8")
        for name, pattern in FORBIDDEN.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                failures.append(f"{path.relative_to(ROOT)}:{line}: forbidden {name}")

    if failures:
        print("QRATUM-FLT placeholder audit FAILED", file=sys.stderr)
        print("\n".join(failures), file=sys.stderr)
        return 1

    print("QRATUM-FLT placeholder audit PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
