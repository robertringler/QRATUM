"""Simple assembler/disassembler for the GB10 tensor ISA."""
from __future__ import annotations

import argparse
import json

MNEMONICS = {
    "TENSORMUL": 0x01,
    "TENSORADD": 0x02,
    "TENSORMAD": 0x03,
}


def assemble(lines):
    words = []
    for line in lines:
        line = line.strip().upper()
        if not line or line.startswith("//"):
            continue
        opcode = MNEMONICS.get(line)
        if opcode is None:
            raise ValueError(f"Unknown opcode {line}")
        words.append(opcode)
    return words


def main() -> None:
    parser = argparse.ArgumentParser(description="GB10 tensor ISA assembler")
    parser.add_argument("source")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    words = assemble(open(args.source).read().splitlines())
    if args.json:
        print(json.dumps(words))
    else:
        for word in words:
            print(f"0x{word:02x}")


if __name__ == "__main__":
    main()
