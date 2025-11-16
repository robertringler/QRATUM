"""Command line interface for the ChatGPT export scraper."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .loader import find_export_root, load_raw_conversations
from .normalize import build_ledger, summarize_models, write_csv, write_jsonl


def build_parser() -> argparse.ArgumentParser:
    """Create the argument parser used by the CLI."""

    parser = argparse.ArgumentParser(description="Normalize a ChatGPT export into JSONL/CSV outputs.")
    parser.add_argument("--export-path", type=Path, required=True, help="Path to the export directory or .zip file")
    parser.add_argument("--out-ledger", type=Path, required=True, help="Destination JSONL ledger path")
    parser.add_argument("--out-summary", type=Path, required=True, help="Destination CSV summary path")
    parser.add_argument("--top-models", type=int, default=5, help="Number of models to display in the summary output")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    """Entry point for the CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    export_root = find_export_root(args.export_path)
    conversations = load_raw_conversations(export_root)
    turns, summaries = build_ledger(conversations)
    write_jsonl(turns, args.out_ledger)
    write_csv(summaries, args.out_summary)

    start = min(turn.timestamp for turn in turns) if turns else None
    end = max(turn.timestamp for turn in turns) if turns else None
    print(f"Processed {len(summaries)} conversations with {len(turns)} total messages.")
    if start and end:
        print(f"Time span: {start.isoformat()} — {end.isoformat()}")
    top_models = summarize_models(turns, args.top_models)
    if top_models:
        print("Top models:")
        for model, count in top_models:
            print(f"  {model}: {count}")


if __name__ == "__main__":  # pragma: no cover
    main()
