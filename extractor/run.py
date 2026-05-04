"""CLI entrypoint for the Empirical Evidence Extractor."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .classifier import SentenceClassifier
from .io import ConversationLoader, LedgerWriter, SummaryWriter


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Empirical Evidence Extractor")
    parser.add_argument("--input", required=True, help="Path to conversations.json export")
    parser.add_argument("--out", required=True, help="Output directory for artifacts")
    parser.add_argument("--log-level", default="INFO", help="Logging level (DEBUG, INFO, ...)")
    return parser


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging(args.log_level)

    input_path = Path(args.input)
    output_dir = Path(args.out)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = ConversationLoader(input_path)
    classifier = SentenceClassifier()
    messages = loader.load()
    sentence_records = classifier.classify_messages(messages, input_path)

    ledger_writer = LedgerWriter(output_dir)
    summary_writer = SummaryWriter(output_dir)

    ledger_path = ledger_writer.write(sentence_records)
    summary_path = summary_writer.write(sentence_records)

    logging.info("Ledger written to %s", ledger_path)
    logging.info("Summary written to %s", summary_path)


if __name__ == "__main__":
    main()
