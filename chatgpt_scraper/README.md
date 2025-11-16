# ChatGPT Export Scraper

This package ingests the JSON export that can be downloaded from the ChatGPT UI and
converts it into a normalized ledger for downstream analytics. It does **not** scrape the
live site or call any OpenAI APIs; all processing happens on local files that you provide.

## Features

* Detects and loads both directory- and zip-based exports.
* Normalizes every message into a consistent schema with timestamps, roles, and metadata.
* Produces a JSON Lines "knowledge ledger" as well as a CSV conversation summary.
* Includes a small CLI so the full pipeline can be executed with a single command.
* Ships with unit tests and synthetic export fixtures so the behavior is easy to verify.

## Quickstart

```bash
python -m chatgpt_scraper \
  --export-path /path/to/unzipped/export \
  --out-ledger chatgpt_ledger.jsonl \
  --out-summary chatgpt_conversations.csv
```

Use the `--help` flag for additional options.

## Development

Run the included test-suite from the project root:

```bash
cd chatgpt_scraper
pytest
```

## License

MIT
