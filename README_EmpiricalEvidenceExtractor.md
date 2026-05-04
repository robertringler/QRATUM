# Empirical Evidence Extractor

The **Empirical Evidence Extractor** ingests exported ChatGPT `conversations.json` files, breaks every assistant response into sentences, classifies the scientific quality of each statement, tags relevant domains, and produces traceable artifacts for downstream analysis.

## Features

- Deterministic sentence segmentation implemented with spaCy's lightweight sentencizer.
- Heuristic multi-class classification for empirical confidence levels.
- Domain tagging across physics, math, computer science, engineering, biology, biometrics, chemistry, and space systems.
- Artifact generation:
  - `empirical_knowledge_ledger.jsonl`: JSON lines ledger with UUID IDs and traceability metadata.
  - `corpus_classification_summary.csv`: aggregate counts and top domain associations.
- Rich CLI (`python -m extractor.run`) with configurable logging.
- Modular, fully type annotated codebase with pytest coverage and mypy/flake8 compatibility.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
python -m extractor.run --input /path/to/conversations.json --out artifacts/
```

The command prints logging information describing how many messages were parsed and where the resulting artifacts were saved. If
spaCy is not available in the current environment, the segmenter automatically falls back to a deterministic regex splitter so
tests and offline workflows still succeed. Installing spaCy enables the more accurate sentencizer-based segmentation.

## Repository Layout

```
extractor/
  classifier.py   # segmentation + classification logic
  domains.py      # domain tagger
  io.py           # ingestion and artifact writers
  run.py          # CLI entrypoint
  models.py       # dataclasses shared across modules
extractor_demo.ipynb  # Example notebook demonstrating the workflow
requirements.txt      # Tooling dependencies
setup.cfg             # flake8/mypy configuration
tests/                # pytest suites per module
```

## Testing

```bash
pytest
```

For static analysis:

```bash
flake8 extractor tests
mypy extractor
```

## Demo Notebook

Open `extractor_demo.ipynb` to see an end-to-end demonstration using a tiny synthetic conversation that can run offline without additional setup.
