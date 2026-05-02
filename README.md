# QRATUM

A deterministic LLM behavioral observability and evaluation framework for detecting drift, discovering token clusters, and running CI-gated regression evaluations across prompts, personas, and decoding configurations.

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](LICENSE)

---

## 1. Overview

QRATUM is a modular observability system for analyzing and testing LLM behavior under different prompts, personas, and decoding configurations. It produces measurable, reproducible signals about how a model's outputs shift relative to a baseline run, and it exposes those signals through structured reports that can fail a CI build when regressions exceed configured thresholds.

The problem it addresses:

* LLM outputs change silently when prompts, personas, system instructions, sampling parameters, or model versions change.
* Ad-hoc evaluation makes those changes hard to detect, attribute, or reproduce.
* There is no shared substrate for treating "this persona drifted from baseline on these prompts" as a build-failing signal.

QRATUM treats every measurement as a baseline-paired comparison. A persona-conditioned run is only meaningful relative to a matched baseline run on the same prompts with the same decoding settings; lift, anomaly rates, and regression verdicts are all computed against that pair. Without a baseline, no claim of drift or regression is made.

Determinism is a hard requirement: the package never reads the global RNG, never depends on wall-clock time, and emits canonical-JSON output, so two runs on the same inputs and configuration produce byte-identical reports.

---

## 2. System Architecture

QRATUM is composed of three layers under `qratum_framework/observability/`. Each layer is independently usable and exposes a stable Python API plus a CLI subcommand.

### Layer A — Drift Engine (`qratum_framework/observability/drift/`)

Detects behavioral drift in model outputs using three measurable signals:

* **Semantic relevance scoring** between output tokens and the input prompt.
* **Log-probability lift** of each token under the persona run vs. the baseline run.
* **Cluster membership scoring** against the emergent token groups produced by Layer B.

Per-token signals are combined and bucketed by a deterministic severity classifier into `low` / `medium` / `high`. Output is a `DriftReport` containing `anomaly_detected`, `off_topic_tokens`, `cluster_id`, `lift_score`, and `severity`, plus diagnostic fields for downstream consumers.

### Layer B — Cluster Discovery Engine (`qratum_framework/observability/clusters/`)

Discovers emergent token groupings from a persona-conditioned token stream and a paired baseline stream:

* **Sliding-window co-occurrence graphs** built incrementally and seeded deterministically.
* **Persona-conditioned vs. baseline lift** comparison (Laplace-smoothed) to weight edges.
* **Deterministic community detection** (Louvain when `networkx` is available, with a spectral fallback).
* **Cluster stability scoring** across consecutive windows so unstable groupings are visible as such.

Output is a `ClusterEngineState` snapshot containing scored `Cluster` records with `id`, `tokens`, `lift`, `density`, and `stability`.

### Layer C — Evaluation Harness (`qratum_framework/observability/eval/`)

Runs structured evaluation matrices:

* **Prompt × model × persona** Cartesian testing under a single decoding configuration.
* **Baseline-paired comparisons** — every non-baseline row is matched to a baseline row on the same prompt and decoding settings.
* **Regression detection** with explicit CI failure thresholds on anomaly rate, cluster activation index, and lift-cluster rate.
* **Structured JSON + Markdown reporting** for machine ingestion and human review.

### Architecture diagram

```
                ┌──────────────────────────────────────────────┐
                │           Evaluation Harness  (C)            │
                │   prompt × model × persona matrix runner     │
                │   baseline pairing · regression verdicts     │
                │   JSON + Markdown reports · CI exit codes    │
                └──────────────────┬───────────────────────────┘
                                   │ invokes per row
                                   ▼
            ┌───────────────────────┴───────────────────────┐
            │                                               │
            ▼                                               ▼
┌──────────────────────────┐                ┌────────────────────────────┐
│      Drift Engine (A)    │ ◀── reads ──── │  Cluster Discovery (B)     │
│  relevance · log-lift ·  │  edge weights  │  sliding-window graph      │
│  cluster membership ·    │  + memberships │  persona vs. baseline lift │
│  severity classifier     │                │  community detection +     │
│  → DriftReport           │                │  stability scoring         │
└──────────────────────────┘                │  → ClusterEngineState      │
                                            └────────────────────────────┘
                                   │
                                   ▼
                ┌──────────────────────────────────────────┐
                │   MerkleLedger (qratum_framework.trace)  │
                │   optional tamper-evident audit log      │
                └──────────────────────────────────────────┘
```

---

## 3. Core Concepts

**Drift detection.** A run "drifts" when its measured signals (relevance, log-lift, cluster membership) deviate from those of its paired baseline beyond configured thresholds. Drift is a per-utterance, per-token statistical statement, not a claim about model intent.

**Lift vs. baseline.** Lift is the Laplace-smoothed log-ratio of an event's frequency under the persona-conditioned run to its frequency under the baseline run. It is undefined without a baseline. All downstream scores in QRATUM are derived from baseline-paired lift, never from raw persona frequencies alone.

**Cluster co-occurrence.** Clusters are statistical groupings of tokens that co-occur within a sliding window more often under the persona run than under the baseline run. They are nothing more than connected, lift-weighted subgraphs of a co-occurrence graph; they carry an id, a token list, a lift, a density, and a stability score across windows. They are not entities, and they encode no hidden structure beyond observed token-pair frequencies.

**Relevance scoring.** A bounded score that quantifies how strongly an output token matches the input prompt, computed by a pluggable scorer. The default scorer is deterministic and embedding-free; alternative scorers (for example, embedding-cosine) can be supplied without changing the rest of the pipeline.

**Regression testing in CI.** A regression is a non-baseline row whose anomaly rate, cluster activation index, or lift-cluster rate exceeds the configured threshold relative to its paired baseline row. The eval harness records a `RegressionVerdict` per row and returns a non-zero exit status under `--regression` when any verdict is positive.

---

## 4. Installation

QRATUM requires Python 3.10 or newer.

```bash
pip install qratum
```

To install from a checkout of this repository:

```bash
pip install .
```

Optional dependencies enable richer Layer B and Layer A backends:

```bash
# Louvain community detection (otherwise the spectral fallback is used)
pip install networkx

# Spectral clustering fallback and embedding-based scorers
pip install scikit-learn
```

Both extras are optional — the default install runs end-to-end with only the standard library and NumPy.

---

## 5. Quick Start

The example below runs the eval harness on built-in default prompts and personas, computes drift signals against a baseline, and writes a JSON + Markdown report.

```python
from qratum_framework.observability import (
    EvalRunner,
    default_personas,
    default_prompts,
)
from qratum_framework.observability.eval.reports import write_reports

runner = EvalRunner(
    prompts=default_prompts(),
    personas=default_personas(),
    seed=0,
)

report = runner.run()

write_reports(
    report,
    json_path="qratum_report.json",
    md_path="qratum_report.md",
)

if report.has_regression:
    raise SystemExit(1)
```

The same workflow from the command line:

```bash
qratum eval \
    --report-json qratum_report.json \
    --report-md  qratum_report.md \
    --regression
```

To inspect a single utterance against a baseline without running the full matrix:

```bash
qratum drift \
    --persona-tokens   tokens/persona.txt \
    --baseline-tokens  tokens/baseline.txt \
    --utterance        "Summarize the change log for release 1.4." \
    --persona          support_agent \
    --json
```

---

## 6. Output Schema

All outputs are JSON-serializable, key-sorted, and stable across runs.

### Drift detection result

```json
{
  "anomaly_detected": true,
  "off_topic_tokens": ["foo", "bar"],
  "cluster_id": ["C03", "C12"],
  "lift_score": 2.41,
  "severity": "medium",
  "persona": "support_agent",
  "utterance_hash": "sha256:…",
  "thresholds": {
    "medium_density": 0.10,
    "medium_peak_lift": 1.0,
    "high_density":   0.30,
    "high_peak_lift": 2.0
  }
}
```

### Cluster discovery result

```json
{
  "schema_version": "1.0",
  "window_index": 7,
  "clusters": [
    {
      "id": "C12",
      "tokens": ["refund", "ticket", "policy"],
      "lift": 3.82,
      "density": 0.41,
      "stability": 0.91
    }
  ],
  "edge_weights": { "...": "..." },
  "accumulator":  { "...": "..." }
}
```

### Evaluation report

```json
{
  "schema_version": "1.0",
  "summary": {
    "rows": 24,
    "regressions": 2,
    "anomaly_rate_mean": 0.07
  },
  "thresholds": {
    "anomaly_rate":            0.20,
    "cluster_activation_index": 0.50,
    "lift_cluster_rate":        0.30
  },
  "rows": [
    {
      "prompt":  "P01",
      "persona": "support_agent",
      "is_baseline": false,
      "drift": { "...": "..." }
    }
  ],
  "verdicts": [
    {
      "prompt":  "P01",
      "model":   "support_agent",
      "is_baseline": false,
      "regression":  true,
      "reasons":     ["anomaly_rate>0.20"],
      "anomaly_rate":              0.27,
      "cluster_activation_index":  0.55,
      "lift_cluster_rate":         0.31
    }
  ]
}
```

---

## 7. CLI Usage

The `qratum` console script exposes one subcommand per layer.

### `qratum clusters`

Run Layer B cluster discovery on a persona token file paired with a baseline token file.

```bash
qratum clusters \
    --persona-tokens   tokens/persona.txt \
    --baseline-tokens  tokens/baseline.txt \
    --window 64 \
    --seed   0 \
    --json
```

Emits a `ClusterEngineState` snapshot to stdout when `--json` is set.

### `qratum drift`

Run Layer A drift detection on a single utterance, given a persona token stream and its paired baseline.

```bash
qratum drift \
    --persona-tokens   tokens/persona.txt \
    --baseline-tokens  tokens/baseline.txt \
    --utterance        "..." \
    --persona          support_agent \
    --json
```

Emits a `DriftReport` to stdout when `--json` is set.

### `qratum eval`

Run the Layer C evaluation matrix and emit a regression report.

```bash
qratum eval \
    --report-json qratum_report.json \
    --report-md   qratum_report.md \
    --regression \
    --json
```

Flags:

* `--report-json PATH` — write the structured JSON report to `PATH`.
* `--report-md PATH` — write the Markdown report to `PATH`.
* `--json` — also print the JSON report to stdout.
* `--regression` — exit with a non-zero status code if any non-baseline row triggers a regression verdict. This is the flag that turns the harness into a CI gate.

---

## 8. CI Integration

QRATUM is designed to be wired into a CI workflow as a build step. A typical configuration:

```yaml
- name: QRATUM regression eval
  run: |
    qratum eval \
        --report-json qratum_report.json \
        --report-md   qratum_report.md \
        --regression
```

**Regression detection workflow.**

1. The eval harness loads the configured prompt × model × persona matrix.
2. For each non-baseline row, it locates the matching baseline row (same prompt, same decoding configuration).
3. It runs Layer A drift detection on each row using Layer B clusters built from the paired baseline stream.
4. It computes anomaly rate, cluster activation index, and lift-cluster rate per row, and compares each to the configured thresholds.
5. It writes a `RegressionVerdict` per row into the report.

**Baseline pairing requirement.** Every non-baseline row must have exactly one matching baseline row. If a baseline is missing, the harness rejects the matrix at load time rather than silently emitting a one-sided result. Regression verdicts are never produced from un-paired data.

**Failure conditions.** Under `--regression`, the CLI exits with a non-zero status when any of the following holds for any non-baseline row:

* `anomaly_rate` exceeds the configured threshold.
* `cluster_activation_index` exceeds the configured threshold.
* `lift_cluster_rate` exceeds the configured threshold.

The full set of reasons for each failing row is recorded in the report's `verdicts[*].reasons` field.

---

## 9. Design Philosophy

QRATUM is a statistical measurement system, not a semantic ontology.

* All outputs are measurable signals over observed tokens, log-probabilities, and co-occurrence counts. Nothing in the pipeline depends on or asserts an interpretation of those signals.
* Clusters are subgraphs of a co-occurrence graph weighted by baseline-paired lift. They have ids, tokens, lift, density, and stability — and nothing else.
* The system makes no assumption about hidden structure inside a model. It only reports what is observable in the token stream and the log-probability stream that the model emits.
* Determinism, baseline pairing, and explicit thresholds are non-negotiable. They are what make the framework's outputs admissible as build-gating evidence.
* The three layers are independently usable and pluggable. Scorers, embeddings, and clustering methods are swappable without changing the contracts of the layers above them.
