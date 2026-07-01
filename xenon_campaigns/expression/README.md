# Expression → XENON mechanism inference

Implements the requested pipeline against XENON's real inference machinery:

```
FASTQ / processed counts
    → gene expression matrix        (load_expression_matrix)
    → select gene modules           (select_gene_modules — numpy co-expression k-means)
    → construct state variables     (construct_state_variables — module activities)
    → XENON mechanism inference     (infer_mechanism — Bayesian model comparison)
```

Code: [`xenon/bioinformatics/expression_to_mechanism.py`](../../xenon/bioinformatics/expression_to_mechanism.py)
· Driver: [`run_expression_pipeline.py`](run_expression_pipeline.py)
· Tests: `xenon/tests/test_expression_to_mechanism.py`

## Data status (read first)

The real **E-MTAB-16935** counts are **not available in this environment**: EBI/ENA egress is blocked
by the session network policy (the `curl`/`WebFetch` attempts returned policy `403`), and no count
matrix was uploaded (only the SRA run-table CSV). So the driver **generates a surrogate gene×sample
matrix from a known ground-truth program and verifies the pipeline recovers it** — the same in-silico
recovery ethos used elsewhere in this repo. **This is not a biological result about mouse embryos.**

To run on real data, drop the processed/TPM/count matrix (genes as rows, samples as columns, TSV/CSV)
in and call `load_expression_matrix(path)` — the rest is unchanged.

## What the demo shows

Ground truth themed for E-MTAB-16935 (2-cell mouse embryo, **zygotic genome activation**):
- Module **M** (maternal): high early, decays by the late (17 h) timepoint.
- Module **Z** (zygotic): activated late. Planted late ratio **Z:M = 2:1 ⇒ effective K = 2**.

Result (`expression_pipeline_results.json`), fully reproducible (`seed=42`):

| Step | Output |
|---|---|
| Modules | 2 modules, 100 genes each, **purity 1.0** (maternal vs zygotic perfectly separated) |
| State variables (late) | module_0 = 33.4, module_1 = 66.6 (the planted 1:2 ratio) |
| Observed ratio | **K\* ≈ 1.99** |
| XENON inference | recovers **K = 2.0** (matches planted) over candidates {0.25, 0.5, 1, 2, 4} |

## Honest scope & caveats

- **Module activities are abstract state variables, not physical concentrations.** Mapping them onto
  XENON's concentration observable is a modelling choice; XENON's thermodynamic/conservation checks
  are **not physically meaningful** in this mode. What is recovered is the *effective regulatory ratio*
  (here, the maternal→zygotic activation strength K), under stated assumptions.
- The surrogate has a clean two-program structure; **real expression is messier** — module count must
  be chosen/justified (e.g. WGCNA, silhouette), batch effects handled, and the module→state mapping
  validated. The k-means here is a deterministic placeholder, not a production module detector.
- This validates **plumbing + recovery**, i.e. that the pipeline inverts a planted program. Real
  biological inference additionally needs real counts, replicate/time-course structure, and a
  defensible state-variable semantics.

## Real-matrix ingest CLI (production path)

One command runs an actual counts/TPM matrix through the full pipeline:

```bash
python -m xenon.bioinformatics.expression_to_mechanism \
  --matrix    data/E-MTAB-16935_counts.tsv \
  --metadata  data/E-MTAB-16935.sdrf.txt \
  --gene-sets xenon_campaigns/expression/configs/mouse_ega_markers.tsv \
  --normalization tpm \
  --out       xenon_campaigns/expression/real_run/
```

Outputs (in `--out`): `normalized_expression.tsv`, `selected_modules.tsv`, `state_variables.tsv`,
`candidate_mechanisms.json`, `xenon_inference_results.json`, `report.md`, `summary.json`.

**Worked example (committed):** [`example_real/`](example_real/) holds a fake-real matrix (real mouse
marker symbols: maternal `Zar1/Npm2/Zp3/…`, zygotic `Zscan4d/Dux/Zfp352/…`, polyA-sensitive histones),
an SDRF-style metadata file, and a one-timepoint variant. Run results are in
[`real_run/`](real_run/) (recovers **K=2.0**) and [`real_run_one_tp/`](real_run_one_tp/) (correctly
**refuses** mechanism recovery).

### What it does (acceptance criteria)
- **Loads** real CSV/TSV genes×samples matrices; **deduplicates** genes (row-averaging), **imputes**
  missing values (per-gene mean).
- **Parses metadata** (SDRF *or* SRA run table — fuzzy column matching) for sample, timepoint/age,
  polyA± treatment, replicate, developmental stage, run accession; **validated against the real
  uploaded `SraRunTable.csv`** (correctly reads 17 h / polyA+ / 2-cell).
- **Matches** expression columns to metadata sample names (exact → case-insensitive → substring).
- **Normalization:** `raw`, `log1p`, `cpm`, `tpm` (pass-through). *Note: the normalization choice
  changes the state-variable scale and therefore the inferred K; use a linear scale (`tpm`/`raw`) when
  interpreting module-activity ratios.*
- **Builds state variables** from gene-set modules (maternal / zygotic-EGA / polyA-sensitive),
  timepoint-resolved.
- **Runs XENON inference** (candidate-K posterior) with an **identifiability warning** when timepoints
  are sparse, and **refuses mechanism-recovery** (ingestion-only) when there is `<2` distinct
  timepoints — emitting: *"insufficient temporal structure for mechanism recovery; ingestion-only
  validation."*

## Reproduce

```bash
pip install numpy
# synthetic planted-recovery demo
PYTHONPATH=$PWD python xenon_campaigns/expression/run_expression_pipeline.py
# real-matrix ingest CLI on the committed fake-real example
PYTHONPATH=$PWD python -m xenon.bioinformatics.expression_to_mechanism \
  --matrix xenon_campaigns/expression/example_real/counts.tsv \
  --metadata xenon_campaigns/expression/example_real/sdrf.tsv \
  --gene-sets xenon_campaigns/expression/configs/mouse_ega_markers.tsv \
  --normalization tpm --out xenon_campaigns/expression/real_run/
python -m pytest xenon/tests/test_expression_to_mechanism.py xenon/tests/test_expression_ingest.py \
  -p no:cacheprovider -o addopts=""
```

## Module detection (production-grade, option a)

Module discovery is now a defensible, auditable layer
([`xenon/bioinformatics/module_detection.py`](../../xenon/bioinformatics/module_detection.py)) with
five modes, per-module quality scoring, and honesty warnings — so a mechanism is never presented as
more solid than the module structure supports.

| `--module-mode` | What it does |
|---|---|
| `gene_sets` | marker sets → modules (most interpretable; **biologically imposed**) |
| `correlation_modules` | co-expression graph (correlation threshold → connected components) |
| `silhouette_kmeans` | automatic *k* via silhouette across `--min-k..--max-k` |
| `hybrid` | markers first, then cluster the remaining genes |
| `manual` | user-supplied `gene→module` mapping (`--manual-modules`) |

Extra CLI flags: `--module-mode`, `--min-k`, `--max-k`, `--correlation-threshold`,
`--min-module-size`, `--manual-modules`.

Per-module quality (in `module_quality.tsv` / `module_diagnostics.json`): size, within-module
correlation, between-module correlation, separation, silhouette, marker enrichment, timepoint trend,
replicate stability, dropout fraction. New outputs: `selected_modules.tsv` (gene→module),
`module_quality.tsv`, `module_activity.tsv`, `module_diagnostics.json`, `module_detection_report.md`.

**Honesty warnings** (emitted, and they downgrade inference to *exploratory* where appropriate):
- weak coherence → *"module structure is weak; mechanism inference is exploratory only"*
- markers-only → *"state variables are biologically imposed, not discovered from expression structure"*
- coherent but no labels → *"modules are statistically coherent but biologically unlabeled"*
- flat dynamics → *"insufficient dynamic signal for mechanism recovery"*
- poor replicate agreement → *"module activities are unstable across replicates"*

**Cross-check on the worked example:** `gene_sets`/`hybrid` and unsupervised `silhouette_kmeans` both
recover **K=2.0**, and the unsupervised modules show **marker enrichment 1.0** against the maternal/
zygotic sets — i.e. the discovered modules *align with the known biology*. See
[`real_run/`](real_run/) (hybrid), [`real_run_kmeans/`](real_run_kmeans/) (unsupervised independent
check), and [`real_run_one_tp/`](real_run_one_tp/) (refusal).

> **Note on unsupervised within-module correlation:** k-means/correlation modes produce
> internally-correlated clusters *by construction*, so a high within-module correlation is not by
> itself evidence of real structure — the **silhouette** and **between-module separation** are the
> honest quality signals, and a weak/random matrix correctly trips `weak_coherence → exploratory`.

### Recommended real-data protocol (per your plan)
1. Drop in the actual E-MTAB-16935 counts/TPM on `--matrix`.
2. Run `--module-mode hybrid` with the EGA markers.
3. Run `--module-mode correlation_modules` (and `silhouette_kmeans`) as **independent checks**.
4. Compare whether discovered modules align with maternal/zygotic biology (marker enrichment).
5. Only then consider multi-module mechanisms.

## Status vs the real-data milestone

This delivers the **executable real-data path** (option c): point `--matrix` at the genuine
E-MTAB-16935 processed matrix and it runs end-to-end. What is **not** yet done — and must not be
claimed — is a real biological result: that still requires the actual counts (EBI egress is blocked
here) plus a production-grade module detector (option a) and defensible state-variable semantics. The
metadata parser is, however, already validated on the **real** SRA run table for this study.
