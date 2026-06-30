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

## Status vs the real-data milestone

This delivers the **executable real-data path** (option c): point `--matrix` at the genuine
E-MTAB-16935 processed matrix and it runs end-to-end. What is **not** yet done — and must not be
claimed — is a real biological result: that still requires the actual counts (EBI egress is blocked
here) plus a production-grade module detector (option a) and defensible state-variable semantics. The
metadata parser is, however, already validated on the **real** SRA run table for this study.
