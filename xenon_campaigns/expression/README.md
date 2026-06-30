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

## Reproduce

```bash
pip install numpy
PYTHONPATH=$PWD python xenon_campaigns/expression/run_expression_pipeline.py
python -m pytest xenon/tests/test_expression_to_mechanism.py -p no:cacheprovider -o addopts=""
```
