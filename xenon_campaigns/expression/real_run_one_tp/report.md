# XENON expression→mechanism ingest report

- **Matrix:** `xenon_campaigns/expression/example_real/counts_one_tp.tsv`
- **Genes × samples:** 13 × 2
- **Normalization:** tpm
- **Distinct timepoints:** 1 [17.0]
- **Mechanism recovery attempted:** False

## Modules
- mode: `gene_sets` (3 modules); see `module_detection_report.md` for quality diagnostics
  - **maternal** (maternal): activity by sample = [33.5, 32.667]
  - **polya** (polya): activity by sample = [50.0, 50.0]
  - **zygotic** (zygotic): activity by sample = [65.6, 66.0]

## Inference
- **REFUSED** — insufficient temporal structure for mechanism recovery; ingestion-only validation

## Limitations
- Module activities are **abstract state variables, not physical concentrations**; XENON's thermodynamic/conservation checks are not physically meaningful in this mode.
- Mechanism recovery requires ≥2 distinct timepoints; with sparse timepoints K is weakly identified and any recovered mechanism is provisional.

## Warnings
- state variables are biologically imposed, not discovered from expression structure
- module structure is weak; mechanism inference is exploratory only
- insufficient temporal structure for mechanism recovery; ingestion-only validation
