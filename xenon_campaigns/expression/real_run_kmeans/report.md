# XENON expression→mechanism ingest report

- **Matrix:** `xenon_campaigns/expression/example_real/counts.tsv`
- **Genes × samples:** 13 × 8
- **Normalization:** tpm
- **Distinct timepoints:** 4 [0.0, 6.0, 12.0, 17.0]
- **Mechanism recovery attempted:** True

## Modules
- mode: `silhouette_kmeans` (2 modules); see `module_detection_report.md` for quality diagnostics
  - **kmeans_module_0** (maternal): activity by sample = [100.5, 99.5, 79.5, 76.5, 56.0, 53.333, 33.5, 32.667]
  - **kmeans_module_1** (zygotic): activity by sample = [5.2, 5.4, 23.4, 23.8, 44.6, 45.0, 65.6, 66.0]

## Inference
- recovered mechanism: **K=2.0** (K=2.0)
- observed activation ratio K* = 1.9889
- posterior: {'K=0.25': 0.0, 'K=0.5': 0.0, 'K=1.0': 0.0, 'K=2.0': 1.0, 'K=4.0': 0.0}

## Limitations
- Module activities are **abstract state variables, not physical concentrations**; XENON's thermodynamic/conservation checks are not physically meaningful in this mode.
- Mechanism recovery requires ≥2 distinct timepoints; with sparse timepoints K is weakly identified and any recovered mechanism is provisional.
