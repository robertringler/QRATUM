# Module detection report — mode: `hybrid`

- modules: 3
- mean within-module correlation: 0.498
- between-module correlation: -0.336

## Per-module quality
| module | role | size | within-corr | separation | enrichment | trend | rep-stability | dropout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| maternal | maternal | 6 | 0.996 | 1.332 | 1.0 | 1.007 | 0.985 | 0.0 |
| polya | polya | 2 | -0.5 | -0.164 | 1.0 | 0.02 | 1.0 | 0.0 |
| zygotic | zygotic | 5 | 0.997 | 1.334 | 1.0 | 1.735 | 0.991 | 0.0 |

## Honesty flags
- weak coherence: False
- flat trend: False
- unstable replicates: False
- biologically unlabeled: False
- **mechanism inference exploratory-only: False**

## Interpretation limits
- Module activities are abstract state variables, not physical concentrations.
- High within-module correlation from k-means/correlation modes is expected by construction; the silhouette score and between-module separation are the honest quality signals for unsupervised modes.
