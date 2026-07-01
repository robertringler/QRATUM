# Module detection report — mode: `gene_sets`

- modules: 3
- mean within-module correlation: -0.022
- between-module correlation: -0.033

## Per-module quality
| module | role | size | within-corr | separation | enrichment | trend | rep-stability | dropout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| maternal | maternal | 6 | 0.133 | 0.167 | 1.0 | None | 0.987 | 0.0 |
| polya | polya | 2 | 0.0 | 0.033 | 1.0 | None | 1.0 | 0.0 |
| zygotic | zygotic | 5 | -0.2 | -0.167 | 1.0 | None | 0.997 | 0.0 |

## Honesty flags
- weak coherence: True
- flat trend: False
- unstable replicates: False
- biologically unlabeled: False
- **mechanism inference exploratory-only: True**

## Warnings
- state variables are biologically imposed, not discovered from expression structure
- module structure is weak; mechanism inference is exploratory only

## Interpretation limits
- Module activities are abstract state variables, not physical concentrations.
- High within-module correlation from k-means/correlation modes is expected by construction; the silhouette score and between-module separation are the honest quality signals for unsupervised modes.
