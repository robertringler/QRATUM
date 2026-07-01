# Module detection report — mode: `silhouette_kmeans`

- modules: 2
- mean within-module correlation: 0.996
- between-module correlation: -0.996
- silhouette: 0.9536 (selected k=4, scores={2: 0.7464, 3: 0.815, 4: 0.9536, 5: 0.5848, 6: 0.6715, 7: 0.6795, 8: 0.4353})

## Per-module quality
| module | role | size | within-corr | separation | enrichment | trend | rep-stability | dropout |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| kmeans_module_0 | maternal | 6 | 0.996 | 1.992 | 1.0 | 1.007 | 0.985 | 0.0 |
| kmeans_module_1 | zygotic | 5 | 0.997 | 1.994 | 1.0 | 1.735 | 0.991 | 0.0 |

## Honesty flags
- weak coherence: False
- flat trend: False
- unstable replicates: False
- biologically unlabeled: False
- **mechanism inference exploratory-only: False**

## Interpretation limits
- Module activities are abstract state variables, not physical concentrations.
- High within-module correlation from k-means/correlation modes is expected by construction; the silhouette score and between-module separation are the honest quality signals for unsupervised modes.
