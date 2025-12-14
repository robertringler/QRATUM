# QuASIM Documentation Package Manifest

**Generated:** 2025-12-14

## Deliverables

1. **Executive Summary:** `executive_summary/EXECUTIVE_SUMMARY.md`
2. **Technical White Paper:** `technical_white_paper/TECHNICAL_WHITE_PAPER.md`
3. **Visualizations:** 148 files in `visualizations/`

## Directory Structure

```
output_package/
├── executive_summary/
│   └── EXECUTIVE_SUMMARY.md
├── technical_white_paper/
│   └── TECHNICAL_WHITE_PAPER.md
├── visualizations/
│   ├── module_dependency_graph.png
│   ├── bm_001_execution_flow.png
│   ├── performance_comparison.png
│   └── architecture/
│   └── benchmarks/
│   └── tensor_networks/
│   └── statistical_analysis/
│   └── hardware_metrics/
│   └── reproducibility/
│   └── compliance/
└── MANIFEST.md
```

## Reproduction

To regenerate this package:

```bash
python3 scripts/generate_documentation_package.py \
  --repo-path . \
  --output-dir output_package
```
