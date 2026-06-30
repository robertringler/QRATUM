#!/usr/bin/env python3
"""Expression -> modules -> state variables -> XENON mechanism inference.

Demonstrates and validates the full pipeline the user specified:

    FASTQ / processed counts
        -> gene expression matrix
        -> select gene modules
        -> construct state variables
        -> XENON mechanism inference

DATA STATUS: the real E-MTAB-16935 counts are not available in this environment
(EBI/ENA egress is blocked and no matrix was uploaded). This driver therefore
GENERATES a surrogate gene x sample matrix from a KNOWN ground-truth program and
verifies the pipeline recovers it — exactly the in-silico recovery ethos used
elsewhere in this repo. It is NOT a biological finding about mouse embryos; it
validates that the plumbing recovers the planted mechanism. Swap in the real
matrix via `load_expression_matrix(<path>)` and the rest runs unchanged.

Ground truth (biologically themed for E-MTAB-16935: 2-cell mouse embryo, ZGA):
  - Module M (maternal): high early, decays by the late (17 h) timepoint.
  - Module Z (zygotic):  low early, activated late (zygotic genome activation).
  - Planted activation ratio at the late timepoint Z:M = 2:1  => effective K = 2.
The pipeline must (a) recover two co-expression modules, (b) read the ~2:1 late
activity ratio, and (c) have XENON infer the K=2 maternal->zygotic transition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xenon.bioinformatics.expression_to_mechanism import (
    construct_state_variables,
    infer_mechanism,
    load_expression_matrix,
    select_gene_modules,
)

SEED = 42
N_GENES = 200
TIMEPOINTS = ["t0h", "t6h", "t12h", "t17h"]  # developmental time course
LATE = ["t17h"]
TRUE_K = 2.0  # planted Z:M activation ratio at the late timepoint


def write_surrogate_matrix(path: Path) -> None:
    """Write a surrogate genes x samples TSV from the known ZGA ground truth."""

    rng = np.random.default_rng(SEED)
    n_time = len(TIMEPOINTS)

    # Module templates (relative activity over the time course).
    maternal = np.array([100.0, 70.0, 30.0, 100.0 / 3.0])  # decays; late ~33.3
    zygotic = np.array([5.0, 20.0, 50.0, 200.0 / 3.0])      # activates; late ~66.7 (Z:M=2:1)

    rows = []
    gene_ids = []
    for g in range(N_GENES):
        if g < N_GENES // 2:
            base, tag = maternal, "MAT"
        else:
            base, tag = zygotic, "ZYG"
        gene_ids.append(f"Gene_{tag}_{g:03d}")
        # Per-gene scale + multiplicative noise (counts-like, non-negative).
        scale = rng.uniform(0.5, 2.0)
        noise = rng.normal(1.0, 0.15, size=n_time)
        rows.append(np.clip(base * scale * noise, 0.0, None))

    X = np.asarray(rows)
    with path.open("w") as fh:
        fh.write("Gene\t" + "\t".join(TIMEPOINTS) + "\n")
        for gid, row in zip(gene_ids, X):
            fh.write(gid + "\t" + "\t".join(f"{v:.3f}" for v in row) + "\n")


def main() -> int:
    out_dir = Path("xenon_campaigns") / "expression"
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix_path = out_dir / "surrogate_expression_matrix.tsv"
    write_surrogate_matrix(matrix_path)

    # 1. expression matrix
    expr = load_expression_matrix(str(matrix_path))

    # 2. select gene modules
    modules = select_gene_modules(expr, n_modules=2, seed=SEED)

    # Module composition: how purely did clustering separate MAT vs ZYG genes?
    purity = []
    for j in range(modules.n_modules):
        members = [expr.genes[i] for i in range(len(expr.genes)) if modules.labels[i] == j]
        n_mat = sum("MAT" in g for g in members)
        n_zyg = sum("ZYG" in g for g in members)
        dominant = "MAT" if n_mat >= n_zyg else "ZYG"
        purity.append({
            "module": j, "size": len(members), "dominant": dominant,
            "purity": round(max(n_mat, n_zyg) / max(len(members), 1), 3),
        })

    # 3. construct state variables (late-timepoint module activities)
    states = construct_state_variables(modules, late_samples=LATE)

    # 4. XENON mechanism inference
    inference = infer_mechanism(states, protein="zga", seed=SEED)

    record = {
        "data_status": "SURROGATE (real E-MTAB-16935 counts unavailable: EBI egress "
                       "blocked, no upload). Validates pipeline recovery, not embryo biology.",
        "experiment_ref": "E-MTAB-16935 (mouse 2-cell embryo polyA+ RNA-seq)",
        "ground_truth": {"process": "maternal->zygotic (ZGA)", "planted_late_ZtoM_ratio": TRUE_K,
                         "n_genes": N_GENES, "timepoints": TIMEPOINTS},
        "modules": {"n": modules.n_modules, "sizes": modules.sizes, "composition": purity},
        "state_variables_late": {k: round(v, 3) for k, v in states.items()},
        "inference": inference,
        "verdict": {
            "modules_separated": all(p["purity"] > 0.9 for p in purity),
            "recovered_K": inference["recovered_K"],
            "recovers_planted_K": abs(inference["recovered_K"] - TRUE_K) < 1e-9,
        },
    }
    # Trim trajectory for the on-disk record.
    record["inference"] = {k: v for k, v in inference.items() if k != "posterior_trajectory"}
    record["inference"]["posterior_first"] = inference["posterior_trajectory"][0]
    record["inference"]["posterior_last"] = inference["posterior_trajectory"][-1]

    out_path = out_dir / "expression_pipeline_results.json"
    out_path.write_text(json.dumps(record, indent=2, default=str))

    print(f"Wrote {matrix_path}")
    print(f"Wrote {out_path}")
    print(f"[modules] sizes={modules.sizes} composition="
          f"{[(p['dominant'], p['purity']) for p in purity]}")
    print(f"[state vars @ late] {record['state_variables_late']}")
    print(f"[inference] observed K*={inference['observed_K']} "
          f"(active={inference['active_module']}, inactive={inference['inactive_module']})")
    print(f"[inference] recovered mechanism {inference['recovered_mechanism']} "
          f"-> K={inference['recovered_K']} (planted {TRUE_K})")
    print(f"[verdict] {record['verdict']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
