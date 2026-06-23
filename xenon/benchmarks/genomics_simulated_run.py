"""End-to-end SIMULATED dry run of the genomics explainable-interpretation pipeline.

PURPOSE: exercise every real stage of the pipeline on synthetic data so the
plumbing is demonstrably end-to-end correct and the output has the exact shape a
real-data run produces. The synthetic dataset is generated here (clearly labeled);
the engine, harness, metrics, VCF I/O and reconciliation it drives are the REAL
modules in xenon.genomics.

THIS IS NOT A LEADERSHIP RESULT. AUROC/calibration here reflect a controlled
synthetic setup, not real ClinVar/ProteinGym. The leadership claim requires
running the same harness on the real public datasets (see
xenon/XENON_GENOMICS_LEADERSHIP_DIRECTIVE.md).

Run:  python -m xenon.benchmarks.genomics_simulated_run
"""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np

from xenon.genomics.evaluation import evaluate, grouped_split, head_to_head
from xenon.genomics.interpretation import VariantInterpreter
from xenon.genomics.truth_eval import reconcile_vcf
from xenon.genomics.vcf_io import VariantRecord, write_vcf

BASES = ["A", "C", "G", "T"]


def _simulate_cohort(n=800, n_features=10, n_informative=4, n_genes=60, seed=0):
    """Generate a synthetic variant cohort: VCF records, features, labels, genes,
    and simulated baseline scorer outputs. Clearly synthetic."""
    rng = np.random.default_rng(seed)

    # Per-variant feature matrix with a planted pathogenicity signal.
    X = rng.standard_normal((n, n_features))
    w = np.zeros(n_features)
    w[:n_informative] = rng.uniform(1.0, 2.0, size=n_informative)
    logit = X @ w
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    genes = rng.integers(0, n_genes, size=n)

    # Synthetic VCF coordinates (truth set) and a "caller" query set with
    # injected false positives / false negatives to exercise reconciliation.
    truth_records, query_records = [], []
    for i in range(n):
        chrom = str(1 + (i % 22))
        pos = 10_000 + i * 17
        ref = BASES[i % 4]
        alt = BASES[(i + 1 + (i % 3)) % 4]
        rec = VariantRecord(chrom=chrom, pos=pos, ref=ref, alt=alt, qual=50.0, filter="PASS")
        truth_records.append(rec)
        # caller misses ~3% (FN) and invents ~3% (FP)
        if rng.random() > 0.03:
            query_records.append(rec)
    n_fp = int(0.03 * n)
    for j in range(n_fp):
        query_records.append(
            VariantRecord(chrom="1", pos=900_000 + j, ref="A", alt="G", qual=30.0, filter="PASS")
        )

    # Simulated baseline scorers (weaker, uncalibrated): noisy functions of the
    # true signal, squashed to [0,1] like published raw scores.
    cadd_sim = 1.0 / (1.0 + np.exp(-(0.8 * logit + rng.normal(0, 1.5, size=n))))
    revel_sim = 1.0 / (1.0 + np.exp(-(0.9 * logit + rng.normal(0, 1.0, size=n))))

    names = [f"feat_{i}" for i in range(n_features)]
    return {
        "X": X, "y": y, "genes": genes, "names": names, "n_informative": n_informative,
        "truth_records": truth_records, "query_records": query_records,
        "baselines": {"CADD_sim": cadd_sim, "REVEL_sim": revel_sim},
    }


def run(seed: int = 0) -> dict:
    data = _simulate_cohort(seed=seed)
    report: dict = {
        "RUN_TYPE": "SIMULATED DRY RUN — synthetic data, NOT a real-data leadership result",
    }

    # --- Phase 0: write VCFs and reconcile caller vs truth (real modules) ---
    tmp = tempfile.mkdtemp()
    truth_path = os.path.join(tmp, "truth.vcf")
    query_path = os.path.join(tmp, "query.vcf")
    write_vcf(data["truth_records"], truth_path)
    write_vcf(data["query_records"], query_path)
    report["phase0_calling_concordance"] = reconcile_vcf(query_path, truth_path)

    # --- Phases 2-3: leakage-controlled split, train, predict, explain ---
    train_idx, test_idx = grouped_split(data["genes"], test_fraction=0.3, seed=seed)
    interp = VariantInterpreter(random_state=seed).fit(
        data["X"][train_idx], data["y"][train_idx], data["names"]
    )
    y_test = data["y"][test_idx]
    xenon_probs = interp.predict_proba(data["X"][test_idx])

    # --- Phase 4: head-to-head vs simulated baselines on the SAME test set ---
    methods = {"XENON": xenon_probs}
    for name, scores in data["baselines"].items():
        methods[name] = np.asarray(scores)[test_idx]
    report["leakage_control"] = {
        "train_genes": int(len(set(data["genes"][train_idx].tolist()))),
        "test_genes": int(len(set(data["genes"][test_idx].tolist()))),
        "disjoint": bool(
            set(data["genes"][train_idx].tolist()).isdisjoint(set(data["genes"][test_idx].tolist()))
        ),
    }
    report["phase4_head_to_head"] = head_to_head(y_test, methods)

    # --- Explanations with provenance for a few test variants ---
    explanations = []
    for k in range(3):
        idx = int(test_idx[k])
        rec = data["truth_records"][idx]
        expl = interp.explain(
            data["X"][idx],
            variant_id=f"{rec.chrom}:{rec.pos}:{rec.ref}:{rec.alt}",
            gene=f"GENE_{int(data['genes'][idx])}",
            pathway="SIMULATED_PATHWAY",
            top_k=4,
        )
        explanations.append(expl.as_dict())
    report["example_explanations"] = explanations

    return report


def main() -> dict:
    report = run()
    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "genomics_simulated_run.json")
    with open(out_path, "w") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(json.dumps(report, indent=2, default=float))
    print(f"\n[written] {out_path}")
    return report


if __name__ == "__main__":
    main()
