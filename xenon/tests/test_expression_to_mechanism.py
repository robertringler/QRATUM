"""Tests for the expression -> modules -> state variables -> XENON pipeline."""

from __future__ import annotations

import numpy as np

from xenon.bioinformatics.expression_to_mechanism import (
    ExpressionMatrix,
    construct_state_variables,
    infer_mechanism,
    load_expression_matrix,
    select_gene_modules,
)


def _surrogate(tmp_path, true_k: float = 2.0, n_genes: int = 120):
    rng = np.random.default_rng(0)
    times = ["t0h", "t6h", "t12h", "t17h"]
    maternal = np.array([100.0, 70.0, 30.0, 100.0 / 3.0])
    zygotic = np.array([5.0, 20.0, 50.0, 100.0 / 3.0 * true_k])
    path = tmp_path / "expr.tsv"
    with path.open("w") as fh:
        fh.write("Gene\t" + "\t".join(times) + "\n")
        for g in range(n_genes):
            base, tag = (maternal, "MAT") if g < n_genes // 2 else (zygotic, "ZYG")
            vals = np.clip(base * rng.uniform(0.5, 2.0) * rng.normal(1.0, 0.15, 4), 0, None)
            fh.write(f"Gene_{tag}_{g}\t" + "\t".join(f"{v:.3f}" for v in vals) + "\n")
    return path, times


def test_loader_reads_matrix(tmp_path):
    path, times = _surrogate(tmp_path)
    expr = load_expression_matrix(str(path))
    assert isinstance(expr, ExpressionMatrix)
    assert expr.samples == times
    assert expr.X.shape == (120, 4)


def test_modules_separate_and_xenon_recovers_planted_K(tmp_path):
    path, _ = _surrogate(tmp_path, true_k=2.0)
    expr = load_expression_matrix(str(path))

    modules = select_gene_modules(expr, n_modules=2, seed=42)
    # Two modules, each dominated by one program at high purity.
    for j in range(2):
        members = [expr.genes[i] for i in range(len(expr.genes)) if modules.labels[i] == j]
        n_mat = sum("MAT" in g for g in members)
        n_zyg = sum("ZYG" in g for g in members)
        assert max(n_mat, n_zyg) / len(members) > 0.9

    states = construct_state_variables(modules, late_samples=["t17h"])
    result = infer_mechanism(states, protein="zga", seed=42)

    # Observed late ratio ~2:1 and XENON recovers the K=2 mechanism.
    assert abs(result["observed_K"] - 2.0) < 0.5
    assert result["recovered_K"] == 2.0


def test_pipeline_is_deterministic(tmp_path):
    path, _ = _surrogate(tmp_path)
    expr = load_expression_matrix(str(path))

    def go():
        m = select_gene_modules(expr, n_modules=2, seed=7)
        s = construct_state_variables(m, late_samples=["t17h"])
        return infer_mechanism(s, protein="zga", seed=7)["final_posterior"]

    assert go() == go()


def test_infer_mechanism_honors_explicit_roles():
    """Explicit active/inactive keys preserve biological direction (Codex P2)."""
    # Maternal (inactive) still higher than zygotic (active) at the late timepoint.
    sa = {"maternal": 66.0, "zygotic": 33.0}
    # Activity-ordering fallback would mislabel the higher module as active.
    assert infer_mechanism(sa, seed=1)["active_module"] == "maternal"
    # Explicit roles keep zygotic as active -> ratio < 1 (ZGA not complete).
    r = infer_mechanism(sa, seed=1, active_key="zygotic", inactive_key="maternal")
    assert r["active_module"] == "zygotic"
    assert r["observed_K"] < 1.0
