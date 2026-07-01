"""Tests for the production module detector (option a)."""

from __future__ import annotations

import numpy as np

from xenon.bioinformatics.module_detection import detect_modules, silhouette_score

SAMPLES = ["t0_r1", "t0_r2", "t6_r1", "t6_r2", "t12_r1", "t12_r2", "t17_r1", "t17_r2"]
TP = {s: float(s.split("_")[0][1:]) for s in SAMPLES}
REP = {s: s.split("_")[1] for s in SAMPLES}


def _planted(n_each=25, seed=0):
    rng = np.random.default_rng(seed)
    mat = np.array([100, 100, 78, 76, 55, 53, 33, 33], float)
    zyg = np.array([5, 6, 23, 25, 44, 46, 66, 64], float)
    genes, rows = [], []
    for i in range(n_each):
        genes.append(f"MAT{i}"); rows.append(mat * rng.uniform(0.8, 1.2) * rng.normal(1, 0.05, 8))
    for i in range(n_each):
        genes.append(f"ZYG{i}"); rows.append(zyg * rng.uniform(0.8, 1.2) * rng.normal(1, 0.05, 8))
    return genes, np.clip(np.array(rows), 0, None)


def test_silhouette_score_separates_clusters():
    Z = np.array([[0, 0], [0.1, 0], [5, 5], [5.1, 5]])
    labels = np.array([0, 0, 1, 1])
    assert silhouette_score(Z, labels) > 0.8


def test_planted_recovers_two_coherent_modules():
    genes, X = _planted()
    r = detect_modules(genes, SAMPLES, X, "silhouette_kmeans",
                       timepoints=TP, replicates=REP, min_module_size=5)
    assert r.diagnostics["n_modules"] == 2
    assert r.diagnostics["mean_within_module_corr"] > 0.9
    assert r.diagnostics["quality_flags"]["exploratory_only"] is False


def test_gene_sets_mode_flags_imposed():
    genes, X = _planted()
    gs = {"maternal": [f"MAT{i}" for i in range(25)], "zygotic": [f"ZYG{i}" for i in range(25)]}
    r = detect_modules(genes, SAMPLES, X, "gene_sets", timepoints=TP, replicates=REP, gene_sets=gs)
    assert any("biologically imposed" in w for w in r.warnings)
    assert all(m.quality["marker_enrichment"] == 1.0 for m in r.modules)


def test_correlation_and_hybrid_modes_run():
    genes, X = _planted()
    gs = {"maternal": [f"MAT{i}" for i in range(25)], "zygotic": [f"ZYG{i}" for i in range(25)]}
    rc = detect_modules(genes, SAMPLES, X, "correlation_modules",
                        timepoints=TP, replicates=REP, corr_threshold=0.6, min_module_size=5)
    assert rc.diagnostics["n_modules"] >= 2
    rh = detect_modules(genes, SAMPLES, X, "hybrid",
                        timepoints=TP, replicates=REP, gene_sets=gs, min_module_size=5)
    assert {m.role for m in rh.modules} >= {"maternal", "zygotic"}


def test_weak_random_matrix_is_exploratory():
    rng = np.random.default_rng(3)
    samples = [f"s{i}" for i in range(12)]
    genes = [f"G{i}" for i in range(60)]
    X = rng.normal(50, 10, (60, 12))
    r = detect_modules(genes, samples, X, "silhouette_kmeans", min_module_size=5)
    assert r.diagnostics["mean_within_module_corr"] < 0.3
    assert r.diagnostics["quality_flags"]["weak_coherence"] is True
    assert any("exploratory only" in w for w in r.warnings)


def test_flat_dynamics_triggers_warning():
    rng = np.random.default_rng(1)
    a = np.array([60, 60, 60, 60], float)
    b = np.array([40, 40, 40, 40], float)
    genes = [f"A{i}" for i in range(15)] + [f"B{i}" for i in range(15)]
    rows = [a + rng.normal(0, 1, 4) for _ in range(15)] + [b + rng.normal(0, 1, 4) for _ in range(15)]
    samples = ["t0_r1", "t0_r2", "t17_r1", "t17_r2"]
    tp = {s: float(s.split("_")[0][1:]) for s in samples}
    r = detect_modules(genes, samples, np.array(rows), "correlation_modules",
                       timepoints=tp, corr_threshold=0.3, min_module_size=3)
    assert r.diagnostics["quality_flags"]["flat_trend"] is True
    assert any("insufficient dynamic signal" in w for w in r.warnings)


def test_unsupervised_modules_align_with_markers():
    """Discovered (unsupervised) modules should align with the known biology."""
    genes, X = _planted()
    gs = {"maternal": [f"MAT{i}" for i in range(25)], "zygotic": [f"ZYG{i}" for i in range(25)]}
    r = detect_modules(genes, SAMPLES, X, "silhouette_kmeans",
                       timepoints=TP, replicates=REP, gene_sets=gs, min_module_size=5)
    # marker enrichment computed against the provided sets should be high
    assert all(m.quality["marker_enrichment"] > 0.9 for m in r.modules)
