"""Tests for the calibrated, explainable interpretation engine + harness.

Validated on a constructed dataset with a planted signal and a gene-grouping, on
a leakage-controlled split. These prove the engine recovers signal, is
calibrated, beats a weak baseline, and explains with provenance - the real
machinery the ClinVar/ProteinGym benchmark plugs into. (No leadership claim:
that requires the real datasets.)
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("sklearn")

from xenon.genomics.evaluation import (
    evaluate,
    expected_calibration_error,
    grouped_split,
    head_to_head,
)
from xenon.genomics.interpretation import VariantInterpreter


def _synthetic_dataset(n=600, n_features=8, n_informative=3, n_genes=40, seed=0):
    """A labeled variant-feature dataset with a real (planted) signal.

    Pathogenicity depends on the first `n_informative` features (logit), plus
    noise features. Each row is assigned to a 'gene' group for leakage control.
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    weights = np.zeros(n_features)
    weights[:n_informative] = rng.uniform(1.0, 2.0, size=n_informative)
    logit = X @ weights
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (rng.random(n) < prob).astype(int)
    genes = rng.integers(0, n_genes, size=n)
    names = [f"feat_{i}" for i in range(n_features)]
    return X, y, genes, names, n_informative


def test_grouped_split_is_leakage_controlled():
    _, _, genes, _, _ = _synthetic_dataset()
    train_idx, test_idx = grouped_split(genes, test_fraction=0.3, seed=1)
    train_groups = set(genes[train_idx].tolist())
    test_groups = set(genes[test_idx].tolist())
    assert train_groups.isdisjoint(test_groups)  # no gene spans both splits
    assert len(train_idx) > 0 and len(test_idx) > 0


def test_interpreter_recovers_signal_and_is_calibrated():
    X, y, genes, names, _ = _synthetic_dataset(seed=2)
    train_idx, test_idx = grouped_split(genes, test_fraction=0.3, seed=2)

    interp = VariantInterpreter(random_state=0).fit(X[train_idx], y[train_idx], names)
    probs = interp.predict_proba(X[test_idx])
    metrics = evaluate(y[test_idx], probs)

    assert metrics["auroc"] > 0.8, metrics  # recovers the planted signal
    assert metrics["ece"] < 0.2, metrics  # reasonably calibrated


def test_beats_weak_baseline_head_to_head():
    X, y, genes, names, _ = _synthetic_dataset(seed=3)
    train_idx, test_idx = grouped_split(genes, test_fraction=0.3, seed=3)

    interp = VariantInterpreter(random_state=0).fit(X[train_idx], y[train_idx], names)
    xenon_probs = interp.predict_proba(X[test_idx])
    # Weak baseline: a single noisy (non-informative) feature, squashed to [0,1].
    baseline = 1.0 / (1.0 + np.exp(-X[test_idx][:, -1]))

    h2h = head_to_head(y[test_idx], {"XENON": xenon_probs, "baseline": baseline})
    assert h2h["best"] == "XENON", h2h
    assert h2h["methods"]["XENON"]["auroc"] > h2h["methods"]["baseline"]["auroc"]


def test_explanation_has_informative_features_and_provenance():
    X, y, genes, names, n_inf = _synthetic_dataset(seed=4)
    train_idx, test_idx = grouped_split(genes, test_fraction=0.3, seed=4)
    interp = VariantInterpreter(random_state=0).fit(X[train_idx], y[train_idx], names)

    expl = interp.explain(
        X[test_idx][0], variant_id="1:100:A:T", gene="BRCA1", pathway="HDR", top_k=4
    )
    assert 0.0 <= expl.probability <= 1.0
    assert expl.gene == "BRCA1" and expl.pathway == "HDR"
    assert expl.provenance.get("feature_hash")  # provenance present
    assert len(expl.top_features) == 4
    # At least one informative feature should appear among the top contributors.
    informative = {f"feat_{i}" for i in range(n_inf)}
    assert any(c.name in informative for c in expl.top_features)


def test_ece_is_zero_for_perfect_calibration():
    # Predicting the base rate for everyone is perfectly calibrated overall.
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    probs = np.full(len(y), 0.5)
    assert expected_calibration_error(y, probs, n_bins=5) == pytest.approx(0.0)
