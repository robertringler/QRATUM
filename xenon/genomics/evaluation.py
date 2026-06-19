"""Leakage-controlled evaluation harness for variant-effect prediction.

This is the benchmark machinery the leadership claim plugs into: a grouped
(leakage-controlled) train/test split, calibration metrics, and head-to-head
comparison of XENON against baseline scorers on the *same* held-out variants.
It is dataset-agnostic - point it at real ClinVar/ProteinGym feature tables to
run the directive's Phase-4 benchmark; here it is validated on controlled data.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


def grouped_split(
    groups: Sequence, test_fraction: float = 0.3, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Split row indices so that no group (e.g. gene/family) spans train+test.

    Returns (train_idx, test_idx). Guarantees leakage control: the intersection
    of train-groups and test-groups is empty.
    """
    groups = np.asarray(groups)
    unique = np.array(sorted(set(groups.tolist())))
    rng = np.random.default_rng(seed)
    rng.shuffle(unique)
    n_test = max(1, int(round(len(unique) * test_fraction)))
    test_groups = set(unique[:n_test].tolist())
    test_mask = np.array([g in test_groups for g in groups])
    test_idx = np.where(test_mask)[0]
    train_idx = np.where(~test_mask)[0]
    return train_idx, test_idx


def expected_calibration_error(
    y_true: Sequence[int], y_prob: Sequence[float], n_bins: int = 10
) -> float:
    """Expected Calibration Error (binned |confidence - accuracy|)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (y_prob > lo) & (y_prob <= hi) if i > 0 else (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / n) * abs(conf - acc)
    return float(ece)


def evaluate(y_true: Sequence[int], y_prob: Sequence[float]) -> dict:
    """Discrimination + calibration metrics for one method."""
    from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    return {
        "auroc": float(roc_auc_score(y_true, y_prob)),
        "auprc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "ece": expected_calibration_error(y_true, y_prob),
        "n": int(len(y_true)),
    }


def head_to_head(y_true: Sequence[int], method_probs: dict[str, Sequence[float]]) -> dict:
    """Compare methods on the same held-out labels; rank by AUROC.

    Returns {"methods": {name: metrics}, "ranking": [names by AUROC desc],
             "best": name}.
    """
    metrics = {name: evaluate(y_true, probs) for name, probs in method_probs.items()}
    ranking = sorted(metrics, key=lambda m: metrics[m]["auroc"], reverse=True)
    return {"methods": metrics, "ranking": ranking, "best": ranking[0] if ranking else None}


__all__ = ["grouped_split", "expected_calibration_error", "evaluate", "head_to_head"]
