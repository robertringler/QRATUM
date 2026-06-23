"""Calibrated, explainable variant-effect interpretation engine.

This is XENON's leadership wedge made real: given per-variant feature vectors it
produces (a) a *calibrated* pathogenicity probability with honest uncertainty and
(b) a mechanistic, provenance-tracked explanation - which black-box scorers
(CADD/REVEL/AlphaMissense/SpliceAI) do not provide.

The engine is data-agnostic: it consumes whatever feature matrix the annotation /
mechanism-mapping layers produce and is validated (see
``xenon/tests/test_genomics_interpretation.py``) to recover a known signal, be
well-calibrated, and beat a weak baseline on a leakage-controlled split. The
leadership *claim* still requires running it on real ClinVar/ProteinGym (the
benchmark harness in ``xenon.genomics.evaluation``); this module is that engine.

Requires scikit-learn (optional dependency).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


@dataclass
class FeatureContribution:
    """One feature's contribution to a single prediction."""

    name: str
    value: float
    contribution: float  # signed, importance-weighted standardized contribution


@dataclass
class MechanisticExplanation:
    """A human-auditable, provenance-tracked rationale for a prediction."""

    variant_id: str
    probability: float
    predicted_label: int
    gene: Optional[str] = None
    pathway: Optional[str] = None
    top_features: list[FeatureContribution] = field(default_factory=list)
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "variant_id": self.variant_id,
            "probability": self.probability,
            "predicted_label": self.predicted_label,
            "gene": self.gene,
            "pathway": self.pathway,
            "top_features": [
                {"name": c.name, "value": c.value, "contribution": c.contribution}
                for c in self.top_features
            ],
            "provenance": self.provenance,
        }


class VariantInterpreter:
    """Calibrated variant-effect classifier with mechanistic explanations."""

    def __init__(self, random_state: int = 0, calibration: str = "sigmoid"):
        self.random_state = random_state
        self.calibration = calibration
        self.feature_names: list[str] = []
        self._model = None
        self._mu: Optional[np.ndarray] = None
        self._sigma: Optional[np.ndarray] = None
        self._importances: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: Sequence[int], feature_names: Sequence[str]) -> "VariantInterpreter":
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.ensemble import GradientBoostingClassifier

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=int)
        self.feature_names = list(feature_names)
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0) + 1e-9

        base = GradientBoostingClassifier(random_state=self.random_state)
        # Calibrated probabilities via cross-validated wrapping of the base model.
        n_per_class = np.bincount(y).min() if len(y) else 0
        cv = max(2, min(3, int(n_per_class)))
        self._model = CalibratedClassifierCV(base, method=self.calibration, cv=cv)
        self._model.fit(X, y)

        # Global feature importances from a base model fit on all data (for
        # explanation weighting; the calibrated model handles probabilities).
        imp_model = GradientBoostingClassifier(random_state=self.random_state).fit(X, y)
        self._importances = np.asarray(imp_model.feature_importances_, dtype=float)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("VariantInterpreter must be fit before prediction")
        return self._model.predict_proba(np.asarray(X, dtype=float))[:, 1]

    def explain(
        self,
        x: Sequence[float],
        variant_id: str,
        gene: Optional[str] = None,
        pathway: Optional[str] = None,
        top_k: int = 5,
        threshold: float = 0.5,
    ) -> MechanisticExplanation:
        """Produce a calibrated, provenance-tracked explanation for one variant."""
        if self._model is None:
            raise RuntimeError("VariantInterpreter must be fit before explanation")
        x = np.asarray(x, dtype=float)
        prob = float(self.predict_proba(x.reshape(1, -1))[0])

        # Per-feature contribution: importance x standardized deviation from mean.
        z = (x - self._mu) / self._sigma
        contrib = self._importances * z
        order = np.argsort(-np.abs(contrib))[:top_k]
        top = [
            FeatureContribution(
                name=self.feature_names[i], value=float(x[i]), contribution=float(contrib[i])
            )
            for i in order
        ]

        provenance = {
            "model": "GradientBoosting+CalibratedClassifierCV",
            "calibration": self.calibration,
            "n_features": len(self.feature_names),
            "feature_hash": hashlib.sha256(
                json.dumps(self.feature_names).encode()
            ).hexdigest()[:16],
        }
        return MechanisticExplanation(
            variant_id=variant_id,
            probability=prob,
            predicted_label=int(prob >= threshold),
            gene=gene,
            pathway=pathway,
            top_features=top,
            provenance=provenance,
        )


__all__ = ["VariantInterpreter", "MechanisticExplanation", "FeatureContribution"]
