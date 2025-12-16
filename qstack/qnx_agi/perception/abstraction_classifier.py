"""Simple abstraction classifier that bins numeric features deterministically."""

from __future__ import annotations

from typing import Dict

from .base import Percept


class AbstractionClassifier:
    def __init__(self, thresholds: Dict[str, float]):
        self.thresholds = thresholds

    def classify(self, percept: Percept) -> Dict[str, str]:
        labels: Dict[str, str] = {}
        if percept.features:
            for key, value in percept.features.items():
                threshold = self.thresholds.get(key, 0.0)
                labels[key] = "high" if value >= threshold else "low"
        return labels
