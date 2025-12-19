"""Explainability service package."""

from .shap_explainer import SHAPExplainer, explain_prediction

__all__ = ['SHAPExplainer', 'explain_prediction']
