"""GIAB-style truth reconciliation (real, replaces the simulated validator).

Compares a query callset against a truth VCF on normalized variant keys and
reports precision / recall / F1 overall and stratified by variant type
(SNV vs indel). This is the honest core of variant-call validation
(cf. hap.py / vcfeval); it does set-based concordance on parsimoniously
normalized variants rather than the earlier placeholder validator.
"""

from __future__ import annotations

from dataclasses import dataclass

from .vcf_io import VariantKey, variant_keys


@dataclass
class ConcordanceMetrics:
    """Precision / recall / F1 from a truth comparison."""

    tp: int
    fp: int
    fn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    def as_dict(self) -> dict:
        return {
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


def _metrics(query: set[VariantKey], truth: set[VariantKey]) -> ConcordanceMetrics:
    return ConcordanceMetrics(
        tp=len(query & truth), fp=len(query - truth), fn=len(truth - query)
    )


def reconcile_keys(
    query: set[VariantKey], truth: set[VariantKey]
) -> dict[str, ConcordanceMetrics]:
    """Reconcile two normalized variant-key sets, stratified by type."""
    snv_q = {k for k in query if k.is_snv}
    snv_t = {k for k in truth if k.is_snv}
    indel_q = {k for k in query if k.is_indel}
    indel_t = {k for k in truth if k.is_indel}
    return {
        "all": _metrics(query, truth),
        "snv": _metrics(snv_q, snv_t),
        "indel": _metrics(indel_q, indel_t),
    }


def reconcile_vcf(query_path: str, truth_path: str, pass_only: bool = False) -> dict:
    """Reconcile a query VCF against a truth VCF; return metrics by type."""
    q = variant_keys(query_path, pass_only=pass_only)
    t = variant_keys(truth_path, pass_only=pass_only)
    return {stratum: m.as_dict() for stratum, m in reconcile_keys(q, t).items()}


__all__ = ["ConcordanceMetrics", "reconcile_keys", "reconcile_vcf"]
