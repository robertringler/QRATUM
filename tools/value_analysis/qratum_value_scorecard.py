#!/usr/bin/env python3
"""Reproducible scorecard, weighting sensitivity, and value-concentration model.

Every number quoted in QRATUM_VALUE_ASSET_ANALYSIS.md and
QRATUM_VALUE_MODEL.md is produced here. Scores are [ESTIMATE] judgements
anchored to the [OBSERVED] / [TESTED] evidence recorded in those documents;
the weightings are explicit so a reader can disagree with the weights
without having to re-derive the scores.

    python tools/value_analysis/qratum_value_scorecard.py
    python tools/value_analysis/qratum_value_scorecard.py --csv
"""

from __future__ import annotations

import os
import sys

DIMENSIONS = [
    "technical_uniqueness",
    "implementation_maturity",
    "cross_domain_leverage",
    "commercial_value",
    "defensibility",
    "switching_cost",
    "strategic_importance",
    "capital_efficiency",
    "competitive_differentiation",
    "evidence_strength",
]

# evidence_strength = strength of evidence that the asset DOES what it claims.
# A thoroughly tested asset whose claims failed scores LOW, not high.
SCORES: dict[str, dict[str, int]] = {
    "ARBITER — authority-arbitration transition function + 270-scenario corpus": {
        "technical_uniqueness": 7, "implementation_maturity": 7,
        "cross_domain_leverage": 8, "commercial_value": 6,
        "defensibility": 3, "switching_cost": 2,
        "strategic_importance": 8, "capital_efficiency": 9,
        "competitive_differentiation": 7, "evidence_strength": 8,
    },
    "EVIDENCE-DISCIPLINE — claim registry, claims policy, traceability gate": {
        "technical_uniqueness": 8, "implementation_maturity": 6,
        "cross_domain_leverage": 9, "commercial_value": 5,
        "defensibility": 2, "switching_cost": 2,
        "strategic_importance": 7, "capital_efficiency": 8,
        "competitive_differentiation": 7, "evidence_strength": 7,
    },
    "HCAL — policy-gated, dry-run-default, audited hardware actuation": {
        "technical_uniqueness": 5, "implementation_maturity": 7,
        "cross_domain_leverage": 5, "commercial_value": 7,
        "defensibility": 3, "switching_cost": 4,
        "strategic_importance": 5, "capital_efficiency": 8,
        "competitive_differentiation": 5, "evidence_strength": 6,
    },
    "CONTRACTS — provenance / rollback-proof + 21 CFR 11 & DO-178C artifacts": {
        "technical_uniqueness": 3, "implementation_maturity": 6,
        "cross_domain_leverage": 6, "commercial_value": 5,
        "defensibility": 2, "switching_cost": 2,
        "strategic_importance": 5, "capital_efficiency": 7,
        "competitive_differentiation": 3, "evidence_strength": 5,
    },
    "KERNEL — no_std UEFI x86_64 kernel hosting the arbiter": {
        "technical_uniqueness": 4, "implementation_maturity": 5,
        "cross_domain_leverage": 3, "commercial_value": 2,
        "defensibility": 4, "switching_cost": 1,
        "strategic_importance": 5, "capital_efficiency": 2,
        "competitive_differentiation": 4, "evidence_strength": 5,
    },
    "AHTC-K — canonical-key execution-graph folding": {
        "technical_uniqueness": 5, "implementation_maturity": 6,
        "cross_domain_leverage": 6, "commercial_value": 3,
        "defensibility": 2, "switching_cost": 1,
        "strategic_importance": 4, "capital_efficiency": 5,
        "competitive_differentiation": 3, "evidence_strength": 2,
    },
    "QUASIM — simulation breadth + seed-locked reproducibility module": {
        "technical_uniqueness": 3, "implementation_maturity": 5,
        "cross_domain_leverage": 6, "commercial_value": 3,
        "defensibility": 2, "switching_cost": 1,
        "strategic_importance": 4, "capital_efficiency": 4,
        "competitive_differentiation": 2, "evidence_strength": 4,
    },
    "Q-SUBSTRATE — <500 KB deterministic embedded AI/quantum runtime": {
        "technical_uniqueness": 4, "implementation_maturity": 5,
        "cross_domain_leverage": 4, "commercial_value": 3,
        "defensibility": 2, "switching_cost": 1,
        "strategic_importance": 3, "capital_efficiency": 6,
        "competitive_differentiation": 3, "evidence_strength": 4,
    },
    "TLA+ SUITE — 14 specs with declared bounds and non-properties": {
        "technical_uniqueness": 6, "implementation_maturity": 2,
        "cross_domain_leverage": 5, "commercial_value": 4,
        "defensibility": 3, "switching_cost": 2,
        "strategic_importance": 6, "capital_efficiency": 7,
        "competitive_differentiation": 5, "evidence_strength": 2,
    },
    "QRADLE — Python deterministic contract engine": {
        "technical_uniqueness": 2, "implementation_maturity": 4,
        "cross_domain_leverage": 3, "commercial_value": 2,
        "defensibility": 1, "switching_cost": 1,
        "strategic_importance": 3, "capital_efficiency": 6,
        "competitive_differentiation": 2, "evidence_strength": 2,
    },
    "AION — polyglot SIR / lifters / proof synthesis": {
        "technical_uniqueness": 4, "implementation_maturity": 2,
        "cross_domain_leverage": 4, "commercial_value": 2,
        "defensibility": 2, "switching_cost": 1,
        "strategic_importance": 3, "capital_efficiency": 3,
        "competitive_differentiation": 3, "evidence_strength": 2,
    },
    "QRATUM-ASI — self-declared theoretical superintelligence architecture": {
        "technical_uniqueness": 3, "implementation_maturity": 1,
        "cross_domain_leverage": 5, "commercial_value": 1,
        "defensibility": 1, "switching_cost": 1,
        "strategic_importance": 3, "capital_efficiency": 2,
        "competitive_differentiation": 2, "evidence_strength": 1,
    },
    "VERTICALS — 14 domain modules": {
        "technical_uniqueness": 1, "implementation_maturity": 2,
        "cross_domain_leverage": 2, "commercial_value": 2,
        "defensibility": 1, "switching_cost": 1,
        "strategic_importance": 2, "capital_efficiency": 5,
        "competitive_differentiation": 1, "evidence_strength": 2,
    },
    "PQC — Dilithium / Kyber / SPHINCS+ modules": {
        "technical_uniqueness": 1, "implementation_maturity": 1,
        "cross_domain_leverage": 3, "commercial_value": 1,
        "defensibility": 1, "switching_cost": 1,
        "strategic_importance": 2, "capital_efficiency": 3,
        "competitive_differentiation": 1, "evidence_strength": 1,
    },
}

# ---------------------------------------------------------------------------
# Weightings. Each must sum to 1.0.
# ---------------------------------------------------------------------------
WEIGHTINGS: dict[str, dict[str, float]] = {
    # Primary: "what is actually worth something to whoever owns this repo".
    # Value only exists where evidence supports it and capital does not block
    # it, so evidence_strength and capital_efficiency are weighted up, and
    # switching_cost down (nothing is deployed, so it cannot yet accrue).
    "primary_value_realization": {
        "technical_uniqueness": 0.10, "implementation_maturity": 0.14,
        "cross_domain_leverage": 0.08, "commercial_value": 0.14,
        "defensibility": 0.11, "switching_cost": 0.04,
        "strategic_importance": 0.11, "capital_efficiency": 0.10,
        "competitive_differentiation": 0.08, "evidence_strength": 0.10,
    },
    "equal": {d: 0.1 for d in DIMENSIONS},
    # Technologist: cares about the idea and whether it is built.
    "technologist": {
        "technical_uniqueness": 0.25, "implementation_maturity": 0.20,
        "cross_domain_leverage": 0.15, "commercial_value": 0.04,
        "defensibility": 0.06, "switching_cost": 0.02,
        "strategic_importance": 0.10, "capital_efficiency": 0.02,
        "competitive_differentiation": 0.10, "evidence_strength": 0.06,
    },
    # Acquirer / IP buyer: cares about what cannot be walked around.
    "acquirer_ip": {
        "technical_uniqueness": 0.16, "implementation_maturity": 0.08,
        "cross_domain_leverage": 0.08, "commercial_value": 0.10,
        "defensibility": 0.24, "switching_cost": 0.12,
        "strategic_importance": 0.12, "capital_efficiency": 0.02,
        "competitive_differentiation": 0.06, "evidence_strength": 0.02,
    },
    # Bootstrapper: $0-$1,000, needs revenue this quarter.
    "bootstrapper": {
        "technical_uniqueness": 0.04, "implementation_maturity": 0.16,
        "cross_domain_leverage": 0.04, "commercial_value": 0.26,
        "defensibility": 0.04, "switching_cost": 0.04,
        "strategic_importance": 0.04, "capital_efficiency": 0.24,
        "competitive_differentiation": 0.04, "evidence_strength": 0.10,
    },
}


def weighted(asset: str, weighting: str) -> float:
    w = WEIGHTINGS[weighting]
    s = SCORES[asset]
    return sum(s[d] * w[d] for d in DIMENSIONS)


def ranking(weighting: str) -> list[tuple[str, float]]:
    rows = [(a, weighted(a, weighting)) for a in SCORES]
    return sorted(rows, key=lambda r: -r[1])


def concentration(weighting: str = "primary_value_realization") -> dict[str, float]:
    """Share of total modelled value held by the top 1 / 3 / 5 assets.

    Methodology: value share is proportional to the CUBE of the weighted
    score. Cubing is a deliberate convexity assumption - in early-stage
    deep-tech portfolios realisable value is strongly superlinear in asset
    quality, because a weak asset is not worth a fraction of a strong one,
    it is usually worth nothing. Linear and squared variants are printed
    alongside so the reader can see how much the conclusion depends on it.
    """
    out = {}
    for label, power in (("linear", 1), ("squared", 2), ("cubed", 3)):
        vals = [v**power for _, v in ranking(weighting)]
        total = sum(vals)
        out[f"{label}_top1"] = 100 * sum(vals[:1]) / total
        out[f"{label}_top3"] = 100 * sum(vals[:3]) / total
        out[f"{label}_top5"] = 100 * sum(vals[:5]) / total
    return out


def main() -> None:
    if "--csv" in sys.argv:
        print("asset," + ",".join(DIMENSIONS) + ",primary_weighted_score")
        for asset, _ in ranking("primary_value_realization"):
            row = [str(SCORES[asset][d]) for d in DIMENSIONS]
            print(f'"{asset}",' + ",".join(row)
                  + f",{weighted(asset, 'primary_value_realization'):.2f}")
        return

    print("=" * 78)
    print("QRATUM / QRADLE — WEIGHTED ASSET SCORECARD")
    print("=" * 78)
    for weighting in WEIGHTINGS:
        print(f"\n--- weighting: {weighting} ---")
        for rank, (asset, score) in enumerate(ranking(weighting), 1):
            marker = " <<<" if rank == 1 else ""
            print(f"  {rank:>2}. {score:5.2f}  {asset[:62]}{marker}")

    print("\n" + "=" * 78)
    print("SENSITIVITY — rank of each asset under every weighting")
    print("=" * 78)
    names = list(WEIGHTINGS)
    print(f"{'asset':<58}" + "".join(f"{n[:11]:>12}" for n in names))
    orders = {n: [a for a, _ in ranking(n)] for n in names}
    for asset in [a for a, _ in ranking("primary_value_realization")]:
        cells = "".join(f"{orders[n].index(asset) + 1:>12}" for n in names)
        print(f"{asset[:56]:<58}{cells}")

    print("\n" + "=" * 78)
    print("VALUE CONCENTRATION (analytical estimates, not measurements)")
    print("=" * 78)
    c = concentration()
    for label in ("linear", "squared", "cubed"):
        print(f"  {label:>8}: top-1 {c[f'{label}_top1']:5.1f}%   "
              f"top-3 {c[f'{label}_top3']:5.1f}%   top-5 {c[f'{label}_top5']:5.1f}%")
    print("\n  Headline figures in the report use the CUBED variant.")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Tolerate `| head`: close stdout cleanly instead of tracebacking.
        try:
            sys.stdout.close()
        finally:
            os._exit(0)
