"""Drug-target interaction scoring for XENON.

Provides functionality for:
- Drug-target binding affinity prediction
- ADMET property calculation
- Drug-likeness scoring
- Target druggability assessment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class DrugCandidate:
    """Drug candidate molecule.

    Attributes:
        compound_id: Compound identifier
        name: Compound name
        smiles: SMILES string
        molecular_weight: Molecular weight (Da)
        logp: Partition coefficient (lipophilicity)
        hbd: Hydrogen bond donors
        hba: Hydrogen bond acceptors
        tpsa: Topological polar surface area
        rotatable_bonds: Number of rotatable bonds
    """

    compound_id: str
    name: str
    smiles: str = ""
    molecular_weight: float = 0.0
    logp: float = 0.0
    hbd: int = 0  # Hydrogen bond donors
    hba: int = 0  # Hydrogen bond acceptors
    tpsa: float = 0.0  # Topological polar surface area
    rotatable_bonds: int = 0

    def compute_lipinski_violations(self) -> int:
        """Count Lipinski rule of five violations.

        Returns:
            Number of violations (0-4)
        """

        violations = 0

        if self.molecular_weight > 500:
            violations += 1
        if self.logp > 5:
            violations += 1
        if self.hbd > 5:
            violations += 1
        if self.hba > 10:
            violations += 1

        return violations

    def is_drug_like(self) -> bool:
        """Check if compound satisfies Lipinski's rule of five.

        Returns:
            True if drug-like (≤1 violation allowed)
        """

        return self.compute_lipinski_violations() <= 1


@dataclass
class DrugTargetInteraction:
    """Drug-target interaction.

    Attributes:
        drug_id: Drug compound ID
        target_id: Target protein ID
        binding_affinity: Binding affinity (Kd, nM)
        ic50: Half-maximal inhibitory concentration (nM)
        ki: Inhibition constant (nM)
        interaction_type: Type of interaction (inhibitor, activator, etc.)
        confidence: Confidence score (0-1)
    """

    drug_id: str
    target_id: str
    binding_affinity: Optional[float] = None
    ic50: Optional[float] = None
    ki: Optional[float] = None
    interaction_type: str = "unknown"
    confidence: float = 0.0

    def get_best_affinity(self) -> Optional[float]:
        """Get best available affinity measure.

        Returns:
            Best affinity value (lower is better)
        """

        affinities = [
            self.binding_affinity,
            self.ic50,
            self.ki,
        ]
        valid = [a for a in affinities if a is not None]
        return min(valid) if valid else None


@dataclass
class ADMETProperties:
    """ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties.

    Attributes:
        absorption: Intestinal absorption score (0-1)
        distribution: Volume of distribution
        metabolism: Metabolic stability score (0-1)
        excretion: Clearance rate
        toxicity: Toxicity score (0-1, lower is better)
        bbb_permeability: Blood-brain barrier permeability (0-1)
        cyp450_inhibition: CYP450 enzyme inhibition scores
    """

    absorption: float = 0.0
    distribution: float = 0.0
    metabolism: float = 0.0
    excretion: float = 0.0
    toxicity: float = 0.0
    bbb_permeability: float = 0.0
    cyp450_inhibition: dict[str, float] = field(default_factory=dict)


class DrugTargetScorer:
    """Drug-target interaction scoring and druggability assessment.

    Provides tools for predicting drug-target interactions, computing
    drug-likeness, and assessing target druggability.
    """

    def __init__(self):
        """Initialize drug-target scorer."""

        self._drugs: dict[str, DrugCandidate] = {}
        self._interactions: list[DrugTargetInteraction] = []
        self._admet_cache: dict[str, ADMETProperties] = {}

    def add_drug(self, drug: DrugCandidate) -> None:
        """Add a drug candidate.

        Args:
            drug: DrugCandidate object
        """

        self._drugs[drug.compound_id] = drug

    def add_interaction(self, interaction: DrugTargetInteraction) -> None:
        """Add a drug-target interaction.

        Args:
            interaction: DrugTargetInteraction object
        """

        self._interactions.append(interaction)

    def compute_binding_affinity_score(
        self,
        drug_id: str,
        target_id: str,
        use_ml_model: bool = False,
    ) -> float:
        """Compute predicted binding affinity score.

        Args:
            drug_id: Drug compound ID
            target_id: Target protein ID
            use_ml_model: Use ML model (Phase 2+)

        Returns:
            Binding affinity score (0-1, higher is better)
        """

        # Phase 1: Simplified scoring
        # Phase 2+: Implement ML-based scoring (e.g., deep learning)

        drug = self._drugs.get(drug_id)
        if not drug:
            return 0.0

        # Simple heuristic: favorable properties increase score
        score = 0.5  # Base score

        # Drug-likeness contributes
        if drug.is_drug_like():
            score += 0.2

        # Optimal lipophilicity (logP 2-3)
        if 2.0 <= drug.logp <= 3.0:
            score += 0.1

        # Moderate molecular weight (300-500 Da)
        if 300 <= drug.molecular_weight <= 500:
            score += 0.1

        # Low TPSA for cell permeability (< 140 Å²)
        if drug.tpsa < 140:
            score += 0.1

        return min(1.0, score)

    def compute_drug_likeness(self, drug_id: str) -> dict[str, any]:
        """Compute comprehensive drug-likeness metrics.

        Args:
            drug_id: Drug compound ID

        Returns:
            Dictionary of drug-likeness metrics
        """

        drug = self._drugs.get(drug_id)
        if not drug:
            return {}

        metrics = {
            "lipinski_violations": drug.compute_lipinski_violations(),
            "is_drug_like": drug.is_drug_like(),
            "molecular_weight_score": 1.0 - abs(drug.molecular_weight - 400) / 400,
            "logp_score": 1.0 - abs(drug.logp - 2.5) / 5.0,
            "tpsa_score": 1.0 if drug.tpsa < 140 else 0.5,
            "rotatable_bonds_score": 1.0 if drug.rotatable_bonds <= 10 else 0.5,
        }

        # Overall score (average of subscores)
        subscores = [
            v for k, v in metrics.items() if isinstance(v, float) and not k.startswith("is_")
        ]
        metrics["overall_score"] = float(np.mean(subscores)) if subscores else 0.0

        return metrics

    def predict_admet(self, drug_id: str) -> ADMETProperties:
        """Predict ADMET properties.

        Note: Phase 1 uses simple heuristics.
        Phase 2+ would use ML models trained on experimental data.

        Args:
            drug_id: Drug compound ID

        Returns:
            ADMET properties
        """

        if drug_id in self._admet_cache:
            return self._admet_cache[drug_id]

        drug = self._drugs.get(drug_id)
        if not drug:
            return ADMETProperties()

        # Simplified prediction based on physicochemical properties
        admet = ADMETProperties()

        # Absorption: favored by moderate lipophilicity and low TPSA
        if drug.logp > 0 and drug.tpsa < 140:
            admet.absorption = 0.8
        else:
            admet.absorption = 0.4

        # BBB permeability: requires lipophilicity and low TPSA
        if drug.logp > 1 and drug.tpsa < 90:
            admet.bbb_permeability = 0.7
        else:
            admet.bbb_permeability = 0.2

        # Metabolism: moderate lipophilicity is favorable
        if 1.0 <= drug.logp <= 3.0:
            admet.metabolism = 0.7
        else:
            admet.metabolism = 0.4

        # Toxicity: heuristic based on rule violations
        violations = drug.compute_lipinski_violations()
        admet.toxicity = violations / 4.0  # Normalize to 0-1

        self._admet_cache[drug_id] = admet
        return admet

    def assess_target_druggability(
        self,
        target_id: str,
        binding_sites: Optional[list[dict[str, any]]] = None,
    ) -> dict[str, float]:
        """Assess target druggability.

        Args:
            target_id: Target protein ID
            binding_sites: List of binding site descriptors

        Returns:
            Dictionary of druggability scores
        """

        scores = {
            "binding_site_score": 0.0,
            "ligandability_score": 0.0,
            "tractability_score": 0.0,
            "overall_score": 0.0,
        }

        if binding_sites:
            # Binding site quality (size, shape, hydrophobicity)
            site_scores = []
            for site in binding_sites:
                volume = site.get("volume", 0)
                hydrophobicity = site.get("hydrophobicity", 0)

                # Ideal binding pocket: 300-1000 Å³, moderate hydrophobicity
                size_score = 1.0 if 300 <= volume <= 1000 else 0.5
                hydro_score = 1.0 if 0.3 <= hydrophobicity <= 0.7 else 0.5

                site_scores.append((size_score + hydro_score) / 2.0)

            scores["binding_site_score"] = max(site_scores) if site_scores else 0.0

        # Ligandability: check for existing interactions
        interactions = [i for i in self._interactions if i.target_id == target_id]

        if interactions:
            # More known interactions suggest higher ligandability
            scores["ligandability_score"] = min(1.0, len(interactions) / 10.0)

        # Tractability: simplified scoring
        scores["tractability_score"] = (
            scores["binding_site_score"] * 0.6 + scores["ligandability_score"] * 0.4
        )

        scores["overall_score"] = scores["tractability_score"]

        return scores

    def rank_drug_candidates(
        self,
        target_id: str,
        criteria: list[str] = None,
    ) -> list[tuple[str, float]]:
        """Rank drug candidates for a target.

        Args:
            target_id: Target protein ID
            criteria: Ranking criteria (affinity, drug_likeness, admet)

        Returns:
            List of (drug_id, score) tuples, sorted by score
        """

        if criteria is None:
            criteria = ["affinity", "drug_likeness", "admet"]

        ranked = []

        for drug_id in self._drugs:
            score = 0.0
            weight_sum = 0.0

            if "affinity" in criteria:
                affinity_score = self.compute_binding_affinity_score(drug_id, target_id)
                score += affinity_score * 0.4
                weight_sum += 0.4

            if "drug_likeness" in criteria:
                drug_metrics = self.compute_drug_likeness(drug_id)
                dl_score = drug_metrics.get("overall_score", 0.0)
                score += dl_score * 0.3
                weight_sum += 0.3

            if "admet" in criteria:
                admet = self.predict_admet(drug_id)
                # ADMET score: high absorption, low toxicity
                admet_score = admet.absorption * 0.5 + (1.0 - admet.toxicity) * 0.5
                score += admet_score * 0.3
                weight_sum += 0.3

            if weight_sum > 0:
                score /= weight_sum

            ranked.append((drug_id, score))

        ranked.sort(key=lambda x: x[1], reverse=True)
        return ranked

    def compute_selectivity(
        self,
        drug_id: str,
        target_id: str,
        off_targets: list[str],
    ) -> float:
        """Compute drug selectivity for target vs off-targets.

        Args:
            drug_id: Drug compound ID
            target_id: Primary target ID
            off_targets: List of off-target protein IDs

        Returns:
            Selectivity score (0-1, higher is better)
        """

        target_affinity = self.compute_binding_affinity_score(drug_id, target_id)

        if not off_targets:
            return 1.0

        off_target_affinities = [
            self.compute_binding_affinity_score(drug_id, ot) for ot in off_targets
        ]

        if not off_target_affinities:
            return 1.0

        max_off_target = max(off_target_affinities)

        # Selectivity: ratio of target to off-target affinity
        if max_off_target > 0:
            selectivity = target_affinity / (target_affinity + max_off_target)
        else:
            selectivity = 1.0

        return selectivity

    def generate_drug_report(
        self,
        drug_id: str,
        target_id: str,
    ) -> dict[str, any]:
        """Generate comprehensive drug candidate report.

        Args:
            drug_id: Drug compound ID
            target_id: Target protein ID

        Returns:
            Dictionary with drug properties and predictions
        """

        drug = self._drugs.get(drug_id)
        if not drug:
            return {}

        report = {
            "drug_id": drug_id,
            "drug_name": drug.name,
            "target_id": target_id,
            "properties": {
                "molecular_weight": drug.molecular_weight,
                "logp": drug.logp,
                "hbd": drug.hbd,
                "hba": drug.hba,
                "tpsa": drug.tpsa,
            },
            "drug_likeness": self.compute_drug_likeness(drug_id),
            "binding_affinity": self.compute_binding_affinity_score(drug_id, target_id),
            "admet": self.predict_admet(drug_id).__dict__,
        }

        return report
