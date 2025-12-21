"""Conflict of laws resolver for QRATUM-OMNILEX.

This module resolves conflicts between different jurisdictions' laws,
applying choice-of-law principles deterministically.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations


class ConflictOfLawsResolver:
    """Resolves conflicts between laws of different jurisdictions.

    This resolver applies choice-of-law methodologies to determine which
    jurisdiction's law should govern a particular issue.
    """

    def __init__(self) -> None:
        """Initialize the conflict of laws resolver."""
        pass

    def resolve_conflict(
        self,
        issue_type: str,
        jurisdictions: list[str],
        connecting_factors: dict,
        forum: str
    ) -> dict:
        """Resolve conflict of laws between jurisdictions.

        Args:
            issue_type: Type of legal issue (e.g., 'contract', 'tort')
            jurisdictions: List of jurisdiction codes with potential interest
            connecting_factors: Dict of connecting factors (e.g., place of performance)
            forum: Forum jurisdiction code

        Returns:
            Dictionary with choice-of-law determination
        """
        if not jurisdictions:
            raise ValueError("At least one jurisdiction must be provided")

        # Apply appropriate choice-of-law methodology
        if issue_type == "contract":
            return self._resolve_contract_conflict(
                jurisdictions, connecting_factors, forum
            )
        elif issue_type == "tort":
            return self._resolve_tort_conflict(
                jurisdictions, connecting_factors, forum
            )
        else:
            # Default to most significant relationship
            return self._apply_most_significant_relationship(
                jurisdictions, connecting_factors, issue_type
            )

    def _resolve_contract_conflict(
        self,
        jurisdictions: list[str],
        factors: dict,
        forum: str
    ) -> dict:
        """Resolve contract choice-of-law issue.

        Args:
            jurisdictions: Candidate jurisdictions
            factors: Connecting factors
            forum: Forum jurisdiction

        Returns:
            Choice-of-law determination
        """
        # Check for express choice of law clause
        if "choice_of_law_clause" in factors:
            chosen = factors["choice_of_law_clause"]
            if chosen in jurisdictions:
                return {
                    "governing_law": chosen,
                    "methodology": "Party autonomy (express choice)",
                    "confidence": "High",
                    "reasoning": (
                        f"The parties expressly chose {chosen} law to govern "
                        f"their contract. This choice is generally respected "
                        f"absent public policy concerns."
                    ),
                    "alternative_analyses": []
                }

        # Apply Rome I for EU contracts
        if self._is_eu_contract(jurisdictions):
            return self._apply_rome_i(jurisdictions, factors)

        # Apply most significant relationship for other contracts
        return self._apply_most_significant_relationship(
            jurisdictions, factors, "contract"
        )

    def _resolve_tort_conflict(
        self,
        jurisdictions: list[str],
        factors: dict,
        forum: str
    ) -> dict:
        """Resolve tort choice-of-law issue.

        Args:
            jurisdictions: Candidate jurisdictions
            factors: Connecting factors
            forum: Forum jurisdiction

        Returns:
            Choice-of-law determination
        """
        # Apply Rome II for EU torts
        if self._is_eu_tort(jurisdictions):
            return self._apply_rome_ii(jurisdictions, factors)

        # Check for place of injury
        place_of_injury = factors.get("place_of_injury")
        if place_of_injury in jurisdictions:
            return {
                "governing_law": place_of_injury,
                "methodology": "Lex loci delicti (place of injury)",
                "confidence": "Moderate",
                "reasoning": (
                    f"Under traditional choice-of-law rules, the law of "
                    f"{place_of_injury} (place where injury occurred) governs "
                    f"tort claims."
                ),
                "alternative_analyses": [
                    "Most significant relationship test may yield different result"
                ]
            }

        # Apply most significant relationship
        return self._apply_most_significant_relationship(
            jurisdictions, factors, "tort"
        )

    def _apply_most_significant_relationship(
        self,
        jurisdictions: list[str],
        factors: dict,
        issue_type: str
    ) -> dict:
        """Apply the most significant relationship test.

        Args:
            jurisdictions: Candidate jurisdictions
            factors: Connecting factors
            issue_type: Type of legal issue

        Returns:
            Choice-of-law determination
        """
        # Score each jurisdiction based on connecting factors
        scores = dict.fromkeys(jurisdictions, 0)

        # Weight different factors
        factor_weights = {
            "place_of_contracting": 2,
            "place_of_performance": 3,
            "place_of_injury": 3,
            "domicile_of_parties": 2,
            "place_of_business": 2,
            "location_of_subject_matter": 2,
        }

        for factor, value in factors.items():
            weight = factor_weights.get(factor, 1)
            if value in jurisdictions:
                scores[value] += weight

        # Select jurisdiction with highest score
        if scores:
            governing = max(scores.items(), key=lambda x: x[1])[0]
            confidence = "Moderate" if max(scores.values()) > 2 else "Low"
        else:
            # Default to first jurisdiction
            governing = jurisdictions[0]
            confidence = "Low"

        return {
            "governing_law": governing,
            "methodology": "Most significant relationship (Restatement Second)",
            "confidence": confidence,
            "reasoning": (
                f"Based on the totality of connecting factors, {governing} "
                f"has the most significant relationship to the {issue_type}. "
                f"Relevant factors include: " +
                ", ".join(f"{k}: {v}" for k, v in factors.items() if k in factor_weights)
            ),
            "factor_scores": scores,
            "alternative_analyses": [
                "Different weight assignments may yield different results",
                "Forum law may apply to procedural matters"
            ]
        }

    def _is_eu_contract(self, jurisdictions: list[str]) -> bool:
        """Check if this is an EU contract subject to Rome I.

        Args:
            jurisdictions: Candidate jurisdictions

        Returns:
            True if Rome I applies
        """
        eu_codes = {"DE", "FR", "IT", "ES", "NL", "BE", "AT", "SE", "DK", "FI", "IE", "PT", "GR"}
        return any(j in eu_codes for j in jurisdictions)

    def _is_eu_tort(self, jurisdictions: list[str]) -> bool:
        """Check if this is an EU tort subject to Rome II.

        Args:
            jurisdictions: Candidate jurisdictions

        Returns:
            True if Rome II applies
        """
        return self._is_eu_contract(jurisdictions)

    def _apply_rome_i(self, jurisdictions: list[str], factors: dict) -> dict:
        """Apply Rome I Regulation for contract conflicts.

        Args:
            jurisdictions: Candidate jurisdictions
            factors: Connecting factors

        Returns:
            Choice-of-law determination
        """
        # Rome I hierarchy: party choice > characteristic performance > closest connection

        # Check for express choice
        if "choice_of_law_clause" in factors:
            chosen = factors["choice_of_law_clause"]
            return {
                "governing_law": chosen,
                "methodology": "Rome I Regulation - Party autonomy (Art. 3)",
                "confidence": "High",
                "reasoning": (
                    f"Under Rome I Article 3, parties may choose the governing law. "
                    f"The parties chose {chosen} law."
                ),
                "alternative_analyses": []
            }

        # Default to place of characteristic performance
        characteristic_performer = factors.get("place_of_characteristic_performance")
        if characteristic_performer in jurisdictions:
            return {
                "governing_law": characteristic_performer,
                "methodology": "Rome I Regulation - Characteristic performance (Art. 4)",
                "confidence": "Moderate",
                "reasoning": (
                    "Under Rome I Article 4, absent choice of law, the law of "
                    "the country where the party required to effect characteristic "
                    "performance has their habitual residence governs."
                ),
                "alternative_analyses": []
            }

        # Fallback to closest connection
        return {
            "governing_law": jurisdictions[0],
            "methodology": "Rome I Regulation - Closest connection (Art. 4(4))",
            "confidence": "Low",
            "reasoning": "Law of country most closely connected applies as fallback.",
            "alternative_analyses": []
        }

    def _apply_rome_ii(self, jurisdictions: list[str], factors: dict) -> dict:
        """Apply Rome II Regulation for tort conflicts.

        Args:
            jurisdictions: Candidate jurisdictions
            factors: Connecting factors

        Returns:
            Choice-of-law determination
        """
        # Rome II: place of damage > common habitual residence > manifestly closer

        place_of_damage = factors.get("place_of_injury")
        if place_of_damage in jurisdictions:
            return {
                "governing_law": place_of_damage,
                "methodology": "Rome II Regulation - Place of damage (Art. 4(1))",
                "confidence": "High",
                "reasoning": (
                    "Under Rome II Article 4(1), the law of the country where "
                    "damage occurs governs unless an exception applies."
                ),
                "alternative_analyses": [
                    "Common habitual residence exception may apply (Art. 4(2))",
                    "Manifestly closer connection exception may apply (Art. 4(3))"
                ]
            }

        return {
            "governing_law": jurisdictions[0],
            "methodology": "Rome II Regulation - Default rule",
            "confidence": "Low",
            "reasoning": "Applying default Rome II rules.",
            "alternative_analyses": []
        }
