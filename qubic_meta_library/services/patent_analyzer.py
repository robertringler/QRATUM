"""Patent analyzer service for Qubic Meta Library."""

from typing import Any

from qubic_meta_library.models import Prompt


class PatentAnalyzer:
    """Service for analyzing patent opportunities in prompt library."""

    def __init__(self):
        """Initialize patent analyzer."""

        self.high_value_threshold = 0.80
        self.premium_threshold = 0.90

    def extract_high_value_prompts(
        self, prompts: dict[int, Prompt], threshold: float | None = None
    ) -> list[Prompt]:
        """

        Extract high-value prompts based on patentability and commercial scores.

        Args:
            prompts: Dictionary of all prompts
            threshold: Custom threshold (defaults to self.high_value_threshold)

        Returns:
            List of high-value prompts sorted by combined score
        """

        if threshold is None:
            threshold = self.high_value_threshold

        high_value = [p for p in prompts.values() if p.is_high_value(threshold)]

        # Sort by combined score
        high_value.sort(
            key=lambda p: (p.patentability_score + p.commercial_potential) / 2, reverse=True
        )

        return high_value

    def identify_patent_clusters(self, prompts: dict[int, Prompt]) -> dict[str, list[Prompt]]:
        """

        Identify patent clusters by domain.

        Args:
            prompts: Dictionary of all prompts

        Returns:
            Dictionary mapping domains to high-value prompts
        """

        clusters: dict[str, list[Prompt]] = {}

        for prompt in prompts.values():
            if prompt.is_high_value(self.high_value_threshold):
                if prompt.domain not in clusters:
                    clusters[prompt.domain] = []
                clusters[prompt.domain].append(prompt)

        # Sort each cluster by patentability score
        for domain in clusters:
            clusters[domain].sort(key=lambda p: p.patentability_score, reverse=True)

        return clusters

    def generate_patent_claim_template(self, prompt: Prompt) -> dict[str, Any]:
        """

        Generate patent claim template for a prompt.

        Args:
            prompt: Prompt to generate claim for

        Returns:
            Dictionary with patent claim structure
        """

        return {
            "prompt_id": prompt.id,
            "title": f"System and Method for {prompt.category}",
            "abstract": prompt.description,
            "technical_field": prompt.domain,
            "background": {
                "problem_statement": f"Improvements needed in {prompt.category.lower()}",
                "prior_art_deficiencies": "Existing solutions lack efficiency and accuracy",
            },
            "summary": {
                "brief_description": prompt.description,
                "advantages": [
                    "Improved computational efficiency",
                    "Enhanced accuracy and precision",
                    "Scalable architecture",
                ],
            },
            "detailed_description": {
                "components": prompt.keystone_nodes,
                "execution_layers": prompt.execution_layers,
                "output_type": prompt.output_type,
            },
            "claims": {
                "independent_claims": [
                    f"A method for {prompt.category.lower()} comprising: {prompt.description}",
                ],
                "dependent_claims": [
                    f"The method of claim 1, wherein execution is performed on {', '.join(prompt.execution_layers)}",
                    f"The method of claim 1, further comprising integration with domains {', '.join(prompt.synergy_connections)}",
                ],
            },
            "metrics": {
                "patentability_score": prompt.patentability_score,
                "commercial_potential": prompt.commercial_potential,
                "synergy_count": len(prompt.synergy_connections),
            },
        }

    def analyze_cross_domain_opportunities(
        self, prompts: dict[int, Prompt]
    ) -> list[dict[str, Any]]:
        """

        Analyze cross-domain patent opportunities.

        Args:
            prompts: Dictionary of all prompts

        Returns:
            List of cross-domain patent opportunities
        """

        opportunities = []

        for prompt in prompts.values():
            if (
                prompt.is_high_value(self.high_value_threshold)
                and len(prompt.synergy_connections) >= 2
            ):
                opportunities.append(
                    {
                        "prompt_id": prompt.id,
                        "description": prompt.description,
                        "primary_domain": prompt.domain,
                        "connected_domains": prompt.synergy_connections,
                        "patentability_score": prompt.patentability_score,
                        "commercial_potential": prompt.commercial_potential,
                        "novelty_score": self._calculate_novelty_score(prompt),
                        "patent_template": self.generate_patent_claim_template(prompt),
                    }
                )

        # Sort by novelty score
        opportunities.sort(key=lambda x: x["novelty_score"], reverse=True)

        return opportunities

    def generate_patent_pipeline_report(self, prompts: dict[int, Prompt]) -> dict[str, Any]:
        """

        Generate comprehensive patent pipeline report.

        Args:
            prompts: Dictionary of all prompts

        Returns:
            Dictionary with patent pipeline metrics and analysis
        """

        high_value = self.extract_high_value_prompts(prompts)
        premium_value = [p for p in prompts.values() if p.is_high_value(self.premium_threshold)]
        cross_domain = self.analyze_cross_domain_opportunities(prompts)

        patent_clusters = self.identify_patent_clusters(prompts)

        return {
            "total_prompts": len(prompts),
            "high_value_count": len(high_value),
            "premium_value_count": len(premium_value),
            "cross_domain_opportunities": len(cross_domain),
            "patent_clusters_by_domain": {
                domain: len(cluster) for domain, cluster in patent_clusters.items()
            },
            "top_10_opportunities": [
                {
                    "prompt_id": p.id,
                    "description": p.description,
                    "patentability": p.patentability_score,
                    "commercial": p.commercial_potential,
                }
                for p in high_value[:10]
            ],
            "average_patentability": (
                sum(p.patentability_score for p in prompts.values()) / len(prompts)
                if prompts
                else 0.0
            ),
            "average_commercial_potential": (
                sum(p.commercial_potential for p in prompts.values()) / len(prompts)
                if prompts
                else 0.0
            ),
            "recommendations": self._generate_recommendations(prompts, high_value),
        }

    def _calculate_novelty_score(self, prompt: Prompt) -> float:
        """

        Calculate novelty score based on multiple factors.

        Args:
            prompt: Prompt to analyze

        Returns:
            Novelty score (0.0-1.0)
        """

        # Weighted combination of factors
        patentability_weight = 0.4
        commercial_weight = 0.3
        synergy_weight = 0.2
        keystone_weight = 0.1

        synergy_score = min(len(prompt.synergy_connections) / 5.0, 1.0)
        keystone_score = min(len(prompt.keystone_nodes) / 3.0, 1.0)

        novelty = (
            prompt.patentability_score * patentability_weight
            + prompt.commercial_potential * commercial_weight
            + synergy_score * synergy_weight
            + keystone_score * keystone_weight
        )

        return round(novelty, 3)

    def _generate_recommendations(
        self, all_prompts: dict[int, Prompt], high_value: list[Prompt]
    ) -> list[str]:
        """Generate patent strategy recommendations."""

        recommendations = []

        # High-value percentage
        hv_percentage = (len(high_value) / len(all_prompts)) * 100
        if hv_percentage < 10:
            recommendations.append(
                "Consider refining prompts to increase high-value patent opportunities"
            )
        elif hv_percentage > 20:
            recommendations.append(
                "Strong patent portfolio with high percentage of valuable prompts"
            )

        # Cross-domain opportunities
        cross_domain_count = sum(1 for p in all_prompts.values() if len(p.synergy_connections) >= 2)
        if cross_domain_count > len(all_prompts) * 0.3:
            recommendations.append(
                "Excellent cross-domain synergy opportunities for broad patent protection"
            )

        # Keystone coverage
        prompts_with_keystones = sum(1 for p in all_prompts.values() if p.keystone_nodes)
        if prompts_with_keystones < len(all_prompts) * 0.5:
            recommendations.append("Identify more keystone technologies to strengthen patents")

        return recommendations
