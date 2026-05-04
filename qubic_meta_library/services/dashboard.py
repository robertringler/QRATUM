"""Dashboard service for Qubic Meta Library KPI tracking."""

from typing import Any

from qubic_meta_library.models import Domain, Pipeline, Prompt, SynergyCluster


class Dashboard:
    """Service for generating dashboards and KPI metrics."""

    def __init__(self):
        """Initialize dashboard."""

        self.kpis: dict[str, Any] = {}

    def calculate_kpis(
        self,
        prompts: dict[int, Prompt],
        domains: dict[str, Domain],
        clusters: dict[str, SynergyCluster],
        pipelines: dict[str, Pipeline],
    ) -> dict[str, Any]:
        """

        Calculate all KPIs for the meta library.

        Args:
            prompts: Dictionary of all prompts
            domains: Dictionary of all domains
            clusters: Dictionary of all synergy clusters
            pipelines: Dictionary of all pipelines

        Returns:
            Dictionary with all KPI metrics
        """

        self.kpis = {
            "prompt_metrics": self._calculate_prompt_metrics(prompts),
            "domain_metrics": self._calculate_domain_metrics(prompts, domains),
            "cluster_metrics": self._calculate_cluster_metrics(clusters),
            "pipeline_metrics": self._calculate_pipeline_metrics(pipelines),
            "patent_metrics": self._calculate_patent_metrics(prompts),
            "commercial_metrics": self._calculate_commercial_metrics(prompts, clusters),
        }

        return self.kpis

    def _calculate_prompt_metrics(self, prompts: dict[int, Prompt]) -> dict[str, Any]:
        """Calculate prompt execution metrics."""

        if not prompts:
            return {
                "total_prompts": 0,
                "high_value_prompts": 0,
                "high_value_percentage": 0.0,
                "average_patentability": 0.0,
                "average_commercial_potential": 0.0,
                "by_phase": {},
                "by_output_type": {},
            }

        # Calculate all metrics in a single pass for efficiency
        total_prompts = 0
        high_value_count = 0
        total_patentability = 0.0
        total_commercial = 0.0

        for p in prompts.values():
            total_prompts += 1
            if p.is_high_value(0.8):
                high_value_count += 1
            total_patentability += p.patentability_score
            total_commercial += p.commercial_potential

        return {
            "total_prompts": total_prompts,
            "high_value_prompts": high_value_count,
            "high_value_percentage": (
                (high_value_count / total_prompts * 100) if total_prompts else 0.0
            ),
            "average_patentability": (
                (total_patentability / total_prompts) if total_prompts else 0.0
            ),
            "average_commercial_potential": (
                (total_commercial / total_prompts) if total_prompts else 0.0
            ),
            "by_phase": self._group_prompts_by_phase(prompts),
            "by_output_type": self._group_prompts_by_output_type(prompts),
        }

    def _calculate_domain_metrics(
        self, prompts: dict[int, Prompt], domains: dict[str, Domain]
    ) -> dict[str, Any]:
        """Calculate domain-level metrics."""

        domain_prompt_counts = {}
        domain_avg_patentability = {}

        for domain_id in domains:
            domain_prompts = [p for p in prompts.values() if p.domain == domain_id]
            domain_prompt_counts[domain_id] = len(domain_prompts)

            if domain_prompts:
                domain_avg_patentability[domain_id] = sum(
                    p.patentability_score for p in domain_prompts
                ) / len(domain_prompts)
            else:
                domain_avg_patentability[domain_id] = 0.0

        return {
            "total_domains": len(domains),
            "prompts_per_domain": domain_prompt_counts,
            "avg_patentability_per_domain": domain_avg_patentability,
            "domains_by_tier": self._group_domains_by_tier(domains),
        }

    def _calculate_cluster_metrics(self, clusters: dict[str, SynergyCluster]) -> dict[str, Any]:
        """Calculate synergy cluster metrics."""

        if not clusters:
            return {
                "total_clusters": 0,
                "cluster_types": {},
                "total_revenue_projection": 0,
                "average_revenue_per_cluster": 0,
                "clusters_with_prompts": 0,
            }

        total_revenue = sum(c.get_total_revenue_projection() for c in clusters.values())

        cluster_types = {}
        for cluster in clusters.values():
            cluster_types[cluster.cluster_type] = cluster_types.get(cluster.cluster_type, 0) + 1

        return {
            "total_clusters": len(clusters),
            "cluster_types": cluster_types,
            "total_revenue_projection": total_revenue,
            "average_revenue_per_cluster": (total_revenue / len(clusters) if clusters else 0),
            "clusters_with_prompts": sum(1 for c in clusters.values() if c.prompts),
        }

    def _calculate_pipeline_metrics(self, pipelines: dict[str, Pipeline]) -> dict[str, Any]:
        """Calculate pipeline execution metrics."""

        if not pipelines:
            return {"total": 0}

        status_counts = {}
        for pipeline in pipelines.values():
            status_counts[pipeline.status] = status_counts.get(pipeline.status, 0) + 1

        total_prompts = sum(len(p.prompts) for p in pipelines.values())
        completed_count = status_counts.get("completed", 0)

        return {
            "total_pipelines": len(pipelines),
            "status_breakdown": status_counts,
            "completion_rate": (completed_count / len(pipelines)) * 100,
            "total_prompts_in_pipelines": total_prompts,
            "average_prompts_per_pipeline": (total_prompts / len(pipelines) if pipelines else 0),
        }

    def _calculate_patent_metrics(self, prompts: dict[int, Prompt]) -> dict[str, Any]:
        """Calculate patent pipeline metrics."""

        if not prompts:
            return {"total": 0}

        high_value = [p for p in prompts.values() if p.is_high_value(0.8)]
        premium_value = [p for p in prompts.values() if p.is_high_value(0.9)]
        cross_domain = [p for p in prompts.values() if len(p.synergy_connections) >= 2]

        return {
            "total_patent_candidates": len(prompts),
            "high_value_patents": len(high_value),
            "premium_value_patents": len(premium_value),
            "cross_domain_patents": len(cross_domain),
            "patent_readiness_score": (len(high_value) / len(prompts)) * 100,
        }

    def _calculate_commercial_metrics(
        self, prompts: dict[int, Prompt], clusters: dict[str, SynergyCluster]
    ) -> dict[str, Any]:
        """Calculate commercial readiness metrics."""

        if not prompts:
            return {"average_score": 0}

        avg_commercial = sum(p.commercial_potential for p in prompts.values()) / len(prompts)

        ready_for_market = sum(1 for p in prompts.values() if p.commercial_potential >= 0.85)

        total_revenue = sum(c.get_total_revenue_projection() for c in clusters.values())

        return {
            "average_commercial_score": avg_commercial,
            "market_ready_prompts": ready_for_market,
            "market_readiness_percentage": (ready_for_market / len(prompts)) * 100,
            "total_revenue_projection": total_revenue,
            "projected_revenue_per_prompt": (total_revenue / len(prompts) if prompts else 0),
        }

    def _group_prompts_by_phase(self, prompts: dict[int, Prompt]) -> dict[int, int]:
        """Group prompts by deployment phase."""

        phase_counts = {}
        for prompt in prompts.values():
            phase_counts[prompt.phase_deployment] = phase_counts.get(prompt.phase_deployment, 0) + 1
        return phase_counts

    def _group_prompts_by_output_type(self, prompts: dict[int, Prompt]) -> dict[str, int]:
        """Group prompts by output type."""

        output_counts = {}
        for prompt in prompts.values():
            output_counts[prompt.output_type] = output_counts.get(prompt.output_type, 0) + 1
        return output_counts

    def _group_domains_by_tier(self, domains: dict[str, Domain]) -> dict[int, int]:
        """Group domains by tier."""

        tier_counts = {}
        for domain in domains.values():
            tier_counts[domain.tier] = tier_counts.get(domain.tier, 0) + 1
        return tier_counts

    def generate_executive_summary(
        self,
        prompts: dict[int, Prompt],
        domains: dict[str, Domain],
        clusters: dict[str, SynergyCluster],
        pipelines: dict[str, Pipeline],
    ) -> str:
        """

        Generate executive summary report.

        Args:
            prompts: Dictionary of all prompts
            domains: Dictionary of all domains
            clusters: Dictionary of all synergy clusters
            pipelines: Dictionary of all pipelines

        Returns:
            Executive summary as formatted string
        """

        kpis = self.calculate_kpis(prompts, domains, clusters, pipelines)

        summary = "# Qubic Meta Library Executive Summary\n\n"
        summary += "## Overview\n"
        summary += f"- Total Prompts: {kpis['prompt_metrics']['total_prompts']}\n"
        summary += f"- Active Domains: {kpis['domain_metrics']['total_domains']}\n"
        summary += f"- Synergy Clusters: {kpis['cluster_metrics']['total_clusters']}\n"
        summary += f"- Execution Pipelines: {kpis['pipeline_metrics']['total_pipelines']}\n\n"

        summary += "## Patent Pipeline\n"
        summary += f"- High-Value Patents: {kpis['patent_metrics']['high_value_patents']}\n"
        summary += f"- Premium Patents: {kpis['patent_metrics']['premium_value_patents']}\n"
        summary += (
            f"- Cross-Domain Opportunities: {kpis['patent_metrics']['cross_domain_patents']}\n"
        )
        summary += (
            f"- Patent Readiness: {kpis['patent_metrics']['patent_readiness_score']:.1f}%\n\n"
        )

        summary += "## Commercial Metrics\n"
        summary += f"- Average Commercial Score: {kpis['commercial_metrics']['average_commercial_score']:.2f}\n"
        summary += f"- Market Ready Prompts: {kpis['commercial_metrics']['market_ready_prompts']}\n"
        summary += f"- Total Revenue Projection: ${kpis['commercial_metrics']['total_revenue_projection']:,.0f}\n\n"

        summary += "## Execution Status\n"
        summary += f"- Pipeline Completion: {kpis['pipeline_metrics']['completion_rate']:.1f}%\n"
        summary += (
            f"- Prompts in Execution: {kpis['pipeline_metrics']['total_prompts_in_pipelines']}\n"
        )

        return summary
