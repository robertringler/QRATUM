"""Synergy mapper service for Qubic Meta Library."""

from pathlib import Path
from typing import Any

import yaml

from qubic_meta_library.models import Domain, Prompt, SynergyCluster


class SynergyMapper:
    """Service for mapping synergy clusters and cross-domain connections."""

    def __init__(self, config_dir: Path | None = None):
        """

        Initialize synergy mapper.

        Args:
            config_dir: Directory containing configuration files
        """

        if config_dir is None:
            config_dir = Path(__file__).parent.parent / "config"

        self.config_dir = Path(config_dir)
        self.clusters: dict[str, SynergyCluster] = {}

    def load_clusters(self) -> dict[str, SynergyCluster]:
        """

        Load synergy cluster configurations.

        Returns:
            Dictionary mapping cluster IDs to SynergyCluster objects
        """

        clusters_file = self.config_dir / "clusters.yaml"
        if not clusters_file.exists():
            raise FileNotFoundError(f"Clusters configuration not found: {clusters_file}")

        with open(clusters_file) as f:
            data = yaml.safe_load(f)

        self.clusters = {}
        for cluster_data in data.get("clusters", []):
            # Convert revenue projection keys to integers
            if "revenue_projection" in cluster_data:
                cluster_data["revenue_projection"] = {
                    int(k): v for k, v in cluster_data["revenue_projection"].items()
                }
            cluster = SynergyCluster.from_dict(cluster_data)
            self.clusters[cluster.id] = cluster

        return self.clusters

    def get_clusters_by_domain(self, domain_id: str) -> list[SynergyCluster]:
        """

        Get all clusters containing a specific domain.

        Args:
            domain_id: Domain identifier

        Returns:
            List of clusters containing the domain
        """

        return [c for c in self.clusters.values() if domain_id in c.domains]

    def get_clusters_by_type(self, cluster_type: str) -> list[SynergyCluster]:
        """

        Get all clusters of a specific type.

        Args:
            cluster_type: Type of cluster (two-domain/multi-domain/full-stack)

        Returns:
            List of clusters of the specified type
        """

        return [c for c in self.clusters.values() if c.cluster_type == cluster_type]

    def find_synergies(
        self, prompts: dict[int, Prompt], domains: dict[str, Domain]
    ) -> dict[str, list[int]]:
        """

        Find synergy opportunities across prompts.

        Args:
            prompts: Dictionary of all prompts
            domains: Dictionary of all domains

        Returns:
            Dictionary mapping synergy cluster IDs to prompt lists
        """

        synergies: dict[str, list[int]] = {}

        for cluster in self.clusters.values():
            # Find prompts that belong to domains in this cluster
            cluster_prompts = []
            for prompt in prompts.values():
                if prompt.domain in cluster.domains:
                    cluster_prompts.append(prompt.id)

                # Check synergy connections
                for conn in prompt.synergy_connections:
                    if conn in cluster.domains and prompt.id not in cluster_prompts:
                        cluster_prompts.append(prompt.id)

            synergies[cluster.id] = sorted(cluster_prompts)

        return synergies

    def calculate_cluster_value(self, cluster_id: str) -> dict[str, Any]:
        """

        Calculate value metrics for a synergy cluster.

        Args:
            cluster_id: Cluster identifier

        Returns:
            Dictionary with value metrics
        """

        if cluster_id not in self.clusters:
            raise ValueError(f"Cluster not found: {cluster_id}")

        cluster = self.clusters[cluster_id]

        return {
            "cluster_id": cluster.id,
            "cluster_name": cluster.name,
            "domain_count": len(cluster.domains),
            "prompt_count": len(cluster.prompts),
            "total_revenue_projection": cluster.get_total_revenue_projection(),
            "cluster_type": cluster.cluster_type,
            "execution_mode": cluster.execution_mode,
            "application": cluster.application,
        }

    def get_cross_domain_opportunities(
        self, prompts: dict[int, Prompt], min_connections: int = 2
    ) -> list[dict[str, Any]]:
        """

        Identify cross-domain patent opportunities.

        Args:
            prompts: Dictionary of all prompts
            min_connections: Minimum number of domain connections

        Returns:
            List of cross-domain opportunities
        """

        opportunities = []

        for prompt in prompts.values():
            if len(prompt.synergy_connections) >= min_connections:
                opportunities.append(
                    {
                        "prompt_id": prompt.id,
                        "description": prompt.description,
                        "domain": prompt.domain,
                        "connected_domains": prompt.synergy_connections,
                        "patentability_score": prompt.patentability_score,
                        "commercial_potential": prompt.commercial_potential,
                        "connection_count": len(prompt.synergy_connections),
                    }
                )

        # Sort by patentability and connection count
        opportunities.sort(
            key=lambda x: (x["patentability_score"], x["connection_count"]), reverse=True
        )

        return opportunities

    def generate_cluster_report(self) -> dict[str, Any]:
        """

        Generate comprehensive cluster report.

        Returns:
            Dictionary with cluster statistics and metrics
        """

        total_revenue = sum(c.get_total_revenue_projection() for c in self.clusters.values())

        cluster_types = {}
        for cluster in self.clusters.values():
            cluster_types[cluster.cluster_type] = cluster_types.get(cluster.cluster_type, 0) + 1

        return {
            "total_clusters": len(self.clusters),
            "cluster_types": cluster_types,
            "total_revenue_projection": total_revenue,
            "average_revenue_per_cluster": (
                total_revenue / len(self.clusters) if self.clusters else 0
            ),
            "clusters_by_type": {ct: len(self.get_clusters_by_type(ct)) for ct in cluster_types},
        }
