"""Pathway analysis for metabolic and signaling networks.

Provides functionality for:
- Metabolic pathway analysis
- Signaling cascade analysis
- Flux balance analysis
- Pathway enrichment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Reaction:
    """Biochemical reaction in a pathway.

    Attributes:
        reaction_id: Reaction identifier
        name: Reaction name
        substrates: List of substrate IDs
        products: List of product IDs
        enzymes: List of enzyme IDs
        reversible: Whether reaction is reversible
        rate_constant: Forward rate constant
    """

    reaction_id: str
    name: str
    substrates: list[str]
    products: list[str]
    enzymes: list[str] = field(default_factory=list)
    reversible: bool = False
    rate_constant: float = 1.0

    def get_stoichiometry(self) -> dict[str, int]:
        """Get stoichiometry of reaction.

        Returns:
            Dictionary mapping compound IDs to stoichiometric coefficients
        """
        stoich = {}
        for substrate in self.substrates:
            stoich[substrate] = stoich.get(substrate, 0) - 1
        for product in self.products:
            stoich[product] = stoich.get(product, 0) + 1
        return stoich


@dataclass
class MetabolicPathway:
    """Metabolic pathway.

    Attributes:
        pathway_id: Pathway identifier
        name: Pathway name
        reactions: List of reactions
        metabolites: Set of metabolite IDs
        entry_points: Entry metabolites
        exit_points: Exit metabolites
    """

    pathway_id: str
    name: str
    reactions: list[Reaction] = field(default_factory=list)
    metabolites: set[str] = field(default_factory=set)
    entry_points: list[str] = field(default_factory=list)
    exit_points: list[str] = field(default_factory=list)

    def add_reaction(self, reaction: Reaction) -> None:
        """Add a reaction to the pathway.

        Args:
            reaction: Reaction object
        """
        self.reactions.append(reaction)
        self.metabolites.update(reaction.substrates)
        self.metabolites.update(reaction.products)

    def get_flux_matrix(self) -> np.ndarray:
        """Get stoichiometric matrix for flux analysis.

        Returns:
            Matrix where rows are metabolites, columns are reactions
        """
        metabolites = sorted(self.metabolites)
        met_index = {met: i for i, met in enumerate(metabolites)}

        n_mets = len(metabolites)
        n_rxns = len(self.reactions)

        matrix = np.zeros((n_mets, n_rxns))

        for j, reaction in enumerate(self.reactions):
            stoich = reaction.get_stoichiometry()
            for met, coeff in stoich.items():
                if met in met_index:
                    i = met_index[met]
                    matrix[i, j] = coeff

        return matrix


class PathwayAnalyzer:
    """Pathway analysis and enrichment.

    Provides tools for analyzing metabolic pathways, signaling cascades,
    and performing enrichment analysis.
    """

    def __init__(self):
        """Initialize pathway analyzer."""
        self._pathways: dict[str, MetabolicPathway] = {}
        self._pathway_memberships: dict[str, set[str]] = {}

    def add_pathway(self, pathway: MetabolicPathway) -> None:
        """Add a pathway to the analyzer.

        Args:
            pathway: Metabolic pathway object
        """
        self._pathways[pathway.pathway_id] = pathway

        # Index pathway memberships
        for metabolite in pathway.metabolites:
            if metabolite not in self._pathway_memberships:
                self._pathway_memberships[metabolite] = set()
            self._pathway_memberships[metabolite].add(pathway.pathway_id)

    def compute_flux_balance(
        self,
        pathway_id: str,
        objective: dict[str, float],
        constraints: Optional[dict[str, tuple[float, float]]] = None,
    ) -> dict[str, float]:
        """Compute flux balance analysis (simplified).

        Note: Full FBA requires linear programming solver (scipy.optimize.linprog).
        This is a placeholder implementation.

        Args:
            pathway_id: Pathway identifier
            objective: Objective function (reaction weights)
            constraints: Flux constraints (lower, upper bounds)

        Returns:
            Optimal flux distribution
        """
        pathway = self._pathways.get(pathway_id)
        if not pathway:
            return {}

        # Placeholder: Phase 1 simple implementation
        # Phase 2+: Implement full FBA with linear programming
        fluxes = {}
        for reaction in pathway.reactions:
            fluxes[reaction.reaction_id] = 0.0

        return fluxes

    def analyze_bottlenecks(
        self,
        pathway_id: str,
        flux_distribution: dict[str, float],
    ) -> list[tuple[str, float]]:
        """Identify pathway bottlenecks.

        Args:
            pathway_id: Pathway identifier
            flux_distribution: Flux through each reaction

        Returns:
            List of (reaction_id, bottleneck_score) tuples
        """
        pathway = self._pathways.get(pathway_id)
        if not pathway:
            return []

        bottlenecks = []
        fluxes = list(flux_distribution.values())

        if not fluxes:
            return []

        mean_flux = np.mean(fluxes)
        std_flux = np.std(fluxes)

        for reaction in pathway.reactions:
            flux = flux_distribution.get(reaction.reaction_id, 0.0)

            # Bottleneck if flux significantly below mean
            if std_flux > 0:
                z_score = (flux - mean_flux) / std_flux
                if z_score < -1.0:  # More than 1 std below mean
                    bottlenecks.append((reaction.reaction_id, abs(z_score)))

        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks

    def compute_pathway_enrichment(
        self,
        query_genes: list[str],
        background_genes: Optional[list[str]] = None,
    ) -> list[tuple[str, float, float]]:
        """Compute pathway enrichment using hypergeometric test.

        Args:
            query_genes: Genes of interest
            background_genes: Background gene set (all genes)

        Returns:
            List of (pathway_id, p_value, enrichment_ratio) tuples
        """
        from scipy.stats import hypergeom

        if background_genes is None:
            # Use all genes in all pathways as background
            background_genes = list(
                set(
                    gene
                    for pathway in self._pathways.values()
                    for reaction in pathway.reactions
                    for gene in reaction.enzymes
                )
            )

        query_set = set(query_genes)
        background_set = set(background_genes)

        N = len(background_set)  # Population size
        n = len(query_set)  # Sample size

        enrichment_results = []

        for pathway_id, pathway in self._pathways.items():
            # Get genes in this pathway
            pathway_genes = set(gene for reaction in pathway.reactions for gene in reaction.enzymes)

            K = len(pathway_genes & background_set)  # Pathway genes in background
            k = len(pathway_genes & query_set)  # Pathway genes in query

            if k == 0:
                continue

            # Hypergeometric test (survival function for over-representation)
            p_value = float(hypergeom.sf(k - 1, N, K, n))

            # Enrichment ratio
            expected = (K / N) * n
            enrichment_ratio = k / expected if expected > 0 else 0.0

            enrichment_results.append((pathway_id, p_value, enrichment_ratio))

        # Sort by p-value
        enrichment_results.sort(key=lambda x: x[1])
        return enrichment_results

    def find_signaling_cascade(
        self,
        start_protein: str,
        end_protein: str,
        max_steps: int = 10,
    ) -> list[list[str]]:
        """Find signaling cascades between proteins.

        Args:
            start_protein: Starting protein
            end_protein: Target protein
            max_steps: Maximum cascade length

        Returns:
            List of protein cascades (paths)
        """
        # Build interaction graph from reactions
        graph: dict[str, set[str]] = {}

        for pathway in self._pathways.values():
            for reaction in pathway.reactions:
                for enzyme in reaction.enzymes:
                    if enzyme not in graph:
                        graph[enzyme] = set()

                    # Enzymes that catalyze reactions producing substrates
                    # for this reaction are upstream
                    for substrate in reaction.substrates:
                        for other_reaction in pathway.reactions:
                            if substrate in other_reaction.products:
                                for other_enzyme in other_reaction.enzymes:
                                    graph[other_enzyme].add(enzyme)

        # BFS to find paths
        paths = []
        queue = [(start_protein, [start_protein])]
        visited = set()

        while queue:
            current, path = queue.pop(0)

            if len(path) > max_steps:
                continue

            if current == end_protein:
                paths.append(path)
                continue

            if current in visited:
                continue
            visited.add(current)

            for neighbor in graph.get(current, set()):
                if neighbor not in path:  # Avoid cycles
                    queue.append((neighbor, path + [neighbor]))

        return paths

    def compute_pathway_activity(
        self,
        pathway_id: str,
        expression_data: dict[str, float],
    ) -> float:
        """Compute pathway activity from gene expression.

        Args:
            pathway_id: Pathway identifier
            expression_data: Gene expression levels

        Returns:
            Pathway activity score (0-1)
        """
        pathway = self._pathways.get(pathway_id)
        if not pathway:
            return 0.0

        # Collect expression for all enzymes in pathway
        enzyme_expressions = []

        for reaction in pathway.reactions:
            for enzyme in reaction.enzymes:
                if enzyme in expression_data:
                    enzyme_expressions.append(expression_data[enzyme])

        if not enzyme_expressions:
            return 0.0

        # Activity score is mean expression
        return float(np.mean(enzyme_expressions))

    def identify_regulatory_targets(
        self,
        pathway_id: str,
        perturbation_data: dict[str, float],
        threshold: float = 0.5,
    ) -> list[tuple[str, float]]:
        """Identify key regulatory targets in pathway.

        Args:
            pathway_id: Pathway identifier
            perturbation_data: Effect of perturbing each gene
            threshold: Minimum effect threshold

        Returns:
            List of (gene_id, regulatory_effect) tuples
        """
        pathway = self._pathways.get(pathway_id)
        if not pathway:
            return []

        regulatory_targets = []

        for reaction in pathway.reactions:
            for enzyme in reaction.enzymes:
                effect = perturbation_data.get(enzyme, 0.0)

                if abs(effect) >= threshold:
                    regulatory_targets.append((enzyme, effect))

        # Sort by absolute effect
        regulatory_targets.sort(key=lambda x: abs(x[1]), reverse=True)
        return regulatory_targets

    def compute_metabolite_importance(
        self,
        pathway_id: str,
    ) -> dict[str, float]:
        """Compute importance of metabolites in pathway.

        Args:
            pathway_id: Pathway identifier

        Returns:
            Dictionary mapping metabolite IDs to importance scores
        """
        pathway = self._pathways.get(pathway_id)
        if not pathway:
            return {}

        # Count connections for each metabolite
        connections: dict[str, int] = dict.fromkeys(pathway.metabolites, 0)

        for reaction in pathway.reactions:
            for substrate in reaction.substrates:
                connections[substrate] = connections.get(substrate, 0) + 1
            for product in reaction.products:
                connections[product] = connections.get(product, 0) + 1

        # Normalize to 0-1
        max_connections = max(connections.values()) if connections else 1
        importance = {met: count / max_connections for met, count in connections.items()}

        return importance

    def export_pathway_graph(
        self,
        pathway_id: str,
    ) -> dict[str, any]:
        """Export pathway as graph data.

        Args:
            pathway_id: Pathway identifier

        Returns:
            Dictionary with nodes and edges
        """
        pathway = self._pathways.get(pathway_id)
        if not pathway:
            return {"nodes": [], "edges": []}

        nodes = []
        edges = []

        # Add metabolite nodes
        for metabolite in pathway.metabolites:
            nodes.append(
                {
                    "id": metabolite,
                    "type": "metabolite",
                }
            )

        # Add reaction edges
        for reaction in pathway.reactions:
            for substrate in reaction.substrates:
                for product in reaction.products:
                    edges.append(
                        {
                            "source": substrate,
                            "target": product,
                            "reaction": reaction.reaction_id,
                            "reversible": reaction.reversible,
                        }
                    )

        return {
            "nodes": nodes,
            "edges": edges,
            "pathway_id": pathway_id,
            "pathway_name": pathway.name,
        }
