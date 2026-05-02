"""Layer B — Cluster Discovery Engine.

See :mod:`qratum_framework.observability` for the three-layer overview.
This package eliminates the need for hardcoded ontologies (e.g.
``{goblin, troll, gremlin}``) by surfacing emergent semantic clusters
from persona-tagged co-occurrence statistics.
"""

from qratum_framework.observability.clusters.community import (
    detect_communities,
    spectral_communities,
)
from qratum_framework.observability.clusters.cooccurrence import (
    DEFAULT_WINDOW,
    CoOccurrenceAccumulator,
)
from qratum_framework.observability.clusters.lift import edge_lift, lift_weighted_graph
from qratum_framework.observability.clusters.scoring import (
    Cluster,
    cluster_density,
    jaccard_stability,
    score_clusters,
)
from qratum_framework.observability.clusters.state import (
    ClusterEngineState,
    discover_clusters,
)

__all__ = [
    "Cluster",
    "ClusterEngineState",
    "CoOccurrenceAccumulator",
    "DEFAULT_WINDOW",
    "cluster_density",
    "detect_communities",
    "discover_clusters",
    "edge_lift",
    "jaccard_stability",
    "lift_weighted_graph",
    "score_clusters",
    "spectral_communities",
]
