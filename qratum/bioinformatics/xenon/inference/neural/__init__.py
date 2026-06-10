"""

Neural Components for XENON

Graph neural networks and embeddings.
"""

from .embeddings import BiologicalEmbeddings
from .gnn import GraphAttentionNetwork

__all__ = ["GraphAttentionNetwork", "BiologicalEmbeddings"]
