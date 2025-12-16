"""
Neural Components for XENON

Graph neural networks and embeddings.
"""

from .gnn import GraphAttentionNetwork
from .embeddings import BiologicalEmbeddings

__all__ = ["GraphAttentionNetwork", "BiologicalEmbeddings"]
