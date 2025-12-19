"""RAG (Retrieval-Augmented Generation) service package."""

from .connector import RAGConnector, retrieve
from .embedder import embed_text, batch_embed

__all__ = ['RAGConnector', 'retrieve', 'embed_text', 'batch_embed']
