"""RAG (Retrieval-Augmented Generation) service package."""

from .connector import RAGConnector, retrieve
from .embedder import batch_embed, embed_text

__all__ = ["RAGConnector", "retrieve", "embed_text", "batch_embed"]
