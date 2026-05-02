"""Text embedder for RAG pipeline."""

import hashlib
from typing import List


def embed_text(text: str) -> List[float]:
    """Generate embeddings for input text.

    This is a placeholder implementation using hash-based pseudo-embeddings.
    In production, replace with actual embedding model (e.g., sentence-transformers).

    Args:
        text: Input text to embed

    Returns:
        384-dimensional embedding vector
    """
    # Placeholder: hash-based pseudo-embedding for demonstration
    # In production: use actual embedding model
    hash_obj = hashlib.sha384(text.encode("utf-8"))
    hash_bytes = hash_obj.digest()

    # Convert to float vector normalized to [-1, 1]
    embedding = [
        (byte - 128) / 128.0 for byte in hash_bytes[:48]  # 384 dimensions (48 bytes * 8 bits)
    ]

    return embedding


def batch_embed(texts: List[str]) -> List[List[float]]:
    """Generate embeddings for multiple texts.

    Args:
        texts: List of input texts

    Returns:
        List of embedding vectors
    """
    return [embed_text(text) for text in texts]
