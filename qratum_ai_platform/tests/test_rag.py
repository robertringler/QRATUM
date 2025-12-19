"""Tests for RAG connector and embedder."""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from services.rag import RAGConnector, embed_text, batch_embed, retrieve


def test_embed_text():
    """Test text embedding generation."""
    text = "This is a test query"
    embedding = embed_text(text)
    
    assert isinstance(embedding, list)
    assert len(embedding) == 48  # 384 bits / 8
    assert all(isinstance(x, float) for x in embedding)
    assert all(-1.0 <= x <= 1.0 for x in embedding)


def test_embed_text_consistency():
    """Test that same text produces same embedding."""
    text = "Consistent embedding test"
    embedding1 = embed_text(text)
    embedding2 = embed_text(text)
    
    assert embedding1 == embedding2


def test_batch_embed():
    """Test batch embedding."""
    texts = ["First text", "Second text", "Third text"]
    embeddings = batch_embed(texts)
    
    assert len(embeddings) == 3
    assert all(len(emb) == 48 for emb in embeddings)


def test_rag_connector_init():
    """Test RAG connector initialization."""
    connector = RAGConnector()
    assert connector.vector_db_url == "http://vector-db:19530"
    
    connector_custom = RAGConnector("http://custom:9999")
    assert connector_custom.vector_db_url == "http://custom:9999"


def test_rag_retrieve():
    """Test RAG retrieval."""
    connector = RAGConnector()
    results = connector.retrieve("test query", k=5)
    
    assert isinstance(results, list)
    assert len(results) == 5
    assert all("id" in r for r in results)
    assert all("text" in r for r in results)
    assert all("score" in r for r in results)
    assert all("metadata" in r for r in results)


def test_rag_retrieve_convenience():
    """Test convenience retrieve function."""
    results = retrieve("another test query", k=3)
    
    assert len(results) == 3
    assert results[0]["score"] > results[-1]["score"]  # Descending scores


def test_rag_index_documents():
    """Test document indexing."""
    connector = RAGConnector()
    
    # Valid documents
    docs = [
        {"id": "doc1", "text": "First document"},
        {"id": "doc2", "text": "Second document"}
    ]
    assert connector.index_documents(docs) is True
    
    # Invalid documents (missing fields)
    invalid_docs = [{"id": "doc3"}]  # Missing 'text'
    assert connector.index_documents(invalid_docs) is False


def test_rag_retrieve_limits():
    """Test retrieval with different k values."""
    connector = RAGConnector()
    
    # Request more than max (10)
    results = connector.retrieve("test", k=20)
    assert len(results) <= 10
    
    # Request specific amount
    results = connector.retrieve("test", k=3)
    assert len(results) == 3


@pytest.mark.slow
def test_rag_integration():
    """Integration test for RAG pipeline (requires vector DB)."""
    # This test would connect to actual vector DB in CI
    # For now, it's marked as slow and uses placeholder
    connector = RAGConnector()
    
    # Index some documents
    docs = [
        {"id": f"doc{i}", "text": f"Document {i} content"}
        for i in range(10)
    ]
    connector.index_documents(docs)
    
    # Retrieve
    results = connector.retrieve("Document 5", k=3)
    assert len(results) > 0
