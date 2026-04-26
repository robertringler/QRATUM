"""Test configuration and fixtures."""

import pytest


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return "This is a sample text for testing purposes"


@pytest.fixture
def sample_documents():
    """Sample documents for RAG testing."""
    return [
        {"id": "doc1", "text": "First document content"},
        {"id": "doc2", "text": "Second document content"},
        {"id": "doc3", "text": "Third document content"},
    ]
