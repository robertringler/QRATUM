"""Tests for neural-symbolic inference engine."""

from __future__ import annotations

import numpy as np
import pytest

from xenon.bioinformatics.inference import (
    NeuralSymbolicEngine,
    GraphEmbedding,
    ConstraintType,
    InferenceResult,
)


class TestNeuralSymbolicEngine:
    """Test suite for neural-symbolic inference."""
    
    def test_initialization(self):
        """Test engine initialization."""
        engine = NeuralSymbolicEngine(seed=42)
        assert engine.input_dim == 64
        assert engine.hidden_dim == 128
        assert engine.output_dim == 32
        assert engine.backend in ["pytorch", "classical"]
    
    def test_deterministic_inference(self):
        """Test inference is deterministic."""
        # Create test data
        rng = np.random.RandomState(42)
        node_features = rng.randn(10, 64)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        
        # Run inference twice
        engine1 = NeuralSymbolicEngine(seed=42)
        result1 = engine1.infer(node_features, edge_index)
        
        engine2 = NeuralSymbolicEngine(seed=42)
        result2 = engine2.infer(node_features, edge_index)
        
        # Results should be identical
        np.testing.assert_array_almost_equal(
            result1.predictions,
            result2.predictions,
            decimal=6
        )
        np.testing.assert_array_almost_equal(
            result1.embeddings.node_embeddings,
            result2.embeddings.node_embeddings,
            decimal=6
        )
    
    def test_classical_fallback(self):
        """Test classical fallback works."""
        engine = NeuralSymbolicEngine(seed=42)
        
        rng = np.random.RandomState(42)
        node_features = rng.randn(10, 64)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        
        result = engine._infer_classical(node_features, edge_index)
        
        assert result.backend == "classical"
        assert result.predictions.shape == (10, engine.output_dim)
        assert result.embeddings.node_embeddings.shape == (10, engine.hidden_dim)
        assert result.embeddings.graph_embedding.shape == (engine.hidden_dim,)
    
    def test_graph_embedding(self):
        """Test graph embedding structure."""
        engine = NeuralSymbolicEngine(seed=42)
        
        rng = np.random.RandomState(42)
        node_features = rng.randn(10, 64)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        
        result = engine.infer(node_features, edge_index)
        
        assert isinstance(result.embeddings, GraphEmbedding)
        assert result.embeddings.node_embeddings is not None
        assert result.embeddings.graph_embedding is not None
    
    def test_different_graph_sizes(self):
        """Test engine handles different graph sizes."""
        engine = NeuralSymbolicEngine(seed=42)
        
        # Small graph
        rng = np.random.RandomState(42)
        node_features_small = rng.randn(5, 64)
        edge_index_small = np.array([[0, 1], [1, 2]])
        result_small = engine.infer(node_features_small, edge_index_small)
        
        # Large graph
        node_features_large = rng.randn(100, 64)
        edge_index_large = np.array([
            list(range(99)),
            list(range(1, 100))
        ])
        result_large = engine.infer(node_features_large, edge_index_large)
        
        assert result_small.predictions.shape[0] == 5
        assert result_large.predictions.shape[0] == 100
    
    def test_constraint_tracking(self):
        """Test constraint violation tracking."""
        engine = NeuralSymbolicEngine(seed=42)
        
        # Add a simple constraint
        def non_negative_constraint(embeddings):
            if hasattr(embeddings, 'relu'):  # PyTorch tensor
                import torch.nn.functional as F
                return F.relu(-embeddings).sum()
            else:
                return np.maximum(0, -embeddings).sum()
        
        engine.add_constraint(ConstraintType.NON_NEGATIVE, non_negative_constraint)
        
        rng = np.random.RandomState(42)
        node_features = rng.randn(10, 64)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        
        result = engine.infer(node_features, edge_index)
        
        # Constraint violations should be tracked
        assert isinstance(result.constraint_violations, dict)
    
    def test_seed_reproducibility(self):
        """Test different seeds produce different results."""
        rng = np.random.RandomState(42)
        node_features = rng.randn(10, 64)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        
        engine1 = NeuralSymbolicEngine(seed=42)
        result1 = engine1.infer(node_features, edge_index)
        
        engine2 = NeuralSymbolicEngine(seed=123)
        result2 = engine2.infer(node_features, edge_index)
        
        # Different seeds should produce different results
        # (unless both use classical fallback)
        if engine1.backend == "pytorch":
            assert not np.allclose(result1.predictions, result2.predictions)
    
    def test_output_shapes(self):
        """Test output dimensions are correct."""
        input_dim = 32
        hidden_dim = 64
        output_dim = 16
        
        engine = NeuralSymbolicEngine(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            seed=42
        )
        
        rng = np.random.RandomState(42)
        node_features = rng.randn(10, input_dim)
        edge_index = np.array([[0, 1, 2], [1, 2, 3]])
        
        result = engine.infer(node_features, edge_index)
        
        assert result.predictions.shape == (10, output_dim)
        assert result.embeddings.node_embeddings.shape == (10, hidden_dim)
