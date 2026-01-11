"""Tests for reasoning engine with Z3 and Pyro integration.

Validates symbolic reasoning with Z3 and probabilistic reasoning with Pyro.
"""

import pytest


def test_reasoning_engine_initialization():
    """Test reasoning engine initializes with Z3 and Pyro support."""
    from qratum.platform.reasoning_engine import UnifiedReasoningEngine
    
    engine = UnifiedReasoningEngine()
    
    # Verify engine has reasoning capabilities
    stats = engine.get_stats()
    assert "reasoning_capabilities" in stats
    assert "symbolic" in stats["reasoning_capabilities"]
    assert "probabilistic" in stats["reasoning_capabilities"]
    assert "deterministic" in stats["reasoning_capabilities"]


def test_z3_symbolic_reasoning():
    """Test Z3 solver integration for symbolic reasoning."""
    try:
        import z3
    except ImportError:
        pytest.skip("z3-solver not installed")
    
    from qratum.platform.reasoning_engine import UnifiedReasoningEngine, ReasoningStrategy
    
    engine = UnifiedReasoningEngine()
    
    # Only test if Z3 is enabled
    if not engine.z3_enabled:
        pytest.skip("Z3 not available")
    
    # Test Z3 reasoning with DEDUCTIVE strategy
    result = engine._apply_z3_reasoning(
        query="Find x and y such that x > 0, y > 0, x + y < 10",
        parameters={}
    )
    
    assert result["enabled"] is True
    assert "satisfiable" in result


def test_pyro_probabilistic_reasoning():
    """Test Pyro integration for probabilistic reasoning."""
    try:
        import pyro
        import torch
    except ImportError:
        pytest.skip("pyro-ppl or torch not installed")
    
    from qratum.platform.reasoning_engine import UnifiedReasoningEngine
    
    engine = UnifiedReasoningEngine()
    
    # Only test if Pyro is enabled
    if not engine.pyro_enabled:
        pytest.skip("Pyro not available")
    
    # Test Pyro reasoning with BAYESIAN strategy
    result = engine._apply_pyro_reasoning(
        query="Estimate confidence for prediction",
        parameters={}
    )
    
    assert result["enabled"] is True
    assert "posterior_confidence" in result or "prior_mean" in result


def test_multi_vertical_synthesis_with_z3():
    """Test multi-vertical synthesis using Z3 for deductive reasoning."""
    from qratum.platform.reasoning_engine import UnifiedReasoningEngine, ReasoningStrategy
    
    engine = UnifiedReasoningEngine()
    
    # Perform synthesis with deductive strategy (uses Z3 if available)
    chain = engine.synthesize(
        query="Analyze system constraints",
        verticals=["VITRA", "JURIS"],
        strategy=ReasoningStrategy.DEDUCTIVE,
    )
    
    # Verify chain was created
    assert chain is not None
    assert chain.query == "Analyze system constraints"
    assert len(chain.nodes) == 2
    assert chain.verticals_used == ["VITRA", "JURIS"]
    
    # Verify provenance
    assert chain.verify_provenance()


def test_multi_vertical_synthesis_with_pyro():
    """Test multi-vertical synthesis using Pyro for Bayesian reasoning."""
    from qratum.platform.reasoning_engine import UnifiedReasoningEngine, ReasoningStrategy
    
    engine = UnifiedReasoningEngine()
    
    # Perform synthesis with Bayesian strategy (uses Pyro if available)
    chain = engine.synthesize(
        query="Estimate reliability",
        verticals=["QUASIM"],
        strategy=ReasoningStrategy.BAYESIAN,
    )
    
    # Verify chain was created
    assert chain is not None
    assert len(chain.nodes) == 1
    
    # Verify confidence is computed
    assert 0.0 <= chain.confidence <= 1.0


def test_reasoning_chain_export():
    """Test that reasoning chains can be exported for audit."""
    from qratum.platform.reasoning_engine import UnifiedReasoningEngine, ReasoningStrategy
    
    engine = UnifiedReasoningEngine()
    
    chain = engine.synthesize(
        query="Test query",
        verticals=["TEST"],
        strategy=ReasoningStrategy.DEDUCTIVE,
    )
    
    # Export chain
    exported = engine.export_reasoning_chain(chain.chain_id)
    
    assert exported is not None
    assert exported["chain_id"] == chain.chain_id
    assert exported["query"] == "Test query"
    assert exported["verified"] is True
    assert "provenance_hash" in exported


def test_reasoning_capabilities_stats():
    """Test that reasoning capabilities are reported in stats."""
    from qratum.platform.reasoning_engine import UnifiedReasoningEngine
    
    engine = UnifiedReasoningEngine()
    stats = engine.get_stats()
    
    # Verify reasoning capabilities are reported
    assert "reasoning_capabilities" in stats
    capabilities = stats["reasoning_capabilities"]
    
    assert "symbolic" in capabilities
    assert "probabilistic" in capabilities
    assert "deterministic" in capabilities
    assert capabilities["deterministic"] is True


def test_z3_constraint_solving():
    """Test Z3 can solve simple constraints."""
    try:
        import z3
    except ImportError:
        pytest.skip("z3-solver not installed")
    
    # Simple Z3 constraint solving test
    x = z3.Int('x')
    y = z3.Int('y')
    solver = z3.Solver()
    solver.add(x > 0)
    solver.add(y > 0)
    solver.add(x + y == 10)
    
    assert solver.check() == z3.sat
    model = solver.model()
    assert model[x].as_long() + model[y].as_long() == 10


def test_pyro_bayesian_inference():
    """Test Pyro can perform basic Bayesian inference."""
    try:
        import pyro
        import pyro.distributions as dist
        import torch
    except ImportError:
        pytest.skip("pyro-ppl or torch not installed")
    
    # Simple Pyro model test
    pyro.clear_param_store()
    
    def simple_model():
        return pyro.sample("x", dist.Normal(0.0, 1.0))
    
    # Sample from the model
    sample = simple_model()
    assert sample is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
