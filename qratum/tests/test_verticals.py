"""
Tests for QRATUM Vertical Modules

Tests all 14 vertical modules for correct implementation
of the VerticalModuleBase interface.
"""

import pytest
from qratum.platform import PlatformIntent, create_contract_from_intent, MerkleEventChain
from qratum.verticals import (
    JurisModule,
    VitraModule,
    EcoraModule,
    CapraModule,
    SentraModule,
    NeuraModule,
    FluxaModule,
    ChronaModule,
    GeonaModule,
    FusiaModule,
    StrataModule,
    VexorModule,
    CohoraModule,
    OrbiaModule,
)


# All vertical modules to test
ALL_VERTICALS = [
    (JurisModule, "JURIS", "analyze_contract", {"contract_text": "Sample"}),
    (VitraModule, "VITRA", "analyze_sequence", {"sequence": "ATCG"}),
    (EcoraModule, "ECORA", "model_climate", {"scenario": "SSP2-4.5"}),
    (CapraModule, "CAPRA", "price_derivative", {"option_type": "call"}),
    (SentraModule, "SENTRA", "simulate_trajectory", {"initial_velocity_ms": 1000}),
    (NeuraModule, "NEURA", "simulate_neural_network", {"neurons": 1000}),
    (FluxaModule, "FLUXA", "optimize_routes", {"num_stops": 10}),
    (ChronaModule, "CHRONA", "simulate_circuit", {"node": "7nm"}),
    (GeonaModule, "GEONA", "analyze_satellite_imagery", {"resolution": 10}),
    (FusiaModule, "FUSIA", "simulate_plasma", {"temperature_kev": 15}),
    (StrataModule, "STRATA", "model_economy", {"country": "USA"}),
    (VexorModule, "VEXOR", "detect_threats", {"network": "corporate"}),
    (CohoraModule, "COHORA", "coordinate_swarm", {"size": 50}),
    (OrbiaModule, "ORBIA", "propagate_orbit", {"altitude_km": 400}),
]


class TestVerticalModuleBase:
    """Test that all verticals correctly implement the base interface"""
    
    @pytest.mark.parametrize("module_class,name,task,params", ALL_VERTICALS)
    def test_vertical_initialization(self, module_class, name, task, params):
        """Test vertical module initialization"""
        module = module_class()
        
        assert module.vertical_name == name
        assert module.description != ""
        assert module.safety_disclaimer != ""
        assert len(module.prohibited_uses) > 0
        assert len(module.required_compliance) > 0
    
    @pytest.mark.parametrize("module_class,name,task,params", ALL_VERTICALS)
    def test_get_supported_tasks(self, module_class, name, task, params):
        """Test that verticals return supported tasks"""
        module = module_class()
        tasks = module.get_supported_tasks()
        
        assert len(tasks) > 0
        assert isinstance(tasks, list)
        assert all(isinstance(t, str) for t in tasks)
    
    @pytest.mark.parametrize("module_class,name,task,params", ALL_VERTICALS)
    def test_execute_task(self, module_class, name, task, params):
        """Test executing a task on each vertical"""
        module = module_class()
        event_chain = MerkleEventChain()
        
        intent = PlatformIntent(
            vertical=name,
            task=task,
            parameters=params,
            requester_id="test_user",
        )
        contract = create_contract_from_intent(intent)
        
        result = module.execute_task(task, params, contract, event_chain)
        
        assert "output" in result
        assert result["vertical"] == name
        assert "safety_disclaimer" in result
        assert "prohibited_uses" in result
        assert "required_compliance" in result
        assert len(event_chain) >= 2  # Start and complete events
    
    @pytest.mark.parametrize("module_class,name,task,params", ALL_VERTICALS)
    def test_unknown_task_raises_error(self, module_class, name, task, params):
        """Test that unknown tasks raise appropriate errors"""
        module = module_class()
        event_chain = MerkleEventChain()
        
        intent = PlatformIntent(
            vertical=name,
            task="unknown_task_xyz",
            parameters={},
            requester_id="test_user",
        )
        contract = create_contract_from_intent(intent)
        
        with pytest.raises(ValueError, match="Unknown task"):
            module.execute_task("unknown_task_xyz", {}, contract, event_chain)
    
    @pytest.mark.parametrize("module_class,name,task,params", ALL_VERTICALS)
    def test_safety_validation(self, module_class, name, task, params):
        """Test safety validation method exists"""
        module = module_class()
        
        # Should not raise error for normal params
        assert module.validate_safety(params) is True
    
    @pytest.mark.parametrize("module_class,name,task,params", ALL_VERTICALS)
    def test_compliance_validation(self, module_class, name, task, params):
        """Test compliance validation returns status"""
        module = module_class()
        
        compliance = module.validate_compliance(params)
        
        assert isinstance(compliance, dict)
        assert len(compliance) > 0


class TestJurisModule:
    """Specific tests for JURIS legal AI module"""
    
    def test_contract_analysis(self):
        """Test contract analysis task"""
        juris = JurisModule()
        event_chain = MerkleEventChain()
        
        intent = PlatformIntent(
            vertical="JURIS",
            task="analyze_contract",
            parameters={"contract_text": "This contract shall indemnify the party..."},
            requester_id="test_user",
        )
        contract = create_contract_from_intent(intent)
        
        result = juris.execute_task(
            "analyze_contract",
            {"contract_text": "This contract shall indemnify the party..."},
            contract,
            event_chain,
        )
        
        assert "output" in result
        output = result["output"]
        assert "risks_identified" in output
        assert "obligations" in output
    
    def test_legal_reasoning(self):
        """Test IRAC legal reasoning"""
        juris = JurisModule()
        event_chain = MerkleEventChain()
        
        intent = PlatformIntent(
            vertical="JURIS",
            task="legal_reasoning",
            parameters={
                "issue": "Contract breach",
                "facts": "Party failed to deliver",
                "applicable_law": "UCC Article 2",
            },
            requester_id="test_user",
        )
        contract = create_contract_from_intent(intent)
        
        result = juris.execute_task(
            "legal_reasoning",
            {
                "issue": "Contract breach",
                "facts": "Party failed to deliver",
                "applicable_law": "UCC Article 2",
            },
            contract,
            event_chain,
        )
        
        output = result["output"]
        assert output["framework"] == "IRAC"
        assert "conclusion" in output


class TestVitraModule:
    """Specific tests for VITRA bioinformatics module"""
    
    def test_sequence_analysis(self):
        """Test DNA sequence analysis"""
        vitra = VitraModule()
        event_chain = MerkleEventChain()
        
        intent = PlatformIntent(
            vertical="VITRA",
            task="analyze_sequence",
            parameters={"sequence": "ATCGATCG", "type": "dna"},
            requester_id="test_user",
        )
        contract = create_contract_from_intent(intent)
        
        result = vitra.execute_task(
            "analyze_sequence",
            {"sequence": "ATCGATCG", "type": "dna"},
            contract,
            event_chain,
        )
        
        output = result["output"]
        assert "gc_content" in output
        assert "sequence_length" in output


class TestCapraModule:
    """Specific tests for CAPRA financial module"""
    
    def test_option_pricing(self):
        """Test Black-Scholes option pricing"""
        capra = CapraModule()
        event_chain = MerkleEventChain()
        
        intent = PlatformIntent(
            vertical="CAPRA",
            task="price_derivative",
            parameters={
                "option_type": "call",
                "spot_price": 100,
                "strike_price": 100,
                "time_to_maturity": 1.0,
                "risk_free_rate": 0.05,
                "volatility": 0.2,
            },
            requester_id="test_user",
        )
        contract = create_contract_from_intent(intent)
        
        result = capra.execute_task(
            "price_derivative",
            {
                "option_type": "call",
                "spot_price": 100,
                "strike_price": 100,
                "time_to_maturity": 1.0,
                "risk_free_rate": 0.05,
                "volatility": 0.2,
            },
            contract,
            event_chain,
        )
        
        output = result["output"]
        assert "price" in output
        assert "delta" in output
        assert "gamma" in output
        assert output["price"] > 0


class TestIntegrationAllVerticals:
    """Integration tests using all verticals together"""
    
    def test_all_verticals_registered(self):
        """Test that all 14 verticals can be registered"""
        from qratum.platform import PlatformOrchestrator
        
        orchestrator = PlatformOrchestrator()
        
        modules = [
            JurisModule(), VitraModule(), EcoraModule(), CapraModule(),
            SentraModule(), NeuraModule(), FluxaModule(), ChronaModule(),
            GeonaModule(), FusiaModule(), StrataModule(), VexorModule(),
            CohoraModule(), OrbiaModule(),
        ]
        
        for module in modules:
            orchestrator.register_vertical(module.vertical_name, module)
        
        status = orchestrator.get_platform_status()
        assert len(status["registered_verticals"]) == 14
    
    def test_sequential_execution_maintains_integrity(self):
        """Test that sequential executions maintain event chain integrity"""
        from qratum.platform import PlatformOrchestrator
        
        orchestrator = PlatformOrchestrator()
        orchestrator.register_vertical("JURIS", JurisModule())
        orchestrator.register_vertical("VITRA", VitraModule())
        orchestrator.register_vertical("CAPRA", CapraModule())
        
        # Execute multiple contracts
        intents = [
            PlatformIntent("JURIS", "analyze_contract", {"contract_text": "Test"}, "user1"),
            PlatformIntent("VITRA", "analyze_sequence", {"sequence": "ATCG"}, "user2"),
            PlatformIntent("CAPRA", "price_derivative", {"option_type": "call"}, "user3"),
        ]
        
        for intent in intents:
            contract = orchestrator.submit_intent(intent)
            orchestrator.execute_contract(contract)
        
        # Verify integrity
        assert orchestrator.contracts_executed == 3
        assert orchestrator.event_chain.verify_integrity() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
