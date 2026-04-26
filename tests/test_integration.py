"""Integration tests for QRATUM Platform v2.0.

Tests end-to-end workflows across multiple modules and invariant enforcement.
"""

import pytest

from qratum_platform.core import (
    PlatformIntent,
    QRATUMPlatform,
    SafetyViolation,
    VerticalModule,
)
from verticals import (
    CAPRAModule,
    ECORAModule,
    FLUXAModule,
    JURISModule,
    NEURAModule,
    SENTRAModule,
    VITRAModule,
)


class TestPlatformIntegration:
    """Test integration of all modules with platform."""

    def test_all_modules_register_successfully(self):
        """Test that all 7 modules can be registered."""
        platform = QRATUMPlatform()

        modules = [
            (VerticalModule.JURIS, JURISModule()),
            (VerticalModule.VITRA, VITRAModule()),
            (VerticalModule.ECORA, ECORAModule()),
            (VerticalModule.CAPRA, CAPRAModule()),
            (VerticalModule.SENTRA, SENTRAModule()),
            (VerticalModule.NEURA, NEURAModule()),
            (VerticalModule.FLUXA, FLUXAModule()),
        ]

        for vertical, module in modules:
            platform.register_module(vertical, module)

        assert len(platform.modules) == 7

    def test_all_modules_execute_successfully(self):
        """Test that all modules can execute operations."""
        platform = QRATUMPlatform()

        # Register all modules
        platform.register_module(VerticalModule.JURIS, JURISModule())
        platform.register_module(VerticalModule.VITRA, VITRAModule())
        platform.register_module(VerticalModule.ECORA, ECORAModule())
        platform.register_module(VerticalModule.CAPRA, CAPRAModule())
        platform.register_module(VerticalModule.SENTRA, SENTRAModule())
        platform.register_module(VerticalModule.NEURA, NEURAModule())
        platform.register_module(VerticalModule.FLUXA, FLUXAModule())

        # Test intents for each module
        intents = [
            PlatformIntent(VerticalModule.JURIS, "legal_reasoning", {"facts": "test"}, "user"),
            PlatformIntent(
                VerticalModule.VITRA,
                "sequence_analysis",
                {"sequence": "ATCG", "sequence_type": "dna"},
                "user",
            ),
            PlatformIntent(VerticalModule.ECORA, "carbon_analysis", {"activities": []}, "user"),
            PlatformIntent(VerticalModule.CAPRA, "var_calculation", {}, "user"),
            PlatformIntent(VerticalModule.SENTRA, "trajectory_simulation", {}, "user"),
            PlatformIntent(VerticalModule.NEURA, "neural_simulation", {"num_neurons": 10}, "user"),
            PlatformIntent(VerticalModule.FLUXA, "demand_forecasting", {}, "user"),
        ]

        results = []
        for intent in intents:
            contract = platform.create_contract(intent)
            result = platform.execute_contract(contract.contract_id)
            results.append(result)

        assert len(results) == 7
        for result in results:
            assert isinstance(result, dict)

    def test_event_chain_integrity_maintained(self):
        """Test that event chain maintains integrity across operations."""
        platform = QRATUMPlatform()
        platform.register_module(VerticalModule.JURIS, JURISModule())

        # Execute multiple operations
        for i in range(5):
            intent = PlatformIntent(
                VerticalModule.JURIS,
                "legal_reasoning",
                {"facts": f"test case {i}"},
                f"user_{i}",
            )
            contract = platform.create_contract(intent)
            platform.execute_contract(contract.contract_id)

        # Verify chain integrity
        assert platform.verify_integrity()
        assert len(platform.event_chain.events) >= 15  # At least 3 events per operation

    def test_contract_immutability_enforced(self):
        """Test that contracts cannot be modified after creation."""
        platform = QRATUMPlatform()
        platform.register_module(VerticalModule.JURIS, JURISModule())

        intent = PlatformIntent(
            VerticalModule.JURIS,
            "legal_reasoning",
            {"facts": "test"},
            "user",
        )

        contract = platform.create_contract(intent)

        # Attempt to modify contract should fail
        with pytest.raises(AttributeError):
            contract.estimated_cost = 999.99

    def test_safety_violations_prevented(self):
        """Test that safety violations are caught across modules."""
        platform = QRATUMPlatform()

        prohibited_tests = [
            (VerticalModule.JURIS, JURISModule(), "legal advice without license"),
            (VerticalModule.VITRA, VITRAModule(), "bioweapon development"),
            (VerticalModule.SENTRA, SENTRAModule(), "weapon targeting without authorization"),
        ]

        for vertical, module, prohibited_op in prohibited_tests:
            platform.register_module(vertical, module)

            intent = PlatformIntent(
                vertical,
                prohibited_op,
                {},
                "user",
            )

            with pytest.raises(SafetyViolation):
                platform.create_contract(intent)

    def test_deterministic_hashing(self):
        """Test that hashing is deterministic for replay."""
        platform1 = QRATUMPlatform()
        platform1.register_module(VerticalModule.JURIS, JURISModule())

        platform2 = QRATUMPlatform()
        platform2.register_module(VerticalModule.JURIS, JURISModule())

        # Create identical intents
        intent1 = PlatformIntent(
            VerticalModule.JURIS,
            "legal_reasoning",
            {"facts": "identical test"},
            "user",
            timestamp=1234567890.0,
        )

        intent2 = PlatformIntent(
            VerticalModule.JURIS,
            "legal_reasoning",
            {"facts": "identical test"},
            "user",
            timestamp=1234567890.0,
        )

        # Hashes should be identical
        assert intent1.compute_hash() == intent2.compute_hash()

    def test_module_metadata_requirements(self):
        """Test that all modules meet metadata requirements."""
        modules = [
            JURISModule(),
            VITRAModule(),
            ECORAModule(),
            CAPRAModule(),
            SENTRAModule(),
            NEURAModule(),
            FLUXAModule(),
        ]

        for module in modules:
            # Check required attributes
            assert hasattr(module, "MODULE_NAME")
            assert hasattr(module, "MODULE_VERSION")
            assert hasattr(module, "SAFETY_DISCLAIMER")
            assert hasattr(module, "PROHIBITED_USES")

            # Check types
            assert isinstance(module.MODULE_NAME, str)
            assert isinstance(module.MODULE_VERSION, str)
            assert isinstance(module.SAFETY_DISCLAIMER, str)
            assert isinstance(module.PROHIBITED_USES, list)

            # Check non-empty
            assert len(module.MODULE_NAME) > 0
            assert len(module.SAFETY_DISCLAIMER) > 0

    def test_substrate_selection(self):
        """Test that modules select appropriate substrates."""
        modules = [
            (JURISModule(), "legal_reasoning"),
            (VITRAModule(), "sequence_analysis"),
            (ECORAModule(), "climate_projection"),
            (CAPRAModule(), "option_pricing"),
            (SENTRAModule(), "trajectory_simulation"),
            (NEURAModule(), "neural_simulation"),
            (FLUXAModule(), "route_optimization"),
        ]

        for module, operation in modules:
            substrate = module.get_optimal_substrate(operation, {})
            assert substrate is not None

    def test_multi_user_isolation(self):
        """Test that operations from different users are properly isolated."""
        platform = QRATUMPlatform()
        platform.register_module(VerticalModule.JURIS, JURISModule())

        # Execute operations for different users
        user_contracts = {}
        for user_id in ["alice", "bob", "carol"]:
            intent = PlatformIntent(
                VerticalModule.JURIS,
                "legal_reasoning",
                {"facts": f"case for {user_id}"},
                user_id,
            )
            contract = platform.create_contract(intent)
            user_contracts[user_id] = contract.contract_id
            platform.execute_contract(contract.contract_id)

        # Verify all contracts exist and are distinct
        assert len(set(user_contracts.values())) == 3

        # Verify events are tracked separately
        for user_id, contract_id in user_contracts.items():
            events = platform.event_chain.get_events_for_contract(contract_id)
            assert len(events) >= 2  # At least start and complete

    def test_error_handling_and_recovery(self):
        """Test that errors are properly handled and recorded."""
        platform = QRATUMPlatform()
        platform.register_module(VerticalModule.JURIS, JURISModule())

        # Create intent with invalid operation
        intent = PlatformIntent(
            VerticalModule.JURIS,
            "nonexistent_operation",
            {},
            "user",
        )

        contract = platform.create_contract(intent)

        # Execution should fail but be recorded
        with pytest.raises(ValueError):
            platform.execute_contract(contract.contract_id)

        # Verify error was recorded in event chain
        events = platform.event_chain.get_events_for_contract(contract.contract_id)
        failed_events = [e for e in events if "failed" in e.event_type]
        assert len(failed_events) > 0


class TestCrossModuleWorkflows:
    """Test workflows that span multiple modules."""

    def test_bioinformatics_to_risk_workflow(self):
        """Test a workflow from VITRA to CAPRA (drug development risk)."""
        platform = QRATUMPlatform()
        platform.register_module(VerticalModule.VITRA, VITRAModule())
        platform.register_module(VerticalModule.CAPRA, CAPRAModule())

        # Step 1: Screen drug candidates
        vitra_intent = PlatformIntent(
            VerticalModule.VITRA,
            "drug_screening",
            {"candidates": ["DRUG_A", "DRUG_B"], "target": "TARGET_X"},
            "pharma_user",
        )
        vitra_contract = platform.create_contract(vitra_intent)
        vitra_result = platform.execute_contract(vitra_contract.contract_id)

        # Step 2: Assess financial risk of development
        capra_intent = PlatformIntent(
            VerticalModule.CAPRA,
            "monte_carlo",
            {"initial_price": 100, "drift": 0.05, "volatility": 0.3},
            "pharma_user",
        )
        capra_contract = platform.create_contract(capra_intent)
        capra_result = platform.execute_contract(capra_contract.contract_id)

        # Verify both steps completed
        assert "top_candidates" in vitra_result
        assert "mean_final_price" in capra_result

    def test_aerospace_to_supply_chain_workflow(self):
        """Test a workflow from SENTRA to FLUXA (mission logistics)."""
        platform = QRATUMPlatform()
        platform.register_module(VerticalModule.SENTRA, SENTRAModule())
        platform.register_module(VerticalModule.FLUXA, FLUXAModule())

        # Step 1: Plan orbital mission
        sentra_intent = PlatformIntent(
            VerticalModule.SENTRA,
            "mission_planning",
            {"mission_type": "satellite_deployment", "waypoints": [{"lat": 0, "lon": 0}]},
            "space_ops",
        )
        sentra_contract = platform.create_contract(sentra_intent)
        sentra_result = platform.execute_contract(sentra_contract.contract_id)

        # Step 2: Optimize ground logistics
        fluxa_intent = PlatformIntent(
            VerticalModule.FLUXA,
            "inventory_optimization",
            {"demand_rate_per_day": 50, "lead_time_days": 14},
            "space_ops",
        )
        fluxa_contract = platform.create_contract(fluxa_intent)
        fluxa_result = platform.execute_contract(fluxa_contract.contract_id)

        # Verify workflow
        assert "mission_type" in sentra_result
        assert "economic_order_quantity" in fluxa_result
