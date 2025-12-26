"""
Tests for QRATUM Planetary Infrastructure Engine.

This module tests all components of the planetary infrastructure including:
- Global Infrastructure
- Multi-Agent Simulation
- Economic Engine
- Cross-Domain Integration
- Security Layer
- Deployment Roadmap
- Continuous Optimization
- Simulation Engine
"""

import pytest
from datetime import datetime

from qratum.planetary.infrastructure import (
    GlobalInfrastructure,
    InfrastructureLayer,
    PhysicalNode,
    LogicalContract,
    AIGovernanceNode,
    SymbolicAttractor,
    NodeType,
    LayerType,
)
from qratum.planetary.agents import (
    PlanetaryAgent,
    NodeAgent,
    DataValidatorAgent,
    AIGovernanceAgent,
    EconomicAgent,
    IntegrationAgent,
    SimulationAgent,
    SecurityAgent,
    AgentNetwork,
    AgentType,
    AgentObservation,
)
from qratum.planetary.economics import (
    EconomicEngine,
    TokenFlow,
    RevenueStream,
    TokenType,
    TransactionFee,
    DataAsAService,
    AnalyticsRevenue,
    ComplianceRevenue,
    TokenAccount,
)
from qratum.planetary.integration import (
    DomainIntegration,
    SectorAdapter,
    DomainType,
    KPIMonitor,
    KPIMetric,
    InteroperabilityLayer,
)
from qratum.planetary.security import (
    SecurityLayer,
    QuantumResistantCrypto,
    AdaptiveConsensus,
    CrossChainBridge,
    DisasterRecovery,
    CryptoAlgorithm,
    ConsensusType,
)
from qratum.planetary.deployment import (
    DeploymentPhase,
    DeploymentRoadmap,
    PhaseType,
    MilestoneTracker,
    Milestone,
    MilestoneStatus,
)
from qratum.planetary.optimization import (
    ContinuousOptimizer,
    PredictiveSimulator,
    RiskAnalyzer,
    PolicyAdjuster,
    RiskLevel,
)
from qratum.planetary.simulation import (
    PlanetarySimulation,
    SimulationScenario,
    VisualizationEngine,
    MetricsCollector,
    ScenarioType,
)


class TestGlobalInfrastructure:
    """Tests for GlobalInfrastructure class."""

    def test_infrastructure_creation(self):
        """Test infrastructure can be created."""
        infra = GlobalInfrastructure()
        assert infra.infrastructure_id is not None
        assert infra.infrastructure_id.startswith("infra_")
        assert len(infra.layers) == 4

    def test_satellite_constellation_deployment(self):
        """Test satellite constellation deployment."""
        infra = GlobalInfrastructure()
        satellites = infra.deploy_satellite_constellation(num_satellites=10)
        assert len(satellites) == 10
        assert all(s.node_type == NodeType.SATELLITE for s in satellites)
        assert infra.total_capacity_tbps > 0

    def test_terrestrial_nodes_deployment(self):
        """Test terrestrial node deployment."""
        infra = GlobalInfrastructure()
        regions = [
            {"name": "us-east", "lat_min": 30, "lat_max": 45, "lon_min": -80, "lon_max": -70},
        ]
        nodes = infra.deploy_terrestrial_nodes(regions, nodes_per_region=5)
        assert len(nodes) == 5
        assert all(n.node_type == NodeType.TERRESTRIAL for n in nodes)

    def test_edge_network_deployment(self):
        """Test edge network deployment."""
        infra = GlobalInfrastructure()
        edges = infra.deploy_edge_network(num_edges=50)
        assert len(edges) == 50
        assert all(e.node_type == NodeType.EDGE for e in edges)

    def test_maritime_nodes_deployment(self):
        """Test maritime node deployment."""
        infra = GlobalInfrastructure()
        nodes = infra.deploy_maritime_nodes(num_nodes=10)
        assert len(nodes) == 10
        assert all(n.node_type == NodeType.MARITIME for n in nodes)

    def test_data_contract_creation(self):
        """Test logical contract creation."""
        infra = GlobalInfrastructure()
        contract = infra.create_data_contract(
            data_schema={"type": "sensor", "fields": ["temperature", "humidity"]},
            validators=["validator_1", "validator_2", "validator_3"],
            consensus_threshold=67,
        )
        assert contract.contract_id is not None
        assert len(contract.validators) == 3

    def test_ai_governance_deployment(self):
        """Test AI governance node deployment."""
        infra = GlobalInfrastructure()
        nodes = infra.deploy_ai_governance(num_nodes=5)
        assert len(nodes) == 5

    def test_symbolic_attractor_creation(self):
        """Test symbolic attractor creation."""
        infra = GlobalInfrastructure()
        attractor = infra.create_symbolic_attractor(
            policy_vector=[0.5, 0.3, 0.8],
            basin_depth=1.5,
        )
        assert attractor.attractor_id is not None
        assert len(attractor.policy_vector) == 3

    def test_infrastructure_statistics(self):
        """Test infrastructure statistics generation."""
        infra = GlobalInfrastructure()
        infra.deploy_satellite_constellation(num_satellites=5)
        infra.deploy_edge_network(num_edges=10)
        stats = infra.get_statistics()
        assert "infrastructure_id" in stats
        assert "total_capacity_tbps" in stats
        assert "layers" in stats


class TestMultiAgentSimulation:
    """Tests for multi-agent simulation components."""

    def test_node_agent_creation(self):
        """Test node agent creation and action."""
        agent = NodeAgent(agent_id="node_001", managed_nodes=["n1", "n2"])
        assert agent.agent_type == AgentType.NODE
        assert len(agent.managed_nodes) == 2

    def test_data_validator_agent(self):
        """Test data validator agent."""
        agent = DataValidatorAgent(
            agent_id="validator_001",
            validation_contracts=["contract_1"],
        )
        observation = AgentObservation(
            tick=1,
            view={"data_hash": "abc123", "schema_valid": True},
        )
        action = agent.act(observation)
        assert action["action_type"] == "validate"

    def test_ai_governance_agent(self):
        """Test AI governance agent."""
        agent = AIGovernanceAgent(
            agent_id="gov_001",
            governance_scope=["policy", "security"],
        )
        observation = AgentObservation(
            tick=1,
            view={"proposal": {"id": "prop_1"}, "risk_score": 0.3},
        )
        action = agent.act(observation)
        assert action["action_type"] in ["approve", "reject", "escalate"]

    def test_economic_agent(self):
        """Test economic agent."""
        agent = EconomicAgent(agent_id="econ_001", initial_balance=10000)
        assert agent.token_balance == 10000

    def test_security_agent(self):
        """Test security agent."""
        agent = SecurityAgent(agent_id="sec_001", threat_threshold=0.7)
        observation = AgentObservation(
            tick=1,
            view={"threat_level": 0.9, "anomaly_score": 0.8},
        )
        action = agent.act(observation)
        assert action["priority"] in ["critical", "high", "medium", "low"]

    def test_agent_network(self):
        """Test agent network functionality."""
        network = AgentNetwork(network_id="test_network")
        network.add_agent(NodeAgent(agent_id="node_1"))
        network.add_agent(SecurityAgent(agent_id="sec_1"))
        network.connect_agents("node_1", "sec_1")

        assert len(network.agents) == 2
        assert "sec_1" in network.connections["node_1"]

    def test_agent_network_step(self):
        """Test agent network simulation step."""
        network = AgentNetwork(network_id="test_network")
        network.add_agent(NodeAgent(agent_id="node_1"))

        actions = network.step({"node_status": "active", "load_percent": 50})
        assert "node" in actions
        assert len(actions["node"]) == 1

    def test_feedback_loop_definition(self):
        """Test feedback loop definition."""
        network = AgentNetwork(network_id="test_network")
        network.add_agent(NodeAgent(agent_id="node_1"))
        network.add_agent(EconomicAgent(agent_id="econ_1"))
        network.add_agent(SecurityAgent(agent_id="sec_1"))

        loop = network.define_feedback_loop(
            name="optimization_loop",
            agents=["node_1", "econ_1", "sec_1"],
            feedback_type="positive",
        )
        assert loop["name"] == "optimization_loop"
        assert len(network.feedback_loops) == 1


class TestEconomicEngine:
    """Tests for economic engine components."""

    def test_economic_engine_creation(self):
        """Test economic engine creation."""
        engine = EconomicEngine()
        assert engine.engine_id is not None
        assert len(engine.token_supply) == 5

    def test_account_creation(self):
        """Test account creation."""
        engine = EconomicEngine()
        account = engine.create_account("user_001")
        assert account.account_id == "user_001"
        assert all(tt in account.balances for tt in TokenType)

    def test_token_transfer(self):
        """Test token transfer between accounts."""
        engine = EconomicEngine()
        sender = engine.create_account("sender")
        receiver = engine.create_account("receiver")

        # Fund sender
        sender.deposit(TokenType.UTILITY, 1000)

        # Transfer
        flow = engine.transfer(
            "sender", "receiver", TokenType.UTILITY, 100
        )
        assert flow is not None
        assert flow.amount == 100
        assert receiver.balances[TokenType.UTILITY] == 100

    def test_staking_rewards(self):
        """Test staking reward distribution."""
        engine = EconomicEngine()
        acc1 = engine.create_account("staker_1")
        acc2 = engine.create_account("staker_2")

        acc1.deposit(TokenType.STAKE, 1000)
        acc2.deposit(TokenType.STAKE, 500)

        acc1.stake(TokenType.STAKE, 1000)
        acc2.stake(TokenType.STAKE, 500)

        rewards = engine.distribute_staking_rewards(150)
        assert len(rewards) == 2
        # staker_1 has 2/3 of stake
        assert rewards["staker_1"] == pytest.approx(100, rel=0.01)
        assert rewards["staker_2"] == pytest.approx(50, rel=0.01)

    def test_adoption_forecast(self):
        """Test adoption forecasting."""
        engine = EconomicEngine()
        forecast = engine.forecast_adoption(years=3, initial_users=1000, growth_rate=0.5)
        assert len(forecast) == 3
        assert forecast[-1]["users"] > forecast[0]["users"]

    def test_transaction_fee_calculation(self):
        """Test transaction fee calculation."""
        fee_stream = TransactionFee(
            stream_id="fee_001",
            base_rate=0.001,
        )
        fee = fee_stream.calculate_fee(1000, network_load=0.5)
        assert fee > 0


class TestCrossDomainIntegration:
    """Tests for cross-domain integration."""

    def test_domain_integration_creation(self):
        """Test domain integration creation."""
        integration = DomainIntegration()
        assert integration.integration_id is not None
        assert len(integration.interop_layer.adapters) == len(DomainType)

    def test_sector_adapter(self):
        """Test sector adapter functionality."""
        adapter = SectorAdapter(
            adapter_id="adapter_energy",
            domain=DomainType.ENERGY,
        )
        assert adapter.kpi_monitor is not None
        assert len(adapter.kpi_monitor.metrics) > 0

    def test_kpi_monitoring(self):
        """Test KPI monitoring."""
        monitor = KPIMonitor(
            monitor_id="kpi_001",
            domain=DomainType.HEALTHCARE,
        )
        metric = monitor.add_metric("Response Time", target=100, unit="ms", initial_value=150)
        assert not metric.is_on_target()

        monitor.update_metric(metric.metric_id, 90)
        assert monitor.metrics[metric.metric_id].is_on_target()

    def test_use_case_creation(self):
        """Test use case creation."""
        integration = DomainIntegration()
        use_case = integration.create_use_case(
            name="Smart Grid Optimization",
            description="Optimize energy distribution",
            domains=[DomainType.ENERGY, DomainType.LOGISTICS],
            kpis={"efficiency": 20, "cost_reduction": 15},
        )
        assert use_case["use_case_id"] is not None
        assert len(integration.use_cases) == 1

    def test_compliance_status(self):
        """Test compliance status retrieval."""
        integration = DomainIntegration()
        status = integration.get_compliance_status()
        assert DomainType.HEALTHCARE.value in status
        assert "HIPAA" in status[DomainType.HEALTHCARE.value]["requirements"]


class TestSecurityLayer:
    """Tests for security layer components."""

    def test_security_layer_creation(self):
        """Test security layer creation."""
        security = SecurityLayer()
        assert security.security_id is not None
        assert security.crypto is not None
        assert security.consensus is not None

    def test_quantum_crypto_key_generation(self):
        """Test quantum-resistant key generation."""
        crypto = QuantumResistantCrypto()
        key = crypto.generate_key_pair(algorithm=CryptoAlgorithm.CRYSTALS_DILITHIUM)
        assert key.key_id is not None
        assert key.algorithm == CryptoAlgorithm.CRYSTALS_DILITHIUM

    def test_signature_verification(self):
        """Test signature creation and verification."""
        crypto = QuantumResistantCrypto()
        key = crypto.generate_key_pair()

        sig = crypto.sign_data(key.key_id, b"test data")
        assert sig is not None

        verified = crypto.verify_signature(key.key_id, b"test data", sig["signature"])
        assert verified

    def test_adaptive_consensus(self):
        """Test adaptive consensus mechanism."""
        consensus = AdaptiveConsensus()
        consensus.add_validator("validator_1")
        consensus.add_validator("validator_2")
        consensus.add_validator("validator_3")

        proposal_id = consensus.create_proposal("test", {"action": "update"})
        consensus.submit_vote("validator_1", proposal_id, True)
        consensus.submit_vote("validator_2", proposal_id, True)

        assert consensus.proposals[proposal_id]["status"] == "approved"

    def test_protocol_adaptation(self):
        """Test consensus protocol adaptation."""
        consensus = AdaptiveConsensus()
        new_protocol = consensus.adapt_protocol(network_latency_ms=600, threat_level=0.5)
        assert new_protocol in ConsensusType

    def test_cross_chain_bridge(self):
        """Test cross-chain bridge operations."""
        bridge = CrossChainBridge(
            bridge_id="bridge_001",
            source_chain="chain_a",
            target_chain="chain_b",
        )
        tx = bridge.lock_and_mint("token_1", 100, "recipient_addr")
        assert tx["status"] == "completed"
        assert bridge.locked_assets["token_1"] == 100

    def test_disaster_recovery(self):
        """Test disaster recovery functionality."""
        recovery = DisasterRecovery(recovery_id="dr_001")
        backup = recovery.create_backup("hash_abc123", 1000)
        assert backup["status"] == "completed"

        test_result = recovery.test_recovery()
        assert "success" in test_result


class TestDeploymentRoadmap:
    """Tests for deployment roadmap."""

    def test_roadmap_creation(self):
        """Test roadmap creation."""
        roadmap = DeploymentRoadmap()
        assert roadmap.roadmap_id is not None
        assert len(roadmap.phases) == 4

    def test_phase_progression(self):
        """Test phase progression through milestones."""
        roadmap = DeploymentRoadmap()
        assert roadmap.start_phase(PhaseType.PHASE_I)

        # Complete first milestone
        roadmap.complete_milestone("p1_m1_core_infra")
        milestone = roadmap.milestone_tracker.milestones["p1_m1_core_infra"]
        assert milestone.status == MilestoneStatus.COMPLETED

    def test_milestone_dependencies(self):
        """Test milestone dependency checking."""
        roadmap = DeploymentRoadmap()
        milestone = roadmap.milestone_tracker.milestones["p1_m2_validators"]
        completed = set()

        assert not milestone.is_ready(completed)
        completed.add("p1_m1_core_infra")
        assert milestone.is_ready(completed)

    def test_overall_progress(self):
        """Test overall progress calculation."""
        roadmap = DeploymentRoadmap()
        progress = roadmap.get_overall_progress()
        assert progress["overall_progress"] == 0
        assert progress["total_milestones"] > 0

    def test_critical_path(self):
        """Test critical path calculation."""
        roadmap = DeploymentRoadmap()
        critical = roadmap.milestone_tracker.get_critical_path()
        assert len(critical) > 0


class TestContinuousOptimization:
    """Tests for continuous optimization."""

    def test_optimizer_creation(self):
        """Test optimizer creation."""
        optimizer = ContinuousOptimizer()
        assert optimizer.optimizer_id is not None
        assert optimizer.risk_analyzer is not None
        assert optimizer.simulator is not None

    def test_risk_assessment(self):
        """Test risk assessment."""
        analyzer = RiskAnalyzer()
        assessment = analyzer.assess_infrastructure_risk({
            "node_uptime_percent": 95,
            "capacity_utilization": 85,
            "error_rate_percent": 2,
        })
        assert assessment.level in RiskLevel

    def test_monte_carlo_simulation(self):
        """Test Monte Carlo simulation."""
        simulator = PredictiveSimulator()
        result = simulator.run_monte_carlo(
            scenario="growth_forecast",
            iterations=100,
            parameters={"base_value": 100, "volatility": 0.2},
        )
        assert "mean" in result["results"]
        assert "std_dev" in result["results"]

    def test_policy_adjustment(self):
        """Test policy adjustment recommendations."""
        adjuster = PolicyAdjuster()
        recommendations = adjuster.analyze_and_recommend({
            "network_load": 0.95,
            "validator_participation": 0.7,
        })
        assert len(recommendations) > 0

    def test_optimization_cycle(self):
        """Test complete optimization cycle."""
        optimizer = ContinuousOptimizer()
        result = optimizer.run_optimization_cycle({
            "node_uptime_percent": 98,
            "capacity_utilization": 70,
            "error_rate_percent": 0.5,
            "token_volatility": 0.1,
            "liquidity_ratio": 0.5,
            "revenue_growth_percent": 10,
            "threat_level": 0.3,
            "anomaly_score": 0.2,
            "active_users": 5000,
            "revenue": 50000,
            "capacity": 200,
            "network_load": 0.6,
        })
        assert "risks" in result
        assert "simulation" in result
        assert "optimization_actions" in result


class TestPlanetarySimulation:
    """Tests for planetary simulation engine."""

    def test_simulation_creation(self):
        """Test simulation engine creation."""
        sim = PlanetarySimulation()
        assert sim.simulation_id is not None
        assert len(sim.scenarios) > 0

    def test_scenario_registration(self):
        """Test custom scenario registration."""
        sim = PlanetarySimulation()
        scenario = sim.create_scenario(
            name="Custom Test",
            scenario_type=ScenarioType.STRESS_TEST,
            description="Custom stress test",
            parameters={"intensity": 0.8},
        )
        assert scenario.scenario_id in sim.scenarios

    def test_global_adoption_simulation(self):
        """Test global adoption simulation."""
        sim = PlanetarySimulation()
        run = sim.run_simulation("sc_global_adoption", seed=42)
        assert run.status.value == "completed"
        assert "final_users" in run.results

    def test_stress_test_simulation(self):
        """Test stress test simulation."""
        sim = PlanetarySimulation()
        run = sim.run_simulation("sc_stress_test", seed=42)
        assert run.status.value == "completed"
        assert "availability" in run.results

    def test_metrics_collection(self):
        """Test metrics collection during simulation."""
        collector = MetricsCollector(collector_id="test_collector")
        collector.record("test_metric", 0, 100)
        collector.record("test_metric", 1, 150)
        collector.record("test_metric", 2, 120)

        time_series = collector.get_time_series("test_metric")
        assert len(time_series) == 3

        aggs = collector.compute_aggregations()
        assert "test_metric" in aggs
        assert aggs["test_metric"]["mean"] == pytest.approx(123.33, rel=0.01)

    def test_visualization_generation(self):
        """Test visualization generation."""
        sim = PlanetarySimulation()
        run = sim.run_simulation("sc_global_adoption", seed=42)
        visualizations = sim.generate_visualizations(run.run_id)
        assert len(visualizations) > 0


class TestIntegration:
    """Integration tests for the complete planetary infrastructure."""

    def test_full_infrastructure_deployment(self):
        """Test complete infrastructure deployment."""
        infra = GlobalInfrastructure()

        # Deploy all node types
        infra.deploy_satellite_constellation(num_satellites=20)
        infra.deploy_terrestrial_nodes([{"name": "global"}], nodes_per_region=10)
        infra.deploy_edge_network(num_edges=50)
        infra.deploy_maritime_nodes(num_nodes=10)

        # Create contracts and governance
        infra.create_data_contract(
            {"type": "telemetry"}, ["v1", "v2", "v3"]
        )
        infra.deploy_ai_governance(num_nodes=5)
        infra.create_symbolic_attractor([0.5, 0.5, 0.5])

        stats = infra.get_statistics()
        assert stats["layers"]["physical"]["physical_nodes"] == 90
        assert stats["layers"]["logical"]["contracts"] == 1
        assert stats["layers"]["ai"]["governance_nodes"] == 5

    def test_agent_network_with_economic_engine(self):
        """Test agent network interaction with economic engine."""
        network = AgentNetwork(network_id="integrated_network")
        engine = EconomicEngine()

        # Add agents
        node_agent = NodeAgent(agent_id="node_1")
        econ_agent = EconomicAgent(agent_id="econ_1", initial_balance=10000)

        network.add_agent(node_agent)
        network.add_agent(econ_agent)

        # Create accounts in engine
        engine.create_account("node_1")
        engine.create_account("econ_1")
        engine.accounts["econ_1"].deposit(TokenType.UTILITY, 10000)

        # Run network step
        actions = network.step({
            "node_status": "active",
            "load_percent": 70,
            "token_balance": 10000,
            "market_price": 1.0,
            "demand": 0.5,
        })

        assert len(actions) > 0

    def test_security_with_cross_domain_integration(self):
        """Test security layer with cross-domain integration."""
        security = SecurityLayer()
        integration = DomainIntegration()

        # Add bridge for cross-domain communication
        bridge = security.add_bridge("healthcare_chain", "finance_chain")

        # Transfer across domains
        tx = bridge.lock_and_mint("health_data_token", 100, "finance_addr")

        # Create cross-domain use case
        use_case = integration.create_use_case(
            name="Healthcare-Finance Integration",
            description="Secure health data for insurance",
            domains=[DomainType.HEALTHCARE, DomainType.FINANCE],
            kpis={"data_accuracy": 99, "processing_time": 100},
        )

        assert tx["status"] == "completed"
        assert use_case is not None

    def test_optimization_driven_deployment(self):
        """Test optimization-driven deployment decisions."""
        optimizer = ContinuousOptimizer()
        roadmap = DeploymentRoadmap()

        # Run optimization
        result = optimizer.run_optimization_cycle({
            "node_uptime_percent": 99,
            "capacity_utilization": 85,
            "error_rate_percent": 0.1,
            "token_volatility": 0.05,
            "liquidity_ratio": 0.7,
            "revenue_growth_percent": 25,
            "threat_level": 0.1,
            "anomaly_score": 0.05,
            "active_users": 10000,
            "revenue": 100000,
            "capacity": 500,
            "network_load": 0.7,
        })

        # Use optimization results to inform deployment
        if result["risks"]["infrastructure"]["level"] == "low":
            roadmap.start_phase(PhaseType.PHASE_I)
            roadmap.complete_milestone("p1_m1_core_infra")

        progress = roadmap.get_overall_progress()
        assert progress["completed_milestones"] >= 1
