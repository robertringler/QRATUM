"""Unit tests for QuASIM×QuNimbus Phase VII components."""

from quasim.qunimbus.phaseVII import (DVLLedger, QMPActivation, TrustKernel,
                                      ValuationEngine)


class TestQMPActivation:
    """Tests for Quantum Market Protocol activation."""

    def test_initialization(self):
        """Test QMP activation initialization."""

        qmp = QMPActivation()
        assert not qmp.is_active
        assert len(qmp.liquidity_partners) == 3
        assert qmp.market_update_latency_target == 10.0
        assert qmp.entanglement_throughput_target == 5e9

    def test_activation(self):
        """Test QMP activation."""

        qmp = QMPActivation()
        result = qmp.activate()

        assert result["status"] == "active"
        assert result["partners_connected"] == 3
        assert result["latency_target_ms"] == 10000.0
        assert result["throughput_target_eph"] == 5e9
        assert qmp.is_active

    def test_deactivation(self):
        """Test QMP deactivation."""

        qmp = QMPActivation()
        qmp.activate()
        result = qmp.deactivate()

        assert result["status"] == "inactive"
        assert result["partners_disconnected"] == 3
        assert not qmp.is_active

    def test_market_feed(self):
        """Test market feed status."""

        qmp = QMPActivation()
        qmp.activate()
        feed = qmp.get_market_feed()

        assert feed["is_active"]
        assert "market_feed" in feed
        assert len(feed["market_feed"]) == 3
        assert all(p["status"] == "connected" for p in feed["market_feed"].values())

    def test_update_price_metrics(self):
        """Test price metrics update."""

        qmp = QMPActivation()
        qmp.activate()

        result = qmp.update_price_metrics(eta_ent=0.97, phi_qevf=1000.0)

        assert result["eta_ent"] == 0.97
        assert result["phi_qevf"] == 1000.0
        assert "price_multiplier" in result
        assert "market_value" in result
        assert result["update_latency_ms"] < 10.0

    def test_get_metrics(self):
        """Test QMP metrics retrieval."""

        qmp = QMPActivation()
        qmp.activate()
        metrics = qmp.get_metrics()

        assert metrics["is_active"]
        assert metrics["market_update_latency_ms"] < 10.0
        assert metrics["latency_within_target"]
        assert metrics["entanglement_throughput_eph"] > 5e9
        assert metrics["throughput_within_target"]
        assert metrics["partners_active"] == 3

    def test_custom_liquidity_partners(self):
        """Test custom liquidity partner configuration."""

        partners = ["partner_a", "partner_b"]
        qmp = QMPActivation(liquidity_partners=partners)

        assert qmp.liquidity_partners == partners
        assert len(qmp.liquidity_partners) == 2


class TestValuationEngine:
    """Tests for Dynamic Φ-Valuation Engine."""

    def test_initialization(self):
        """Test valuation engine initialization."""

        engine = ValuationEngine()
        assert engine.base_phi_value == 1000.0
        assert engine.eta_baseline == 0.95
        assert engine.coherence_variance_threshold == 0.02
        assert len(engine.valuation_history) == 0

    def test_calculate_phi_qevf(self):
        """Test Φ_QEVF calculation."""

        engine = ValuationEngine()
        phi_qevf = engine.calculate_phi_qevf(
            eta_ent=0.97, coherence_variance=0.015, runtime_hours=100.0
        )

        assert phi_qevf > 0
        # Verify calculation includes efficiency and coherence factors
        # phi_qevf = base * (eta/baseline) * coherence_penalty * runtime_factor
        # With runtime capped at 120h, 100h gives factor of ~0.833
        assert phi_qevf > 0 and phi_qevf < engine.base_phi_value * 2

    def test_map_eta_to_price_metrics(self):
        """Test η_ent to price metrics mapping."""

        engine = ValuationEngine()
        metrics = engine.map_eta_to_price_metrics(eta_ent=0.97)

        assert metrics["eta_ent"] == 0.97
        assert "phi_qevf" in metrics
        assert "eph_price" in metrics
        assert "coherence_variance" in metrics
        assert metrics["coherence_within_threshold"]
        assert len(engine.valuation_history) == 1

    def test_valuation_history_limit(self):
        """Test valuation history is limited to 1000 records."""

        engine = ValuationEngine()

        # Add more than 1000 records
        for i in range(1100):
            engine.map_eta_to_price_metrics(eta_ent=0.95 + i * 0.0001)

        assert len(engine.valuation_history) == 1000

    def test_get_valuation_metrics(self):
        """Test valuation metrics retrieval."""

        engine = ValuationEngine()
        engine.map_eta_to_price_metrics(eta_ent=0.97)
        metrics = engine.get_valuation_metrics()

        assert metrics["base_phi_value"] == 1000.0
        assert metrics["eta_baseline"] == 0.95
        assert metrics["coherence_variance_threshold"] == 0.02
        assert metrics["history_records"] == 1
        assert "avg_phi_qevf" in metrics

    def test_reset_history(self):
        """Test history reset."""

        engine = ValuationEngine()
        engine.map_eta_to_price_metrics(eta_ent=0.97)
        assert len(engine.valuation_history) == 1

        engine.reset_history()
        assert len(engine.valuation_history) == 0

    def test_coherence_variance_penalty(self):
        """Test that high coherence variance reduces Φ_QEVF."""

        engine = ValuationEngine()

        # Low variance (good)
        phi_low = engine.calculate_phi_qevf(
            eta_ent=0.97, coherence_variance=0.01, runtime_hours=100.0
        )

        # High variance (bad)
        phi_high = engine.calculate_phi_qevf(
            eta_ent=0.97, coherence_variance=0.019, runtime_hours=100.0
        )

        assert phi_low > phi_high


class TestDVLLedger:
    """Tests for Decentralized Verification Ledger."""

    def test_initialization(self):
        """Test DVL ledger initialization."""

        ledger = DVLLedger()
        assert len(ledger.chain) == 1  # Genesis block
        assert len(ledger.compliance_frameworks) == 6
        assert ledger.verify_chain()

    def test_add_block(self):
        """Test adding a block to the ledger."""

        ledger = DVLLedger()
        block = ledger.add_block(phi_qevf=1000.0, eta_ent=0.97)

        assert block.index == 1
        assert block.phi_qevf == 1000.0
        assert block.eta_ent == 0.97
        assert len(ledger.chain) == 2

    def test_chain_verification(self):
        """Test chain integrity verification."""

        ledger = DVLLedger()
        ledger.add_block(phi_qevf=1000.0, eta_ent=0.97)
        ledger.add_block(phi_qevf=1050.0, eta_ent=0.98)

        assert ledger.verify_chain()
        assert len(ledger.chain) == 3

    def test_chain_tampering_detection(self):
        """Test that chain detects tampering."""

        ledger = DVLLedger()
        ledger.add_block(phi_qevf=1000.0, eta_ent=0.97)

        # Tamper with a block
        ledger.chain[1].phi_qevf = 2000.0

        # Chain should be invalid
        assert not ledger.verify_chain()

    def test_get_latest_block(self):
        """Test getting latest block."""

        ledger = DVLLedger()
        ledger.add_block(phi_qevf=1000.0, eta_ent=0.97)
        latest = ledger.get_latest_block()

        assert latest.index == 1
        assert latest.phi_qevf == 1000.0

    def test_get_chain_summary(self):
        """Test chain summary."""

        ledger = DVLLedger()
        ledger.add_block(phi_qevf=1000.0, eta_ent=0.97)
        summary = ledger.get_chain_summary()

        assert summary["chain_length"] == 2
        assert summary["is_valid"]
        assert summary["latest_phi_qevf"] == 1000.0
        assert summary["latest_eta_ent"] == 0.97
        assert len(summary["compliance_frameworks"]) == 6

    def test_export_for_grafana(self):
        """Test Grafana export format."""

        ledger = DVLLedger()
        ledger.add_block(phi_qevf=1000.0, eta_ent=0.97)
        export = ledger.export_for_grafana()

        assert len(export) == 2
        assert all("hash" in block for block in export)
        assert all("timestamp" in block for block in export)

    def test_get_attestation_history(self):
        """Test attestation history for specific framework."""

        ledger = DVLLedger()
        ledger.add_block(phi_qevf=1000.0, eta_ent=0.97)
        ledger.add_block(phi_qevf=1050.0, eta_ent=0.98)

        history = ledger.get_attestation_history("DO-178C")
        assert len(history) == 3  # Genesis + 2 blocks
        assert all("attestation" in h for h in history)

    def test_custom_compliance_frameworks(self):
        """Test custom compliance frameworks."""

        frameworks = ["ISO-27001", "GDPR"]
        ledger = DVLLedger(compliance_frameworks=frameworks)

        assert ledger.compliance_frameworks == frameworks
        assert len(ledger.chain[0].compliance_attestations) == 2


class TestTrustKernel:
    """Tests for Trust Kernel."""

    def test_initialization(self):
        """Test trust kernel initialization."""

        kernel = TrustKernel()
        assert len(kernel.regions) == 6
        assert kernel.canary_percentage == 0.05
        assert kernel.mtbf_target_hours == 120.0

    def test_get_region_status(self):
        """Test getting region status."""

        kernel = TrustKernel()
        status = kernel.get_region_status("Americas")

        assert status["region"] == "Americas"
        assert status["status"] == "active"
        assert status["trust_score"] == 1.0
        assert status["mtbf_target_hours"] == 120.0

    def test_update_region_status(self):
        """Test updating region status."""

        kernel = TrustKernel()
        result = kernel.update_region_status(
            "EU", status="degraded", trust_score=0.8, uptime_hours=100.0
        )

        assert result["status"] == "degraded"
        assert result["trust_score"] == 0.8
        assert result["uptime_hours"] == 100.0

    def test_get_orchestration_mesh_status(self):
        """Test orchestration mesh status."""

        kernel = TrustKernel()
        mesh = kernel.get_orchestration_mesh_status()

        assert mesh["total_regions"] == 6
        assert mesh["active_regions"] == 6
        assert mesh["avg_trust_score"] == 1.0
        assert mesh["mtbf_target_hours"] == 120.0
        assert "region_status" in mesh

    def test_configure_canary_deployment(self):
        """Test canary deployment configuration."""

        kernel = TrustKernel()
        config = kernel.configure_canary_deployment()

        assert config["canary_region"] == "Americas"
        assert config["canary_percentage"] == 0.05
        assert config["deployment_strategy"] == "blue-green"
        assert len(config["rollout_regions"]) == 5

    def test_verify_compliance_continuous(self):
        """Test continuous compliance verification."""

        kernel = TrustKernel()
        compliance = kernel.verify_compliance_continuous()

        assert compliance["all_compliant"]
        assert "ISO-27001" in compliance["compliance_checks"]
        assert "ITAR" in compliance["compliance_checks"]
        assert "GDPR" in compliance["compliance_checks"]
        assert all(check["compliant"] for check in compliance["compliance_checks"].values())

    def test_get_metrics(self):
        """Test trust kernel metrics."""

        kernel = TrustKernel()
        metrics = kernel.get_metrics()

        assert "initialization_timestamp" in metrics
        assert metrics["regions"] == kernel.REGIONS
        assert metrics["canary_percentage"] == 0.05
        assert "mesh_status" in metrics
        assert "compliance_status" in metrics

    def test_custom_regions(self):
        """Test custom region configuration."""

        regions = ["Americas", "EU", "APAC"]
        kernel = TrustKernel(regions=regions)

        assert kernel.regions == regions
        assert len(kernel.region_status) == 3

    def test_trust_score_boundaries(self):
        """Test trust score is bounded between 0 and 1."""

        kernel = TrustKernel()

        # Try to set trust score above 1
        kernel.update_region_status("Americas", "active", trust_score=1.5)
        status = kernel.get_region_status("Americas")
        assert status["trust_score"] == 1.0

        # Try to set trust score below 0
        kernel.update_region_status("Americas", "active", trust_score=-0.5)
        status = kernel.get_region_status("Americas")
        assert status["trust_score"] == 0.0

    def test_mtbf_compliance(self):
        """Test MTBF compliance checking."""

        kernel = TrustKernel()

        # Set all regions to meet MTBF target
        for region in kernel.regions:
            kernel.update_region_status(region, "active", uptime_hours=125.0)

        mesh = kernel.get_orchestration_mesh_status()
        assert mesh["mtbf_compliance"]
        assert mesh["avg_uptime_hours"] >= 120.0
