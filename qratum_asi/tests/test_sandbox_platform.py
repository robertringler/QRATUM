"""Tests for the Sandboxed Autonomous Action Platform.

Validates the complete sandbox platform including:
- Isolated execution with containers
- Merkle-verified communication
- Parallel and batch evaluation
- Lazy and speculative execution
- Resource throttling
- Non-blocking production integration
"""

import pytest
import time

from qratum_asi.sandbox_platform import (
    # Types
    SandboxConfig,
    SandboxProposal,
    SandboxEvaluationResult,
    ExecutionMode,
    IsolationLevel,
    ResourceAllocation,
    MetricsSnapshot,
    ProposalPriority,
    # Isolated execution
    IsolatedSandboxExecutor,
    SandboxContainer,
    MemoryIsolation,
    # Merkle verification
    MerkleVerifiedChannel,
    VerificationResult,
    AuditChainLogger,
    # Async pipeline
    AsyncEvaluationPipeline,
    PipelineStage,
    NonBlockingQueue,
    # Parallelism
    ShardedSandboxExecutor,
    WorkloadShard,
    NodeAllocation,
    BatchProposalEvaluator,
    BatchResult,
    EvaluationBatch,
    # Lightweight evaluation
    EphemeralContainer,
    ContainerLifecycle,
    AutoDestroyPolicy,
    DeterministicStubRegistry,
    ComputationStub,
    StubFidelityLevel,
    IncrementalStateEvaluator,
    StateDelta,
    StateCheckpoint,
    # Observability
    PassiveMetricsCollector,
    LatencyMetrics,
    ThroughputMetrics,
    EntropyMetrics,
    TopologicalIndices,
    DiagnosticsOffloader,
    DiagnosticJob,
    OffloadTarget,
    # Non-blocking integration
    ProposalQueue,
    QueuedProposal,
    QueuePriority,
    DualControlGateway,
    ApprovalRequest,
    AuthorizationStatus,
    # Optimization
    LazyEvaluator,
    EvaluationPolicy,
    CriticalityAssessment,
    SpeculativeExecutor,
    SpeculativeResult,
    LikelihoodEstimator,
    ResourceThrottler,
    ThrottlePolicy,
    LoadMetrics,
    # Quantum
    QuantumSandbox,
    HybridExecutor,
    QuantumIsolation,
    TensorNetworkPrecomputer,
    TensorCache,
    HybridComputation,
    # Orchestrator
    SandboxPlatformOrchestrator,
    PlatformConfig,
    PlatformStatus,
)


class TestSandboxTypes:
    """Tests for sandbox type definitions."""

    def test_resource_allocation_creation(self):
        """Test creating resource allocation."""
        alloc = ResourceAllocation(
            cpu_cores=4,
            memory_mb=2048,
            gpu_units=1.0,
            quantum_qubits=10,
        )

        assert alloc.cpu_cores == 4
        assert alloc.memory_mb == 2048
        assert alloc.gpu_units == 1.0
        assert alloc.quantum_qubits == 10

    def test_sandbox_config_creation(self):
        """Test creating sandbox configuration."""
        config = SandboxConfig(
            sandbox_id="test_sandbox",
            isolation_level=IsolationLevel.CONTAINER,
            execution_mode=ExecutionMode.ASYNC,
        )

        assert config.sandbox_id == "test_sandbox"
        assert config.isolation_level == IsolationLevel.CONTAINER
        assert config.execution_mode == ExecutionMode.ASYNC

    def test_sandbox_proposal_creation(self):
        """Test creating sandbox proposal."""
        proposal = SandboxProposal(
            proposal_id="prop_001",
            source_id="source_001",
            proposal_type="test",
            payload={"key": "value"},
            priority=ProposalPriority.HIGH,
        )

        assert proposal.proposal_id == "prop_001"
        assert proposal.proposal_type == "test"
        assert proposal.priority == ProposalPriority.HIGH

    def test_proposal_hash_deterministic(self):
        """Test proposal hash is deterministic."""
        proposal = SandboxProposal(
            proposal_id="prop_001",
            source_id="source_001",
            proposal_type="test",
            payload={"key": "value"},
        )

        hash1 = proposal.compute_hash()
        hash2 = proposal.compute_hash()

        assert hash1 == hash2
        assert len(hash1) == 64

    def test_evaluation_result_creation(self):
        """Test creating evaluation result."""
        result = SandboxEvaluationResult(
            result_id="result_001",
            proposal_id="prop_001",
            sandbox_id="sandbox_001",
            success=True,
            outcome="completed",
            fidelity_score=0.95,
        )

        assert result.success is True
        assert result.fidelity_score == 0.95


class TestIsolatedSandboxExecutor:
    """Tests for isolated sandbox executor."""

    def test_executor_creation(self):
        """Test creating isolated executor."""
        executor = IsolatedSandboxExecutor(executor_id="test")
        assert executor.executor_id == "test"

    def test_container_creation(self):
        """Test creating container."""
        executor = IsolatedSandboxExecutor()
        container = executor.create_container("sandbox_001")

        assert container.sandbox_id == "sandbox_001"
        assert container.memory_isolation is not None
        assert container.memory_isolation.is_isolated is True

    def test_sync_execution(self):
        """Test synchronous execution."""
        executor = IsolatedSandboxExecutor()
        container = executor.create_container("sandbox_002")

        proposal = SandboxProposal(
            proposal_id="prop_001",
            source_id="test",
            proposal_type="test",
            payload={"value": 42},
        )

        def executor_func(prop, ctx):
            return {"result": prop.payload["value"] * 2}

        result = executor.execute(
            container,
            proposal,
            executor_func,
            async_mode=False,
        )

        assert result is not None
        assert result.success is True

    def test_container_stats(self):
        """Test executor statistics."""
        executor = IsolatedSandboxExecutor()
        executor.create_container("sandbox_003")

        stats = executor.get_executor_stats()

        assert "total_containers" in stats
        assert stats["total_containers"] >= 1


class TestMerkleVerifiedChannel:
    """Tests for Merkle-verified communication."""

    def test_channel_creation(self):
        """Test creating verified channel."""
        channel = MerkleVerifiedChannel(channel_id="test_channel")
        assert channel.channel_id == "test_channel"

    def test_message_send_receive(self):
        """Test sending and receiving verified messages."""
        channel = MerkleVerifiedChannel(channel_id="test_channel")

        message = channel.send(
            payload={"data": "test"},
            sender_id="sender_001",
        )

        assert message.content_hash is not None
        assert message.merkle_proof is not None

        verification = channel.receive(message)
        assert verification.is_valid is True

    def test_channel_stats(self):
        """Test channel statistics."""
        channel = MerkleVerifiedChannel(channel_id="test_channel")
        channel.send({"data": "test"}, "sender")

        stats = channel.get_channel_stats()
        assert stats["messages_sent"] == 1


class TestAuditChainLogger:
    """Tests for audit chain logger."""

    def test_logger_creation(self):
        """Test creating audit logger."""
        logger = AuditChainLogger(logger_id="test_audit")
        assert logger.logger_id == "test_audit"

    def test_sync_logging(self):
        """Test synchronous logging."""
        logger = AuditChainLogger()

        event_hash = logger.log(
            "test_event",
            {"key": "value"},
            sync=True,
        )

        assert event_hash is not None
        assert logger.verify_chain() is True

    def test_async_logging_flush(self):
        """Test async logging with flush."""
        logger = AuditChainLogger()

        logger.log("event_1", {"data": 1}, sync=False)
        logger.log("event_2", {"data": 2}, sync=False)

        count = logger.flush()
        assert count == 2


class TestAsyncPipeline:
    """Tests for async evaluation pipeline."""

    def test_pipeline_creation(self):
        """Test creating pipeline."""
        pipeline = AsyncEvaluationPipeline(pipeline_id="test")
        assert pipeline.pipeline_id == "test"

    def test_pipeline_stage_addition(self):
        """Test adding pipeline stages."""
        pipeline = AsyncEvaluationPipeline()

        def process(proposal, ctx):
            return proposal

        stage = pipeline.add_stage("validation", process)
        assert stage.stage_name == "validation"
        assert len(pipeline.stages) == 1

    def test_non_blocking_queue(self):
        """Test non-blocking queue."""
        queue = NonBlockingQueue(queue_id="test_queue")

        proposal = SandboxProposal(
            proposal_id="prop_001",
            source_id="test",
            proposal_type="test",
            payload={},
        )

        item = queue.enqueue(proposal)
        assert item is not None

        dequeued = queue.dequeue()
        assert dequeued.proposal.proposal_id == "prop_001"


class TestShardedExecutor:
    """Tests for sharded executor."""

    def test_executor_creation(self):
        """Test creating sharded executor."""
        executor = ShardedSandboxExecutor(executor_id="test")
        assert executor.executor_id == "test"
        assert len(executor.nodes) > 0

    def test_node_addition(self):
        """Test adding compute nodes."""
        from qratum_asi.sandbox_platform.types import ResourceType

        executor = ShardedSandboxExecutor()
        node = executor.add_node(
            resource_type=ResourceType.GPU,
            capacity=100.0,
            max_concurrent=5,
        )

        assert node.resource_type == ResourceType.GPU
        assert node.capacity == 100.0

    def test_shard_creation(self):
        """Test creating workload shards."""
        executor = ShardedSandboxExecutor()

        proposals = [
            SandboxProposal(
                proposal_id=f"prop_{i}",
                source_id="test",
                proposal_type="test",
                payload={},
            )
            for i in range(5)
        ]

        shard = executor.create_shard(proposals)
        assert len(shard.proposals) == 5


class TestBatchEvaluator:
    """Tests for batch evaluator."""

    def test_evaluator_creation(self):
        """Test creating batch evaluator."""
        evaluator = BatchProposalEvaluator(evaluator_id="test")
        assert evaluator.evaluator_id == "test"

    def test_proposal_submission(self):
        """Test submitting proposals for batching."""
        evaluator = BatchProposalEvaluator(default_batch_size=10)

        proposal = SandboxProposal(
            proposal_id="prop_001",
            source_id="test",
            proposal_type="test",
            payload={},
        )

        batch_id = evaluator.submit(proposal)
        assert batch_id is not None

    def test_batch_evaluation(self):
        """Test evaluating a batch."""
        evaluator = BatchProposalEvaluator(default_batch_size=5)

        for i in range(3):
            evaluator.submit(
                SandboxProposal(
                    proposal_id=f"prop_{i}",
                    source_id="test",
                    proposal_type="test",
                    payload={"index": i},
                )
            )

        evaluator.flush()

        def executor(prop):
            return SandboxEvaluationResult(
                result_id=f"result_{prop.proposal_id}",
                proposal_id=prop.proposal_id,
                sandbox_id="test",
                success=True,
                outcome="ok",
                fidelity_score=1.0,
            )

        results = evaluator.evaluate_all_ready(executor)
        assert len(results) == 1  # One batch
        assert results[0].success is True


class TestEphemeralContainer:
    """Tests for ephemeral containers."""

    def test_container_creation(self):
        """Test creating ephemeral container."""
        container = EphemeralContainer(
            container_id="test",
            auto_destroy_policy=AutoDestroyPolicy.MANUAL,
        )

        assert container.container_id == "test"
        assert container.is_alive is True

    def test_container_execution(self):
        """Test executing in container."""
        container = EphemeralContainer(
            container_id="test",
            auto_destroy_policy=AutoDestroyPolicy.MANUAL,
        )

        result = container.execute(
            lambda ctx: ctx.get("initialized", False),
        )

        assert result is True

    def test_container_auto_destroy(self):
        """Test auto-destroy policy."""
        container = EphemeralContainer(
            container_id="test",
            auto_destroy_policy=AutoDestroyPolicy.IMMEDIATE,
        )

        container.execute(lambda ctx: True)

        # Should be destroyed after execution
        assert container.is_alive is False


class TestDeterministicStubs:
    """Tests for deterministic computation stubs."""

    def test_registry_creation(self):
        """Test creating stub registry."""
        registry = DeterministicStubRegistry()
        assert len(registry.stubs) > 0

    def test_stub_execution(self):
        """Test executing stub computation."""
        registry = DeterministicStubRegistry()

        result = registry.execute(
            "matrix_multiply",
            {"shape": (10, 10)},
            fidelity_level=StubFidelityLevel.MEDIUM,
        )

        assert result.is_stubbed is True
        assert result.fidelity_level == StubFidelityLevel.MEDIUM

    def test_cache_hit(self):
        """Test stub result caching."""
        registry = DeterministicStubRegistry()

        # First execution
        registry.execute(
            "neural_inference",
            {"input_size": 100},
            fidelity_level=StubFidelityLevel.HIGH,
        )

        # Second execution should hit cache
        registry.execute(
            "neural_inference",
            {"input_size": 100},
            fidelity_level=StubFidelityLevel.HIGH,
        )

        stats = registry.get_registry_stats()
        assert stats["cache_hits"] >= 1


class TestIncrementalState:
    """Tests for incremental state evaluator."""

    def test_evaluator_creation(self):
        """Test creating incremental evaluator."""
        evaluator = IncrementalStateEvaluator(evaluator_id="test")
        assert evaluator.evaluator_id == "test"

    def test_state_set_get(self):
        """Test setting and getting state values."""
        evaluator = IncrementalStateEvaluator()

        delta = evaluator.set_value("config.setting", 42)
        assert delta.new_value == 42

        value = evaluator.get_value("config.setting")
        assert value == 42

    def test_incremental_evaluation(self):
        """Test incremental evaluation."""
        evaluator = IncrementalStateEvaluator()

        evaluator.set_value("data.count", 10)

        def eval_func(state, deltas):
            return {"total": state.get("data", {}).get("count", 0)}

        result, was_cached = evaluator.evaluate(eval_func)
        assert result["total"] == 10
        assert was_cached is False

        # Second evaluation should use cache
        result2, was_cached2 = evaluator.evaluate(eval_func)
        assert was_cached2 is True


class TestPassiveMetrics:
    """Tests for passive metrics collection."""

    def test_collector_creation(self):
        """Test creating metrics collector."""
        collector = PassiveMetricsCollector(collector_id="test")
        assert collector.collector_id == "test"

    def test_latency_recording(self):
        """Test recording latency samples."""
        collector = PassiveMetricsCollector()

        collector.record_latency("operation", 10.5)
        collector.record_latency("operation", 15.2)
        collector.record_latency("operation", 12.8)

        metrics = collector.get_latency_metrics("operation")
        assert metrics.sample_count == 3
        assert metrics.min_ms == 10.5
        assert metrics.max_ms == 15.2

    def test_throughput_recording(self):
        """Test recording throughput."""
        collector = PassiveMetricsCollector()

        collector.record_operation("process", count=10, bytes_processed=1000)
        collector.record_operation("process", count=5, bytes_processed=500)

        metrics = collector.get_throughput_metrics("process")
        assert metrics.total_operations == 15

    def test_entropy_recording(self):
        """Test recording entropy metrics."""
        collector = PassiveMetricsCollector()

        metrics = collector.record_entropy("state", {"key": "value", "count": 42})
        assert metrics.shannon_entropy >= 0


class TestDiagnosticsOffloader:
    """Tests for diagnostics offloading."""

    def test_offloader_creation(self):
        """Test creating diagnostics offloader."""
        offloader = DiagnosticsOffloader(offloader_id="test")
        assert offloader.offloader_id == "test"

    def test_job_submission(self):
        """Test submitting diagnostic job."""
        from qratum_asi.sandbox_platform.diagnostics_offload import DiagnosticType

        offloader = DiagnosticsOffloader()

        job = offloader.submit(
            DiagnosticType.PROFILING,
            {"target": "function_x"},
        )

        assert job.job_type == DiagnosticType.PROFILING


class TestProposalQueue:
    """Tests for proposal queue."""

    def test_queue_creation(self):
        """Test creating proposal queue."""
        queue = ProposalQueue(queue_id="test")
        assert queue.queue_id == "test"

    def test_proposal_enqueue(self):
        """Test enqueueing proposals."""
        queue = ProposalQueue()

        proposal = SandboxProposal(
            proposal_id="prop_001",
            source_id="test",
            proposal_type="test",
            payload={},
        )

        queued = queue.enqueue(proposal, priority=QueuePriority.HIGH)
        assert queued is not None
        assert queued.priority == QueuePriority.HIGH


class TestDualControlGateway:
    """Tests for dual-control approval gateway."""

    def test_gateway_creation(self):
        """Test creating approval gateway."""
        gateway = DualControlGateway(gateway_id="test")
        assert gateway.gateway_id == "test"

    def test_approval_request_creation(self):
        """Test creating approval request."""
        gateway = DualControlGateway()

        request = gateway.create_request(
            resource_id="prop_001",
            resource_type="proposal",
            description="Test proposal for approval",
        )

        assert request.status == AuthorizationStatus.PENDING
        assert len(request.required_approvers) >= 2

    def test_dual_control_approval(self):
        """Test dual-control approval workflow."""
        gateway = DualControlGateway()

        request = gateway.create_request(
            resource_id="prop_001",
            resource_type="proposal",
            description="Test proposal",
        )

        # First approval
        gateway.add_approval(
            request.request_id,
            "approver_1",
            "approve",
            "Looks good",
        )

        assert request.status == AuthorizationStatus.AWAITING_SECOND

        # Second approval
        gateway.add_approval(
            request.request_id,
            "approver_2",
            "approve",
            "Also approved",
        )

        assert request.status == AuthorizationStatus.APPROVED
        assert request.is_approved is True


class TestLazyEvaluator:
    """Tests for lazy evaluation."""

    def test_evaluator_creation(self):
        """Test creating lazy evaluator."""
        evaluator = LazyEvaluator(evaluator_id="test")
        assert evaluator.evaluator_id == "test"

    def test_criticality_assessment(self):
        """Test assessing proposal criticality."""
        evaluator = LazyEvaluator()

        proposal = SandboxProposal(
            proposal_id="prop_001",
            source_id="test",
            proposal_type="test",
            payload={},
            priority=ProposalPriority.CRITICAL,
            target_subsystems=["security"],
        )

        assessment = evaluator.assess_criticality(proposal)
        assert assessment.should_evaluate is True

    def test_lazy_skip(self):
        """Test skipping non-critical proposals."""
        evaluator = LazyEvaluator(
            policy=EvaluationPolicy.CRITICAL_ONLY,
        )

        proposal = SandboxProposal(
            proposal_id="prop_001",
            source_id="test",
            proposal_type="test",
            payload={},
            priority=ProposalPriority.LOW,
            target_subsystems=[],
        )

        result = evaluator.evaluate(
            proposal,
            lambda p: {"result": "evaluated"},
        )

        assert result.evaluated is False
        assert result.saved_time_ms > 0


class TestSpeculativeExecutor:
    """Tests for speculative execution."""

    def test_executor_creation(self):
        """Test creating speculative executor."""
        executor = SpeculativeExecutor(executor_id="test")
        assert executor.executor_id == "test"

    def test_likelihood_estimation(self):
        """Test estimating approval likelihood."""
        executor = SpeculativeExecutor()

        proposal = SandboxProposal(
            proposal_id="prop_001",
            source_id="test",
            proposal_type="standard",
            payload={},
            priority=ProposalPriority.HIGH,
        )

        estimate = executor.estimator.estimate(proposal)
        assert 0 <= estimate.likelihood <= 1


class TestResourceThrottler:
    """Tests for resource throttling."""

    def test_throttler_creation(self):
        """Test creating resource throttler."""
        throttler = ResourceThrottler(throttler_id="test")
        assert throttler.throttler_id == "test"

    def test_load_update(self):
        """Test updating load metrics."""
        throttler = ResourceThrottler()

        metrics = LoadMetrics(
            metric_id="test",
            cpu_utilization=0.5,
            memory_utilization=0.6,
        )

        state = throttler.update_load(metrics)
        assert state is not None

    def test_budget_throttling(self):
        """Test budget throttling based on load."""
        from qratum_asi.sandbox_platform.resource_throttler import ThrottleLevel

        throttler = ResourceThrottler()

        # Set high load
        throttler.set_throttle_level(ThrottleLevel.HEAVY, "high load")

        budget = throttler.get_available_budget()
        # Heavy throttle should reduce budget to 25%
        assert budget.cpu_cores < throttler.base_budget.cpu_cores


class TestQuantumSandbox:
    """Tests for quantum sandbox."""

    def test_sandbox_creation(self):
        """Test creating quantum sandbox."""
        sandbox = QuantumSandbox(sandbox_id="test")
        assert sandbox.sandbox_id == "test"

    def test_isolation_creation(self):
        """Test creating quantum isolation."""
        from qratum_asi.sandbox_platform.quantum_sandbox import (
            QuantumIsolationLevel,
            QuantumBackend,
        )

        sandbox = QuantumSandbox()
        isolation = sandbox.create_isolation(
            level=QuantumIsolationLevel.ISOLATED,
            qubits=8,
        )

        assert isolation.qubits_allocated == 8
        assert isolation.is_isolated is True

    def test_job_submission_execution(self):
        """Test submitting and executing quantum job."""
        sandbox = QuantumSandbox()

        job = sandbox.submit_job(
            circuit_description={"qubits": 4, "gates": ["H", "CNOT"]},
            shots=100,
        )

        result = sandbox.execute_job(job.job_id)
        assert "counts" in result
        assert result["success"] is True


class TestTensorNetwork:
    """Tests for tensor network precomputer."""

    def test_precomputer_creation(self):
        """Test creating tensor precomputer."""
        precomputer = TensorNetworkPrecomputer(precomputer_id="test")
        assert precomputer.precomputer_id == "test"

    def test_tensor_contraction_cache(self):
        """Test caching tensor contraction."""
        from qratum_asi.sandbox_platform.tensor_network import TensorComputationType

        precomputer = TensorNetworkPrecomputer()

        def compute(params):
            return {"contracted": True}

        result = precomputer.precompute(
            TensorComputationType.CONTRACTION,
            {"tensor_a": [1, 2], "tensor_b": [3, 4]},
            compute,
            background=False,
        )

        assert result is not None


class TestSandboxPlatformOrchestrator:
    """Tests for main platform orchestrator."""

    def test_orchestrator_creation(self):
        """Test creating platform orchestrator."""
        orchestrator = SandboxPlatformOrchestrator()
        assert orchestrator.status == PlatformStatus.READY

    def test_orchestrator_with_config(self):
        """Test orchestrator with custom config."""
        config = PlatformConfig(
            platform_id="custom_platform",
            enable_parallelism=True,
            enable_speculative=True,
            enable_lazy=True,
            enable_quantum=False,
        )

        orchestrator = SandboxPlatformOrchestrator(config=config)
        assert orchestrator.config.platform_id == "custom_platform"

    def test_proposal_submission(self):
        """Test submitting proposal through orchestrator."""
        orchestrator = SandboxPlatformOrchestrator()

        proposal = orchestrator.submit_proposal(
            proposal_type="test",
            payload={"key": "value"},
            priority=ProposalPriority.NORMAL,
        )

        assert proposal.proposal_id is not None
        assert proposal.proposal_type == "test"

    def test_approval_workflow(self):
        """Test approval workflow through orchestrator."""
        orchestrator = SandboxPlatformOrchestrator()

        proposal = orchestrator.submit_proposal(
            proposal_type="sensitive",
            payload={"action": "modify"},
        )

        request_id = orchestrator.request_approval(
            proposal,
            "Test proposal requiring approval",
        )

        # Add approvals
        orchestrator.approve(request_id, "approver_1", True, "Approved")
        orchestrator.approve(request_id, "approver_2", True, "Also approved")

        # Check authorization
        assert orchestrator.dual_control_gateway.check_authorization(proposal.proposal_id)

    def test_platform_stats(self):
        """Test getting platform statistics."""
        orchestrator = SandboxPlatformOrchestrator()

        stats = orchestrator.get_platform_stats()

        assert "platform_id" in stats
        assert "status" in stats
        assert "config" in stats
        assert stats["merkle_chain_valid"] is True

    def test_provenance_verification(self):
        """Test provenance chain verification."""
        orchestrator = SandboxPlatformOrchestrator()

        # Submit some proposals
        for i in range(3):
            orchestrator.submit_proposal(
                proposal_type=f"test_{i}",
                payload={"index": i},
            )

        assert orchestrator.verify_provenance() is True

    def test_metrics_snapshot(self):
        """Test getting metrics snapshot."""
        orchestrator = SandboxPlatformOrchestrator()

        snapshot = orchestrator.get_metrics_snapshot()

        assert snapshot.sandbox_id == orchestrator.config.platform_id


class TestIntegration:
    """Integration tests for the complete platform."""

    def test_full_evaluation_workflow(self):
        """Test complete proposal evaluation workflow."""
        config = PlatformConfig(
            platform_id="integration_test",
            enable_parallelism=True,
            enable_speculative=False,  # Disable for deterministic test
            enable_lazy=True,
            enable_quantum=False,
            num_workers=2,
        )

        orchestrator = SandboxPlatformOrchestrator(config=config)
        orchestrator.start()

        try:
            # Submit multiple proposals
            proposals = []
            for i in range(5):
                proposal = orchestrator.submit_proposal(
                    proposal_type=f"test_{i}",
                    payload={"index": i, "value": i * 10},
                    priority=ProposalPriority.NORMAL,
                )
                proposals.append(proposal)

            # Give time for processing
            time.sleep(0.1)

            # Get statistics
            stats = orchestrator.get_platform_stats()
            assert stats["status"] == PlatformStatus.RUNNING.value

            # Verify provenance
            assert orchestrator.verify_provenance() is True

        finally:
            orchestrator.stop()
            assert orchestrator.status == PlatformStatus.STOPPED

    def test_dual_control_integration(self):
        """Test dual-control integration with evaluation."""
        orchestrator = SandboxPlatformOrchestrator()

        # Submit proposal requiring approval
        proposal = orchestrator.submit_proposal(
            proposal_type="critical_change",
            payload={"action": "modify_core"},
            priority=ProposalPriority.CRITICAL,
            requires_approval=True,
        )

        # Create approval request
        request_id = orchestrator.request_approval(
            proposal,
            "Critical change requiring dual-control approval",
        )

        # Verify not yet authorized
        assert not orchestrator.dual_control_gateway.check_authorization(proposal.proposal_id)

        # Add dual approvals
        orchestrator.approve(request_id, "admin_1", True, "Approved after review")
        orchestrator.approve(request_id, "admin_2", True, "Confirmed")

        # Now should be authorized
        assert orchestrator.dual_control_gateway.check_authorization(proposal.proposal_id)

    def test_resource_throttling_integration(self):
        """Test resource throttling under load."""
        from qratum_asi.sandbox_platform.resource_throttler import ThrottleLevel

        orchestrator = SandboxPlatformOrchestrator()

        # Simulate high load
        orchestrator.resource_throttler.set_throttle_level(
            ThrottleLevel.HEAVY,
            "simulated_high_load",
        )

        # Check budget is reduced
        budget = orchestrator.resource_throttler.get_available_budget()
        assert budget.cpu_cores < orchestrator.resource_throttler.base_budget.cpu_cores

        # Reset to normal
        orchestrator.resource_throttler.set_throttle_level(
            ThrottleLevel.NONE,
            "load_normalized",
        )

        budget_normal = orchestrator.resource_throttler.get_available_budget()
        assert budget_normal.cpu_cores == orchestrator.resource_throttler.base_budget.cpu_cores


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
