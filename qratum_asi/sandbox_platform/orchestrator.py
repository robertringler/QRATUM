"""Main Orchestrator for Sandboxed Autonomous Action Platform.

Coordinates all sandbox platform components to provide a unified
interface for sandboxed autonomous action evaluation.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain

from qratum_asi.sandbox_platform.types import (
    ExecutionMode,
    IsolationLevel,
    MetricsSnapshot,
    ProposalPriority,
    ResourceAllocation,
    SandboxConfig,
    SandboxEvaluationResult,
    SandboxProposal,
)
from qratum_asi.sandbox_platform.isolated_sandbox import IsolatedSandboxExecutor
from qratum_asi.sandbox_platform.merkle_verifier import (
    AuditChainLogger,
    MerkleVerifiedChannel,
)
from qratum_asi.sandbox_platform.async_pipeline import AsyncEvaluationPipeline
from qratum_asi.sandbox_platform.sharded_executor import ShardedSandboxExecutor
from qratum_asi.sandbox_platform.batch_evaluator import BatchProposalEvaluator
from qratum_asi.sandbox_platform.ephemeral_container import EphemeralContainerPool
from qratum_asi.sandbox_platform.deterministic_stubs import DeterministicStubRegistry
from qratum_asi.sandbox_platform.incremental_state import IncrementalStateEvaluator
from qratum_asi.sandbox_platform.passive_metrics import PassiveMetricsCollector
from qratum_asi.sandbox_platform.diagnostics_offload import DiagnosticsOffloader
from qratum_asi.sandbox_platform.proposal_queue import ProposalQueue
from qratum_asi.sandbox_platform.dual_control_gateway import DualControlGateway
from qratum_asi.sandbox_platform.lazy_evaluator import LazyEvaluator
from qratum_asi.sandbox_platform.speculative_executor import SpeculativeExecutor
from qratum_asi.sandbox_platform.resource_throttler import ResourceThrottler
from qratum_asi.sandbox_platform.quantum_sandbox import QuantumSandbox, HybridExecutor
from qratum_asi.sandbox_platform.tensor_network import TensorNetworkPrecomputer


class PlatformStatus(Enum):
    """Status of the sandbox platform."""

    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    SHUTTING_DOWN = "shutting_down"
    STOPPED = "stopped"


@dataclass
class PlatformConfig:
    """Configuration for the sandbox platform.

    Attributes:
        platform_id: Unique platform identifier
        enable_parallelism: Enable parallel execution
        enable_speculative: Enable speculative execution
        enable_lazy: Enable lazy evaluation
        enable_quantum: Enable quantum sandbox
        num_workers: Number of worker threads
        max_queue_size: Maximum queue size
        batch_size: Default batch size
        throttle_threshold: Load threshold for throttling
    """

    platform_id: str = "sandbox_platform"
    enable_parallelism: bool = True
    enable_speculative: bool = True
    enable_lazy: bool = True
    enable_quantum: bool = False
    num_workers: int = 4
    max_queue_size: int = 10000
    batch_size: int = 50
    throttle_threshold: float = 0.8
    default_isolation: IsolationLevel = IsolationLevel.CONTAINER

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration."""
        return {
            "platform_id": self.platform_id,
            "enable_parallelism": self.enable_parallelism,
            "enable_speculative": self.enable_speculative,
            "enable_lazy": self.enable_lazy,
            "enable_quantum": self.enable_quantum,
            "num_workers": self.num_workers,
            "max_queue_size": self.max_queue_size,
            "batch_size": self.batch_size,
            "throttle_threshold": self.throttle_threshold,
            "default_isolation": self.default_isolation.value,
        }


class SandboxPlatformOrchestrator:
    """Main orchestrator for the Sandboxed Autonomous Action Platform.

    Coordinates all platform components to provide:
    - Decoupled execution with isolated containers
    - Merkle-verified communication
    - Parallel and batch evaluation
    - Lazy and speculative execution
    - Resource throttling
    - Full observability
    - Non-blocking production integration

    This is the entry point for all sandboxed autonomous actions.
    """

    def __init__(
        self,
        config: PlatformConfig | None = None,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize the sandbox platform orchestrator.

        Args:
            config: Platform configuration
            merkle_chain: Shared Merkle chain for audit trail
        """
        self.config = config or PlatformConfig()
        self.merkle_chain = merkle_chain or MerkleChain()
        self.status = PlatformStatus.INITIALIZING

        # Core components
        self.audit_logger = AuditChainLogger(
            logger_id=f"{self.config.platform_id}_audit",
            merkle_chain=self.merkle_chain,
        )

        self.isolated_executor = IsolatedSandboxExecutor(
            executor_id=f"{self.config.platform_id}_isolated",
            merkle_chain=self.merkle_chain,
            default_isolation=self.config.default_isolation,
        )

        self.verified_channel = MerkleVerifiedChannel(
            channel_id=f"{self.config.platform_id}_channel",
            merkle_chain=self.merkle_chain,
        )

        # Parallelism components
        if self.config.enable_parallelism:
            self.sharded_executor = ShardedSandboxExecutor(
                executor_id=f"{self.config.platform_id}_sharded",
                merkle_chain=self.merkle_chain,
            )
            self.batch_evaluator = BatchProposalEvaluator(
                evaluator_id=f"{self.config.platform_id}_batch",
                default_batch_size=self.config.batch_size,
                merkle_chain=self.merkle_chain,
            )
        else:
            self.sharded_executor = None
            self.batch_evaluator = None

        # Pipeline and queue
        self.pipeline = AsyncEvaluationPipeline(
            pipeline_id=f"{self.config.platform_id}_pipeline",
            num_workers=self.config.num_workers,
            merkle_chain=self.merkle_chain,
        )

        self.proposal_queue = ProposalQueue(
            queue_id=f"{self.config.platform_id}_queue",
            max_size=self.config.max_queue_size,
            num_workers=self.config.num_workers,
            merkle_chain=self.merkle_chain,
        )

        # Lightweight evaluation
        self.container_pool = EphemeralContainerPool(
            pool_id=f"{self.config.platform_id}_containers",
            merkle_chain=self.merkle_chain,
        )

        self.stub_registry = DeterministicStubRegistry(
            registry_id=f"{self.config.platform_id}_stubs",
            merkle_chain=self.merkle_chain,
        )

        self.incremental_evaluator = IncrementalStateEvaluator(
            evaluator_id=f"{self.config.platform_id}_incremental",
            merkle_chain=self.merkle_chain,
        )

        # Observability
        self.metrics_collector = PassiveMetricsCollector(
            collector_id=f"{self.config.platform_id}_metrics",
            merkle_chain=self.merkle_chain,
        )

        self.diagnostics_offloader = DiagnosticsOffloader(
            offloader_id=f"{self.config.platform_id}_diagnostics",
            merkle_chain=self.merkle_chain,
        )

        # Non-blocking integration
        self.dual_control_gateway = DualControlGateway(
            gateway_id=f"{self.config.platform_id}_gateway",
            merkle_chain=self.merkle_chain,
        )

        # Optimization
        if self.config.enable_lazy:
            self.lazy_evaluator = LazyEvaluator(
                evaluator_id=f"{self.config.platform_id}_lazy",
                merkle_chain=self.merkle_chain,
            )
        else:
            self.lazy_evaluator = None

        if self.config.enable_speculative:
            self.speculative_executor = SpeculativeExecutor(
                executor_id=f"{self.config.platform_id}_speculative",
                merkle_chain=self.merkle_chain,
            )
        else:
            self.speculative_executor = None

        self.resource_throttler = ResourceThrottler(
            throttler_id=f"{self.config.platform_id}_throttler",
            merkle_chain=self.merkle_chain,
        )

        # Quantum/High-performance (optional)
        if self.config.enable_quantum:
            self.quantum_sandbox = QuantumSandbox(
                sandbox_id=f"{self.config.platform_id}_quantum",
                merkle_chain=self.merkle_chain,
            )
            self.hybrid_executor = HybridExecutor(
                executor_id=f"{self.config.platform_id}_hybrid",
                quantum_sandbox=self.quantum_sandbox,
                merkle_chain=self.merkle_chain,
            )
            self.tensor_precomputer = TensorNetworkPrecomputer(
                precomputer_id=f"{self.config.platform_id}_tensor",
                merkle_chain=self.merkle_chain,
            )
        else:
            self.quantum_sandbox = None
            self.hybrid_executor = None
            self.tensor_precomputer = None

        # Counters and state
        self._proposal_counter = 0
        self._result_counter = 0
        self._lock = threading.RLock()

        # Initialize pipeline stages
        self._setup_pipeline()

        # Set processor for queue
        self.proposal_queue.set_processor(self._process_proposal)

        self.status = PlatformStatus.READY

        # Log initialization
        self.merkle_chain.add_event(
            "sandbox_platform_initialized",
            {
                "platform_id": self.config.platform_id,
                "config": self.config.to_dict(),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _setup_pipeline(self) -> None:
        """Setup pipeline stages."""
        # Stage 1: Validation
        self.pipeline.add_stage(
            "validation",
            self._validation_stage,
        )

        # Stage 2: Lazy evaluation check
        if self.config.enable_lazy:
            self.pipeline.add_stage(
                "lazy_check",
                self._lazy_check_stage,
            )

        # Stage 3: Execution
        self.pipeline.add_stage(
            "execution",
            self._execution_stage,
        )

    def _validation_stage(
        self,
        proposal: SandboxProposal,
        context: dict[str, Any],
    ) -> SandboxProposal | None:
        """Pipeline validation stage."""
        # Basic validation
        if not proposal.proposal_id:
            return None

        if not proposal.payload:
            return None

        return proposal

    def _lazy_check_stage(
        self,
        proposal: SandboxProposal,
        context: dict[str, Any],
    ) -> SandboxProposal | None:
        """Pipeline lazy evaluation check stage."""
        if self.lazy_evaluator is None:
            return proposal

        assessment = self.lazy_evaluator.assess_criticality(proposal)
        if not assessment.should_evaluate:
            return None  # Filter out non-critical

        return proposal

    def _execution_stage(
        self,
        proposal: SandboxProposal,
        context: dict[str, Any],
    ) -> SandboxProposal | None:
        """Pipeline execution stage."""
        # Execute in isolated container
        return proposal

    def start(self) -> None:
        """Start the sandbox platform."""
        if self.status == PlatformStatus.RUNNING:
            return

        # Start pipeline
        self.pipeline.start()

        # Start proposal queue
        self.proposal_queue.start()

        # Start diagnostics offloader
        self.diagnostics_offloader.start()

        self.status = PlatformStatus.RUNNING

        self.merkle_chain.add_event(
            "sandbox_platform_started",
            {
                "platform_id": self.config.platform_id,
            },
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Stop the sandbox platform.

        Args:
            timeout: Maximum time to wait for shutdown
        """
        if self.status == PlatformStatus.STOPPED:
            return

        self.status = PlatformStatus.SHUTTING_DOWN

        # Stop components
        self.pipeline.stop(timeout / 3)
        self.proposal_queue.stop(timeout / 3)
        self.diagnostics_offloader.stop(timeout / 3)

        # Clear resources
        self.container_pool.clear()

        self.status = PlatformStatus.STOPPED

        self.merkle_chain.add_event(
            "sandbox_platform_stopped",
            {
                "platform_id": self.config.platform_id,
            },
        )

    def submit_proposal(
        self,
        proposal_type: str,
        payload: dict[str, Any],
        priority: ProposalPriority = ProposalPriority.NORMAL,
        target_subsystems: list[str] | None = None,
        requires_approval: bool = False,
    ) -> SandboxProposal:
        """Submit a proposal for sandboxed evaluation.

        Args:
            proposal_type: Type of proposal
            payload: Proposal payload
            priority: Evaluation priority
            target_subsystems: Target subsystems
            requires_approval: Whether dual-control approval is required

        Returns:
            Created SandboxProposal
        """
        with self._lock:
            self._proposal_counter += 1
            proposal_id = f"prop_{self.config.platform_id}_{self._proposal_counter:010d}"

        proposal = SandboxProposal(
            proposal_id=proposal_id,
            source_id=self.config.platform_id,
            proposal_type=proposal_type,
            payload=payload,
            priority=priority,
            target_subsystems=target_subsystems or [],
        )

        # Enqueue for processing
        self.proposal_queue.enqueue(proposal, requires_approval=requires_approval)

        # Record metrics
        self.metrics_collector.record_operation(
            "proposal_submitted",
            count=1,
        )

        # Log submission
        self.audit_logger.log(
            "proposal_submitted",
            {
                "proposal_id": proposal_id,
                "proposal_type": proposal_type,
                "priority": priority.value,
            },
        )

        return proposal

    def _process_proposal(
        self,
        queued: Any,
    ) -> dict[str, Any]:
        """Process a queued proposal."""
        proposal = queued.proposal
        start_time = time.perf_counter()

        # Check resource availability
        self.resource_throttler.update_load()
        if self.resource_throttler.should_defer(0.2):
            return {"status": "deferred", "reason": "resource_constraints"}

        # Check for speculative result
        if self.speculative_executor:
            spec_result = self.speculative_executor.validate_result(proposal.proposal_id)
            if spec_result:
                return {
                    "status": "completed",
                    "speculative": True,
                    "result": spec_result.to_dict(),
                }

        # Execute in container
        result = self.container_pool.execute(
            lambda ctx: self._evaluate_proposal(proposal, ctx),
            {"proposal": proposal.to_dict()},
        )

        execution_time = (time.perf_counter() - start_time) * 1000

        # Record metrics
        self.metrics_collector.record_latency("proposal_processing", execution_time)
        self.metrics_collector.record_operation("proposal_processed")

        return {
            "status": "completed",
            "result": result,
            "execution_time_ms": execution_time,
        }

    def _evaluate_proposal(
        self,
        proposal: SandboxProposal,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        """Evaluate a proposal in isolated context."""
        # Create sandbox config
        config = SandboxConfig(
            sandbox_id=f"sandbox_{proposal.proposal_id}",
            isolation_level=self.config.default_isolation,
            execution_mode=ExecutionMode.SYNC,
        )

        # Create container
        container = self.isolated_executor.create_container(config.sandbox_id, config)

        try:
            # Execute in container
            def executor(prop: SandboxProposal, ctx: dict[str, Any]) -> dict[str, Any]:
                return {
                    "evaluated": True,
                    "proposal_id": prop.proposal_id,
                    "payload_key_count": len(prop.payload.keys()),
                }

            result = self.isolated_executor.execute(
                container,
                proposal,
                executor,
                context,
                async_mode=False,
            )

            if result:
                return result.to_dict()
            return {"status": "async_submitted"}

        finally:
            # Clean up container
            self.isolated_executor.destroy_container(container.container_id)

    def evaluate_batch(
        self,
        proposals: list[SandboxProposal],
    ) -> list[SandboxEvaluationResult]:
        """Evaluate a batch of proposals.

        Args:
            proposals: Proposals to evaluate

        Returns:
            List of evaluation results
        """
        if self.batch_evaluator is None:
            # Fall back to sequential
            results = []
            for proposal in proposals:
                self.submit_proposal(
                    proposal.proposal_type,
                    proposal.payload,
                    proposal.priority,
                )
            return results

        # Submit to batch evaluator
        for proposal in proposals:
            self.batch_evaluator.submit(proposal)

        # Flush and evaluate
        self.batch_evaluator.flush()
        batch_results = self.batch_evaluator.evaluate_all_ready(
            executor=lambda p: self._create_evaluation_result(p),
        )

        results = []
        for br in batch_results:
            results.extend(br.results)

        return results

    def _create_evaluation_result(
        self,
        proposal: SandboxProposal,
    ) -> SandboxEvaluationResult:
        """Create evaluation result for a proposal."""
        self._result_counter += 1
        result_id = f"result_{self.config.platform_id}_{self._result_counter:08d}"

        return SandboxEvaluationResult(
            result_id=result_id,
            proposal_id=proposal.proposal_id,
            sandbox_id=self.config.platform_id,
            success=True,
            outcome="evaluated",
            fidelity_score=0.95,
            merkle_proof=self.merkle_chain.get_chain_proof(),
        )

    def request_approval(
        self,
        proposal: SandboxProposal,
        description: str,
    ) -> str:
        """Request dual-control approval for a proposal.

        Args:
            proposal: Proposal requiring approval
            description: Description for approvers

        Returns:
            Request ID
        """
        request = self.dual_control_gateway.create_request(
            resource_id=proposal.proposal_id,
            resource_type=proposal.proposal_type,
            description=description,
        )

        return request.request_id

    def approve(
        self,
        request_id: str,
        approver_id: str,
        approved: bool,
        reason: str = "",
    ) -> bool:
        """Add approval decision.

        Args:
            request_id: Request to approve
            approver_id: Approver identifier
            approved: Approval decision
            reason: Reason for decision

        Returns:
            True if approval was recorded
        """
        decision = "approve" if approved else "reject"
        record = self.dual_control_gateway.add_approval(
            request_id,
            approver_id,
            decision,
            reason,
        )
        return record is not None

    def get_metrics_snapshot(self) -> MetricsSnapshot:
        """Get current metrics snapshot.

        Returns:
            MetricsSnapshot with current metrics
        """
        self._result_counter += 1
        snapshot_id = f"snapshot_{self.config.platform_id}_{self._result_counter:08d}"

        queue_stats = self.proposal_queue.get_queue_stats()
        throttle_state = self.resource_throttler.get_current_state()

        return MetricsSnapshot(
            snapshot_id=snapshot_id,
            sandbox_id=self.config.platform_id,
            queue_depth=queue_stats.get("total_size", 0),
            cpu_utilization=throttle_state.resource_multiplier if throttle_state else 1.0,
        )

    def get_platform_stats(self) -> dict[str, Any]:
        """Get comprehensive platform statistics.

        Returns:
            Dictionary of all platform statistics
        """
        stats = {
            "platform_id": self.config.platform_id,
            "status": self.status.value,
            "config": self.config.to_dict(),
            "isolated_executor": self.isolated_executor.get_executor_stats(),
            "pipeline": self.pipeline.get_pipeline_stats(),
            "proposal_queue": self.proposal_queue.get_queue_stats(),
            "metrics": self.metrics_collector.get_all_metrics(),
            "dual_control": self.dual_control_gateway.get_gateway_stats(),
            "resource_throttler": self.resource_throttler.get_throttler_stats(),
            "container_pool": self.container_pool.get_pool_stats(),
            "stub_registry": self.stub_registry.get_registry_stats(),
            "incremental_evaluator": self.incremental_evaluator.get_evaluator_stats(),
            "diagnostics": self.diagnostics_offloader.get_offloader_stats(),
            "merkle_chain_valid": self.merkle_chain.verify_integrity(),
            "merkle_chain_length": len(self.merkle_chain.chain),
        }

        if self.config.enable_parallelism and self.sharded_executor:
            stats["sharded_executor"] = self.sharded_executor.get_executor_stats()
            if self.batch_evaluator:
                stats["batch_evaluator"] = self.batch_evaluator.get_evaluator_stats()

        if self.config.enable_lazy and self.lazy_evaluator:
            stats["lazy_evaluator"] = self.lazy_evaluator.get_evaluator_stats()

        if self.config.enable_speculative and self.speculative_executor:
            stats["speculative_executor"] = self.speculative_executor.get_executor_stats()

        if self.config.enable_quantum:
            if self.quantum_sandbox:
                stats["quantum_sandbox"] = self.quantum_sandbox.get_sandbox_stats()
            if self.hybrid_executor:
                stats["hybrid_executor"] = self.hybrid_executor.get_executor_stats()
            if self.tensor_precomputer:
                stats["tensor_precomputer"] = self.tensor_precomputer.get_precomputer_stats()

        return stats

    def verify_provenance(self) -> bool:
        """Verify complete provenance chain integrity.

        Returns:
            True if all chains are valid
        """
        return self.merkle_chain.verify_integrity()
