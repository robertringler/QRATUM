"""Sharded Executor for Parallel Sandbox Workloads.

Distributes sandbox workloads across sharded nodes for parallel
evaluation. Dedicated hardware (CPU/GPU/quantum) prevents interference.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain
from qratum_asi.sandbox_platform.types import (
    ProposalPriority,
    ResourceType,
    SandboxEvaluationResult,
    SandboxProposal,
)


class NodeStatus(Enum):
    """Status of a shard node."""

    OFFLINE = "offline"
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    DRAINING = "draining"
    FAILED = "failed"


class ShardingStrategy(Enum):
    """Strategy for workload sharding."""

    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    HASH_BASED = "hash_based"
    PRIORITY_BASED = "priority_based"
    RESOURCE_AFFINITY = "resource_affinity"


@dataclass
class NodeAllocation:
    """Resource allocation for a shard node.

    Attributes:
        node_id: Unique node identifier
        resource_type: Primary resource type
        capacity: Total capacity units
        allocated: Currently allocated units
        max_concurrent: Maximum concurrent tasks
        current_tasks: Current number of tasks
    """

    node_id: str
    resource_type: ResourceType
    capacity: float = 100.0
    allocated: float = 0.0
    max_concurrent: int = 10
    current_tasks: int = 0
    status: NodeStatus = NodeStatus.OFFLINE

    @property
    def utilization(self) -> float:
        """Get current utilization (0-1)."""
        return self.allocated / self.capacity if self.capacity > 0 else 0.0

    @property
    def available_capacity(self) -> float:
        """Get available capacity."""
        return self.capacity - self.allocated

    @property
    def can_accept_task(self) -> bool:
        """Check if node can accept more tasks."""
        return (
            self.status == NodeStatus.READY
            and self.current_tasks < self.max_concurrent
            and self.available_capacity > 0
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize node allocation."""
        return {
            "node_id": self.node_id,
            "resource_type": self.resource_type.value,
            "capacity": self.capacity,
            "allocated": self.allocated,
            "utilization": self.utilization,
            "max_concurrent": self.max_concurrent,
            "current_tasks": self.current_tasks,
            "status": self.status.value,
        }


@dataclass
class WorkloadShard:
    """A shard of workload assigned to a node.

    Attributes:
        shard_id: Unique shard identifier
        node_id: Assigned node
        proposals: Proposals in this shard
        resource_requirement: Required resources
        priority: Shard priority
        created_at: Creation timestamp
    """

    shard_id: str
    node_id: str
    proposals: list[SandboxProposal]
    resource_requirement: float = 10.0
    priority: ProposalPriority = ProposalPriority.NORMAL
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    completed_at: str | None = None
    results: list[SandboxEvaluationResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize workload shard."""
        return {
            "shard_id": self.shard_id,
            "node_id": self.node_id,
            "proposal_count": len(self.proposals),
            "resource_requirement": self.resource_requirement,
            "priority": self.priority.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "result_count": len(self.results),
        }


class ShardedSandboxExecutor:
    """Executor that distributes workloads across sharded nodes.

    Provides:
    - Workload distribution across multiple nodes
    - Load balancing and resource affinity
    - Dedicated hardware support (CPU/GPU/quantum)
    - Fault tolerance and failover
    """

    def __init__(
        self,
        executor_id: str = "sharded",
        sharding_strategy: ShardingStrategy = ShardingStrategy.LOAD_BALANCED,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize sharded executor.

        Args:
            executor_id: Unique executor identifier
            sharding_strategy: Strategy for workload distribution
            merkle_chain: Merkle chain for audit trail
        """
        self.executor_id = executor_id
        self.sharding_strategy = sharding_strategy
        self.merkle_chain = merkle_chain or MerkleChain()

        # Node management
        self.nodes: dict[str, NodeAllocation] = {}
        self.shards: dict[str, WorkloadShard] = {}
        self._node_counter = 0
        self._shard_counter = 0
        self._lock = threading.RLock()

        # Execution tracking
        self._pending_shards: list[str] = []
        self._executing_shards: set[str] = set()
        self._completed_shards: list[str] = []

        # Node executor functions
        self._node_executors: dict[
            str, Callable[[WorkloadShard], list[SandboxEvaluationResult]]
        ] = {}

        # Initialize default nodes
        self._initialize_default_nodes()

        # Log initialization
        self.merkle_chain.add_event(
            "sharded_executor_initialized",
            {
                "executor_id": executor_id,
                "sharding_strategy": sharding_strategy.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def _initialize_default_nodes(self) -> None:
        """Initialize default compute nodes."""
        # CPU nodes
        for i in range(4):
            self.add_node(
                resource_type=ResourceType.CPU,
                capacity=100.0,
                max_concurrent=10,
            )

        # GPU node (optional)
        self.add_node(
            resource_type=ResourceType.GPU,
            capacity=50.0,
            max_concurrent=4,
        )

        # Quantum node (optional/simulated)
        self.add_node(
            resource_type=ResourceType.QUANTUM,
            capacity=10.0,
            max_concurrent=1,
        )

    def add_node(
        self,
        resource_type: ResourceType = ResourceType.CPU,
        capacity: float = 100.0,
        max_concurrent: int = 10,
    ) -> NodeAllocation:
        """Add a compute node.

        Args:
            resource_type: Type of resource
            capacity: Node capacity
            max_concurrent: Maximum concurrent tasks

        Returns:
            Created NodeAllocation
        """
        with self._lock:
            self._node_counter += 1
            node_id = f"node_{self.executor_id}_{resource_type.value}_{self._node_counter:04d}"

            node = NodeAllocation(
                node_id=node_id,
                resource_type=resource_type,
                capacity=capacity,
                max_concurrent=max_concurrent,
                status=NodeStatus.READY,
            )

            self.nodes[node_id] = node

            # Log node addition
            self.merkle_chain.add_event(
                "node_added",
                {
                    "node_id": node_id,
                    "resource_type": resource_type.value,
                    "capacity": capacity,
                },
            )

            return node

    def remove_node(self, node_id: str) -> bool:
        """Remove a compute node.

        Args:
            node_id: Node to remove

        Returns:
            True if removed successfully
        """
        with self._lock:
            if node_id not in self.nodes:
                return False

            node = self.nodes[node_id]
            if node.current_tasks > 0:
                node.status = NodeStatus.DRAINING
                return False

            del self.nodes[node_id]
            return True

    def create_shard(
        self,
        proposals: list[SandboxProposal],
        priority: ProposalPriority = ProposalPriority.NORMAL,
        resource_requirement: float = 10.0,
        target_resource: ResourceType | None = None,
    ) -> WorkloadShard:
        """Create a workload shard from proposals.

        Args:
            proposals: Proposals to include in shard
            priority: Shard priority
            resource_requirement: Resource requirement
            target_resource: Preferred resource type

        Returns:
            Created WorkloadShard
        """
        with self._lock:
            self._shard_counter += 1
            shard_id = f"shard_{self.executor_id}_{self._shard_counter:06d}"

            # Select node based on strategy
            node = self._select_node(priority, resource_requirement, target_resource)
            if not node:
                raise RuntimeError("No available nodes for shard execution")

            shard = WorkloadShard(
                shard_id=shard_id,
                node_id=node.node_id,
                proposals=proposals,
                resource_requirement=resource_requirement,
                priority=priority,
            )

            self.shards[shard_id] = shard
            self._pending_shards.append(shard_id)

            # Log shard creation
            self.merkle_chain.add_event(
                "shard_created",
                {
                    "shard_id": shard_id,
                    "node_id": node.node_id,
                    "proposal_count": len(proposals),
                    "priority": priority.value,
                },
            )

            return shard

    def _select_node(
        self,
        priority: ProposalPriority,
        resource_requirement: float,
        target_resource: ResourceType | None,
    ) -> NodeAllocation | None:
        """Select best node for workload based on strategy."""
        available_nodes = [
            n
            for n in self.nodes.values()
            if n.can_accept_task and n.available_capacity >= resource_requirement
        ]

        if target_resource:
            resource_nodes = [n for n in available_nodes if n.resource_type == target_resource]
            if resource_nodes:
                available_nodes = resource_nodes

        if not available_nodes:
            return None

        if self.sharding_strategy == ShardingStrategy.LOAD_BALANCED:
            # Select least loaded node
            return min(available_nodes, key=lambda n: n.utilization)

        elif self.sharding_strategy == ShardingStrategy.ROUND_ROBIN:
            # Simple round-robin
            return available_nodes[self._shard_counter % len(available_nodes)]

        elif self.sharding_strategy == ShardingStrategy.PRIORITY_BASED:
            # High priority tasks go to less loaded nodes
            if priority in (ProposalPriority.CRITICAL, ProposalPriority.HIGH):
                return min(available_nodes, key=lambda n: n.utilization)
            return available_nodes[0]

        elif self.sharding_strategy == ShardingStrategy.RESOURCE_AFFINITY:
            # Prefer matching resource type
            return available_nodes[0]

        return available_nodes[0]

    def execute_shard(
        self,
        shard_id: str,
        executor_func: Callable[[SandboxProposal], SandboxEvaluationResult],
    ) -> list[SandboxEvaluationResult]:
        """Execute a workload shard.

        Args:
            shard_id: Shard to execute
            executor_func: Function to execute each proposal

        Returns:
            List of evaluation results
        """
        with self._lock:
            if shard_id not in self.shards:
                raise ValueError(f"Shard {shard_id} not found")

            shard = self.shards[shard_id]
            node = self.nodes.get(shard.node_id)

            if not node or not node.can_accept_task:
                raise RuntimeError(f"Node {shard.node_id} not available")

            # Allocate resources
            node.allocated += shard.resource_requirement
            node.current_tasks += 1
            node.status = NodeStatus.BUSY

            self._executing_shards.add(shard_id)
            if shard_id in self._pending_shards:
                self._pending_shards.remove(shard_id)

        shard.started_at = datetime.now(timezone.utc).isoformat()

        # Log execution start
        self.merkle_chain.add_event(
            "shard_execution_started",
            {
                "shard_id": shard_id,
                "node_id": shard.node_id,
            },
        )

        try:
            results: list[SandboxEvaluationResult] = []
            for proposal in shard.proposals:
                result = executor_func(proposal)
                results.append(result)
                shard.results.append(result)

            shard.completed_at = datetime.now(timezone.utc).isoformat()

            # Log execution completion
            self.merkle_chain.add_event(
                "shard_execution_completed",
                {
                    "shard_id": shard_id,
                    "result_count": len(results),
                },
            )

            return results

        finally:
            with self._lock:
                # Release resources
                node.allocated = max(0, node.allocated - shard.resource_requirement)
                node.current_tasks = max(0, node.current_tasks - 1)
                if node.current_tasks == 0:
                    node.status = NodeStatus.READY

                self._executing_shards.discard(shard_id)
                self._completed_shards.append(shard_id)

    def execute_parallel(
        self,
        proposals: list[SandboxProposal],
        executor_func: Callable[[SandboxProposal], SandboxEvaluationResult],
        shard_size: int = 10,
        max_parallel: int = 4,
    ) -> list[SandboxEvaluationResult]:
        """Execute proposals in parallel across shards.

        Args:
            proposals: Proposals to execute
            executor_func: Function to execute each proposal
            shard_size: Number of proposals per shard
            max_parallel: Maximum parallel shards

        Returns:
            List of all evaluation results
        """
        # Create shards
        shards: list[WorkloadShard] = []
        for i in range(0, len(proposals), shard_size):
            batch = proposals[i : i + shard_size]
            shard = self.create_shard(batch)
            shards.append(shard)

        # Execute shards in parallel (simulated with threads)
        results: list[SandboxEvaluationResult] = []
        threads: list[threading.Thread] = []
        results_lock = threading.Lock()

        def execute_and_collect(shard: WorkloadShard) -> None:
            shard_results = self.execute_shard(shard.shard_id, executor_func)
            with results_lock:
                results.extend(shard_results)

        # Process in batches of max_parallel
        for i in range(0, len(shards), max_parallel):
            batch_shards = shards[i : i + max_parallel]
            threads = []

            for shard in batch_shards:
                thread = threading.Thread(
                    target=execute_and_collect,
                    args=(shard,),
                    daemon=True,
                )
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()

        return results

    def get_node_stats(self) -> dict[str, Any]:
        """Get statistics for all nodes."""
        node_stats = {node_id: node.to_dict() for node_id, node in self.nodes.items()}

        total_capacity = sum(n.capacity for n in self.nodes.values())
        total_allocated = sum(n.allocated for n in self.nodes.values())

        return {
            "executor_id": self.executor_id,
            "sharding_strategy": self.sharding_strategy.value,
            "total_nodes": len(self.nodes),
            "total_capacity": total_capacity,
            "total_allocated": total_allocated,
            "overall_utilization": total_allocated / total_capacity if total_capacity > 0 else 0,
            "nodes": node_stats,
        }

    def get_shard_stats(self) -> dict[str, Any]:
        """Get statistics for shards."""
        return {
            "total_shards": len(self.shards),
            "pending_shards": len(self._pending_shards),
            "executing_shards": len(self._executing_shards),
            "completed_shards": len(self._completed_shards),
        }

    def get_executor_stats(self) -> dict[str, Any]:
        """Get comprehensive executor statistics."""
        return {
            **self.get_node_stats(),
            **self.get_shard_stats(),
            "merkle_chain_valid": self.merkle_chain.verify_integrity(),
        }
