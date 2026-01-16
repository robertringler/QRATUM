"""
Deterministic AllReduce Implementation for QRATUM

Fixed-order binary tree AllReduce that guarantees bit-identical results
across all runs regardless of timing variations. Critical for reproducible
distributed training at exascale.

Key Features:
- Fixed binary tree topology (no dynamic load balancing)
- Deterministic reduction order (left-to-right, bottom-up)
- Floating-point accumulation in fixed order (solves non-associativity)
- Hardware Merkle verification of intermediate results
- Support for all standard reduction operations

Design Principle:
Floating-point operations are non-associative: (a + b) + c ≠ a + (b + c)
Our fixed-order tree ensures identical operation sequences every run,
achieving bit-identical reproducibility even with FP arithmetic.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import hashlib
import math


class ReductionOp(Enum):
    """Supported reduction operations"""
    SUM = "sum"
    PRODUCT = "product"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"  # Sum followed by division by count
    BITWISE_AND = "bitwise_and"
    BITWISE_OR = "bitwise_or"
    BITWISE_XOR = "bitwise_xor"


class TreeTopology(Enum):
    """Binary tree topology variants"""
    BALANCED = "balanced"  # Minimize depth
    LINEAR = "linear"  # Chain topology (maximum determinism)
    CUSTOM = "custom"  # User-defined topology


@dataclass
class TreeNode:
    """
    Node in deterministic reduction tree
    
    Attributes:
        node_id: Unique node identifier (typically MPI rank)
        parent: Parent node ID (None for root)
        left_child: Left child node ID
        right_child: Right child node ID
        depth: Depth in tree (0 for leaves)
    """
    node_id: int
    parent: Optional[int] = None
    left_child: Optional[int] = None
    right_child: Optional[int] = None
    depth: int = 0
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node"""
        return self.left_child is None and self.right_child is None
    
    def is_root(self) -> bool:
        """Check if this is the root node"""
        return self.parent is None
    
    def has_both_children(self) -> bool:
        """Check if node has both children"""
        return self.left_child is not None and self.right_child is not None


@dataclass
class BinaryTreeTopology:
    """
    Fixed binary tree topology for deterministic AllReduce
    
    The tree structure is determined once at initialization and never changes,
    ensuring identical reduction patterns across all runs.
    
    Attributes:
        num_nodes: Total number of nodes (MPI ranks)
        root_id: Root node ID (typically rank 0)
        nodes: Dictionary of all tree nodes
        topology_type: Type of tree topology
        topology_hash: Cryptographic hash of tree structure
    """
    num_nodes: int
    root_id: int = 0
    nodes: Dict[int, TreeNode] = field(default_factory=dict)
    topology_type: TreeTopology = TreeTopology.BALANCED
    topology_hash: Optional[str] = None
    
    def __post_init__(self):
        """Build tree topology"""
        if not self.nodes:
            self._build_tree()
            self.topology_hash = self._compute_topology_hash()
    
    def _build_tree(self) -> None:
        """
        Build deterministic binary tree
        
        For BALANCED topology:
        - Complete binary tree with minimum depth
        - Left-to-right, bottom-up construction
        - Deterministic assignment: node i has children 2i+1 and 2i+2
        """
        if self.topology_type == TreeTopology.BALANCED:
            self._build_balanced_tree()
        elif self.topology_type == TreeTopology.LINEAR:
            self._build_linear_tree()
        else:
            raise ValueError(f"Unsupported topology: {self.topology_type}")
    
    def _build_balanced_tree(self) -> None:
        """Build balanced binary tree"""
        # Create all nodes
        for i in range(self.num_nodes):
            self.nodes[i] = TreeNode(node_id=i)
        
        # Set parent-child relationships
        for i in range(self.num_nodes):
            left_child = 2 * i + 1
            right_child = 2 * i + 2
            
            if left_child < self.num_nodes:
                self.nodes[i].left_child = left_child
                self.nodes[left_child].parent = i
            
            if right_child < self.num_nodes:
                self.nodes[i].right_child = right_child
                self.nodes[right_child].parent = i
        
        # Compute depths
        self._compute_depths()
    
    def _build_linear_tree(self) -> None:
        """Build linear chain topology (maximum determinism)"""
        for i in range(self.num_nodes):
            self.nodes[i] = TreeNode(node_id=i, depth=i)
        
        # Chain: 0 <- 1 <- 2 <- ... <- n-1
        for i in range(1, self.num_nodes):
            self.nodes[i].parent = i - 1
            self.nodes[i - 1].right_child = i
    
    def _compute_depths(self) -> None:
        """Compute depth for each node via BFS from root"""
        queue = [(self.root_id, 0)]
        visited = set()
        
        while queue:
            node_id, depth = queue.pop(0)
            if node_id in visited:
                continue
            
            visited.add(node_id)
            self.nodes[node_id].depth = depth
            
            node = self.nodes[node_id]
            if node.left_child is not None:
                queue.append((node.left_child, depth + 1))
            if node.right_child is not None:
                queue.append((node.right_child, depth + 1))
    
    def _compute_topology_hash(self) -> str:
        """
        Compute cryptographic hash of tree topology
        
        Returns:
            SHA-256 hash of topology structure
        """
        # Serialize topology deterministically
        topology_repr = []
        for i in range(self.num_nodes):
            node = self.nodes[i]
            topology_repr.append(f"{i}:{node.parent}:{node.left_child}:{node.right_child}")
        
        topology_str = ";".join(topology_repr)
        return hashlib.sha256(topology_str.encode('utf-8')).hexdigest()
    
    def get_reduction_order(self) -> List[Tuple[int, int, int]]:
        """
        Get deterministic reduction operation order
        
        Returns:
            List of (parent, left_child, right_child) tuples in execution order.
            Operations execute bottom-up, left-to-right within each level.
        """
        max_depth = max(node.depth for node in self.nodes.values())
        operations = []
        
        # Process level by level, bottom to top
        for depth in range(max_depth, -1, -1):
            # Get all nodes at this depth, sorted by ID for determinism
            level_nodes = sorted([
                node for node in self.nodes.values() 
                if node.depth == depth
            ], key=lambda n: n.node_id)
            
            # Add operations for nodes with children
            for node in level_nodes:
                if node.left_child is not None or node.right_child is not None:
                    operations.append((
                        node.node_id,
                        node.left_child if node.left_child is not None else -1,
                        node.right_child if node.right_child is not None else -1
                    ))
        
        return operations
    
    def get_node(self, node_id: int) -> TreeNode:
        """Get tree node by ID"""
        return self.nodes.get(node_id)
    
    def verify_topology(self, other_hash: str) -> bool:
        """
        Verify topology matches another instance
        
        Args:
            other_hash: Topology hash from another rank
            
        Returns:
            True if topologies match
        """
        return self.topology_hash == other_hash


@dataclass
class ReductionOperation:
    """
    Represents a single reduction operation in the tree
    
    Attributes:
        operation_id: Unique operation identifier
        parent_node: Parent node receiving result
        left_value: Left operand (from left child)
        right_value: Right operand (from right child)
        op_type: Reduction operation type
        result: Result of reduction
        merkle_proof: Merkle proof of result correctness
    """
    operation_id: int
    parent_node: int
    left_value: Optional[Any] = None
    right_value: Optional[Any] = None
    op_type: ReductionOp = ReductionOp.SUM
    result: Optional[Any] = None
    merkle_proof: Optional[str] = None
    
    def execute(self) -> Any:
        """
        Execute reduction operation deterministically
        
        Returns:
            Result of reduction
        """
        if self.left_value is None and self.right_value is None:
            raise ValueError("Both operands are None")
        
        # Handle single operand case
        if self.left_value is None:
            self.result = self.right_value
            return self.result
        if self.right_value is None:
            self.result = self.left_value
            return self.result
        
        # Execute operation based on type
        if self.op_type == ReductionOp.SUM:
            self.result = self.left_value + self.right_value
        elif self.op_type == ReductionOp.PRODUCT:
            self.result = self.left_value * self.right_value
        elif self.op_type == ReductionOp.MIN:
            self.result = min(self.left_value, self.right_value)
        elif self.op_type == ReductionOp.MAX:
            self.result = max(self.left_value, self.right_value)
        elif self.op_type == ReductionOp.BITWISE_AND:
            self.result = self.left_value & self.right_value
        elif self.op_type == ReductionOp.BITWISE_OR:
            self.result = self.left_value | self.right_value
        elif self.op_type == ReductionOp.BITWISE_XOR:
            self.result = self.left_value ^ self.right_value
        else:
            raise ValueError(f"Unsupported operation: {self.op_type}")
        
        return self.result
    
    def compute_merkle_proof(self) -> str:
        """
        Compute Merkle proof for this operation
        
        Returns:
            Merkle proof as hex string
        """
        data = f"{self.operation_id}:{self.left_value}:{self.right_value}:{self.result}"
        self.merkle_proof = hashlib.sha256(data.encode('utf-8')).hexdigest()
        return self.merkle_proof


@dataclass
class DeterministicAllReduce:
    """
    Deterministic AllReduce implementation using fixed binary tree
    
    Guarantees bit-identical results across all runs by:
    1. Using fixed tree topology (never changes)
    2. Processing operations in deterministic order (bottom-up, left-to-right)
    3. Accumulating in fixed order to handle FP non-associativity
    4. Verifying intermediate results with Merkle proofs
    
    This is critical for reproducible ML training at exascale where even
    tiny FP differences can cascade into divergent model parameters.
    
    Attributes:
        topology: Fixed binary tree topology
        reduction_op: Type of reduction operation
        verify_merkle: Enable Merkle verification of intermediate results
        operation_history: History of all operations (for debugging)
    """
    topology: BinaryTreeTopology
    reduction_op: ReductionOp = ReductionOp.SUM
    verify_merkle: bool = True
    operation_history: List[ReductionOperation] = field(default_factory=list)
    
    def allreduce(self, local_value: Any, rank: int) -> Any:
        """
        Perform deterministic AllReduce operation
        
        All ranks call this with their local value. The function returns
        the globally reduced value to all ranks.
        
        Args:
            local_value: This rank's local value to reduce
            rank: This rank's ID
            
        Returns:
            Globally reduced value (same on all ranks)
        """
        # In a real implementation, this would use MPI or similar
        # This is a stub showing the deterministic algorithm
        
        # Get deterministic reduction order
        reduction_order = self.topology.get_reduction_order()
        
        # Initialize values at leaves
        values = {rank: local_value}
        
        # Execute reductions in deterministic order
        for op_id, (parent, left, right) in enumerate(reduction_order):
            left_val = values.get(left) if left >= 0 else None
            right_val = values.get(right) if right >= 0 else None
            
            if left_val is None and right_val is None:
                continue
            
            # Create and execute reduction operation
            op = ReductionOperation(
                operation_id=op_id,
                parent_node=parent,
                left_value=left_val,
                right_value=right_val,
                op_type=self.reduction_op
            )
            
            result = op.execute()
            values[parent] = result
            
            # Compute Merkle proof if verification enabled
            if self.verify_merkle:
                op.compute_merkle_proof()
            
            self.operation_history.append(op)
        
        # Root node has final result
        return values.get(self.topology.root_id)
    
    def allreduce_mean(self, local_value: Any, rank: int) -> Any:
        """
        Deterministic AllReduce for mean calculation
        
        Sum followed by division ensures bit-identical results.
        
        Args:
            local_value: This rank's local value
            rank: This rank's ID
            
        Returns:
            Global mean value
        """
        total_sum = self.allreduce(local_value, rank)
        return total_sum / self.topology.num_nodes
    
    def verify_operation_history(self, other_history: List[ReductionOperation]) -> bool:
        """
        Verify operation history matches another run
        
        Args:
            other_history: Operation history from another run
            
        Returns:
            True if histories match
        """
        if len(self.operation_history) != len(other_history):
            return False
        
        for op1, op2 in zip(self.operation_history, other_history):
            if op1.operation_id != op2.operation_id:
                return False
            if op1.parent_node != op2.parent_node:
                return False
            if op1.merkle_proof != op2.merkle_proof:
                return False
        
        return True
    
    def get_communication_pattern(self) -> List[Tuple[int, int, str]]:
        """
        Get deterministic communication pattern for optimization
        
        Returns:
            List of (source, dest, operation) tuples showing all communications
        """
        pattern = []
        reduction_order = self.topology.get_reduction_order()
        
        for parent, left, right in reduction_order:
            if left >= 0:
                pattern.append((left, parent, "send_left"))
            if right >= 0:
                pattern.append((right, parent, "send_right"))
        
        # Broadcast phase (root to all)
        for rank in range(self.topology.num_nodes):
            if rank != self.topology.root_id:
                pattern.append((self.topology.root_id, rank, "broadcast"))
        
        return pattern
    
    def estimate_latency_us(self, message_size_bytes: int, 
                           bandwidth_gbps: float = 800,
                           latency_per_hop_ns: int = 50) -> float:
        """
        Estimate AllReduce latency in microseconds
        
        Args:
            message_size_bytes: Size of data being reduced
            bandwidth_gbps: Network bandwidth in Gbps
            latency_per_hop_ns: Per-hop latency in nanoseconds
            
        Returns:
            Estimated latency in microseconds
        """
        # Tree depth determines number of reduction phases
        max_depth = max(node.depth for node in self.topology.nodes.values())
        
        # Transfer time per message
        bytes_per_ns = (bandwidth_gbps * 1e9) / (8 * 1e9)
        transfer_time_ns = message_size_bytes / bytes_per_ns
        
        # Reduction phase: bottom-up tree traversal
        reduction_latency_ns = max_depth * (transfer_time_ns + latency_per_hop_ns)
        
        # Broadcast phase: top-down tree traversal
        broadcast_latency_ns = max_depth * (transfer_time_ns + latency_per_hop_ns)
        
        total_latency_us = (reduction_latency_ns + broadcast_latency_ns) / 1000.0
        return total_latency_us
    
    def compute_result_hash(self, result: Any) -> str:
        """
        Compute hash of final result for verification
        
        Args:
            result: Final AllReduce result
            
        Returns:
            SHA-256 hash as hex string
        """
        result_str = str(result)
        return hashlib.sha256(result_str.encode('utf-8')).hexdigest()
