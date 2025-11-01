"""Neural kernel fusion engine with learned cost models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Set, Tuple


@dataclass
class KernelNode:
    """Represents a kernel operation in the fusion graph."""
    name: str
    op_type: str
    cost: float = 1.0
    dependencies: List[KernelNode] = field(default_factory=list)
    consumers: List[KernelNode] = field(default_factory=list)
    fusable: bool = True
    
    def __hash__(self) -> int:
        return id(self)


@dataclass
class KernelGraph:
    """Graph representation of kernel operations for fusion analysis."""
    nodes: List[KernelNode] = field(default_factory=list)
    
    def add_node(self, name: str, op_type: str, dependencies: List[KernelNode] | None = None) -> KernelNode:
        """Add a kernel node to the graph."""
        node = KernelNode(name=name, op_type=op_type, dependencies=dependencies or [])
        for dep in node.dependencies:
            dep.consumers.append(node)
        self.nodes.append(node)
        return node
        
    def get_fusable_groups(self) -> List[List[KernelNode]]:
        """Identify groups of kernels that can be fused together."""
        groups: List[List[KernelNode]] = []
        visited: Set[KernelNode] = set()
        
        for node in self.nodes:
            if node in visited or not node.fusable:
                continue
                
            # Start a new fusion group
            group = [node]
            visited.add(node)
            
            # Try to extend the group with consumers
            for consumer in node.consumers:
                if consumer.fusable and len(consumer.dependencies) == 1:
                    group.append(consumer)
                    visited.add(consumer)
                    
            if len(group) > 1:
                groups.append(group)
                
        return groups


class FusionEngine:
    """Neural kernel fusion engine with learned cost models."""
    
    def __init__(self) -> None:
        self._cost_model_weights: dict[str, float] = {
            "matmul": 100.0,
            "conv2d": 150.0,
            "add": 1.0,
            "mul": 1.0,
            "relu": 0.5,
            "softmax": 10.0,
        }
        self._fusion_threshold = 5.0
        
    def estimate_cost(self, op_type: str, size: int = 1) -> float:
        """Estimate execution cost for an operation using learned model."""
        base_cost = self._cost_model_weights.get(op_type, 1.0)
        return base_cost * size
        
    def should_fuse(self, nodes: List[KernelNode]) -> bool:
        """Decide whether to fuse a group of kernels based on cost model."""
        if len(nodes) < 2:
            return False
            
        # Compute total cost without fusion
        unfused_cost = sum(self.estimate_cost(n.op_type) for n in nodes)
        
        # Estimate fused cost (includes fusion overhead but saves memory transfers)
        memory_transfer_savings = (len(nodes) - 1) * 2.0
        fusion_overhead = 1.0
        fused_cost = unfused_cost * 0.8 + fusion_overhead - memory_transfer_savings
        
        return fused_cost < unfused_cost and unfused_cost > self._fusion_threshold
        
    def optimize_graph(self, graph: KernelGraph) -> List[Tuple[str, List[KernelNode]]]:
        """Optimize kernel graph by identifying fusion opportunities."""
        fusable_groups = graph.get_fusable_groups()
        optimized_groups = []
        
        for group in fusable_groups:
            if self.should_fuse(group):
                group_name = "_".join(n.name for n in group)
                optimized_groups.append((f"fused_{group_name}", group))
                
        return optimized_groups
        
    def update_cost_model(self, op_type: str, measured_cost: float) -> None:
        """Update learned cost model with measured execution data."""
        if op_type in self._cost_model_weights:
            # Exponential moving average for online learning
            alpha = 0.1
            self._cost_model_weights[op_type] = (
                alpha * measured_cost + (1 - alpha) * self._cost_model_weights[op_type]
            )
        else:
            self._cost_model_weights[op_type] = measured_cost
