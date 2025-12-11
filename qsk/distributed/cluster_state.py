"""Cluster state abstraction."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class NodeDescriptor:
    node_id: str
    trust: int = 100


class ClusterState:
    def __init__(self) -> None:
        self.nodes: dict[str, NodeDescriptor] = {}

    def add_node(self, node_id: str, trust: int = 100) -> None:
        self.nodes[node_id] = NodeDescriptor(node_id=node_id, trust=trust)

    def remove_node(self, node_id: str) -> None:
        self.nodes.pop(node_id, None)

    def describe(self) -> list[NodeDescriptor]:
        return [self.nodes[k] for k in sorted(self.nodes.keys())]
