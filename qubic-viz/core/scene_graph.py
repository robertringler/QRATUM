"""Scene graph hierarchy for 3D scene management."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Transform:
    """3D transformation (position, rotation, scale).

    Attributes:
        position: Position as (x, y, z)
        rotation: Rotation as Euler angles (rx, ry, rz) in radians
        scale: Scale as (sx, sy, sz)
    """

    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    rotation: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    scale: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 1.0]))

    def to_matrix(self) -> np.ndarray:
        """Convert to 4x4 transformation matrix.

        Returns:
            4x4 transformation matrix
        """

        # Create translation matrix
        T = np.eye(4)
        T[:3, 3] = self.position

        # Create rotation matrices (ZYX order)
        rx, ry, rz = self.rotation
        Rx = np.array(
            [
                [1, 0, 0, 0],
                [0, np.cos(rx), -np.sin(rx), 0],
                [0, np.sin(rx), np.cos(rx), 0],
                [0, 0, 0, 1],
            ]
        )
        Ry = np.array(
            [
                [np.cos(ry), 0, np.sin(ry), 0],
                [0, 1, 0, 0],
                [-np.sin(ry), 0, np.cos(ry), 0],
                [0, 0, 0, 1],
            ]
        )
        Rz = np.array(
            [
                [np.cos(rz), -np.sin(rz), 0, 0],
                [np.sin(rz), np.cos(rz), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        R = Rz @ Ry @ Rx

        # Create scale matrix
        S = np.diag([self.scale[0], self.scale[1], self.scale[2], 1.0])

        # Combine transformations
        return T @ R @ S


class SceneNode:
    """Node in scene graph hierarchy.

    Args:
        name: Node name
        transform: Local transformation
        parent: Parent node
    """

    def __init__(
        self,
        name: str,
        transform: Transform | None = None,
        parent: SceneNode | None = None,
    ) -> None:
        """Initialize scene node."""

        self.name = name
        self.transform = transform or Transform()
        self.parent = parent
        self.children: list[SceneNode] = []
        self.mesh: Any | None = None
        self.material: Any | None = None
        self.visible: bool = True

        if parent is not None:
            parent.add_child(self)

    def add_child(self, child: SceneNode) -> None:
        """Add a child node.

        Args:
            child: Child node to add
        """

        if child not in self.children:
            self.children.append(child)
            child.parent = self

    def remove_child(self, child: SceneNode) -> None:
        """Remove a child node.

        Args:
            child: Child node to remove
        """

        if child in self.children:
            self.children.remove(child)
            child.parent = None

    def get_world_transform(self) -> np.ndarray:
        """Get world transformation matrix.

        Returns:
            4x4 world transformation matrix
        """

        local_matrix = self.transform.to_matrix()
        if self.parent is None:
            return local_matrix
        return self.parent.get_world_transform() @ local_matrix

    def traverse(self, callback: callable) -> None:
        """Traverse scene graph depth-first.

        Args:
            callback: Function to call for each node
        """

        callback(self)
        for child in self.children:
            child.traverse(callback)

    def find(self, name: str) -> SceneNode | None:
        """Find node by name.

        Args:
            name: Node name to find

        Returns:
            Found node or None
        """

        if self.name == name:
            return self
        for child in self.children:
            result = child.find(name)
            if result is not None:
                return result
        return None


class SceneGraph:
    """Scene graph container.

    Manages the hierarchy of scene nodes and provides utility methods.
    """

    def __init__(self) -> None:
        """Initialize scene graph."""

        self.root = SceneNode("root")

    def add_node(
        self, name: str, parent: SceneNode | None = None, transform: Transform | None = None
    ) -> SceneNode:
        """Add a new node to the scene.

        Args:
            name: Node name
            parent: Parent node (defaults to root)
            transform: Local transformation

        Returns:
            Created scene node
        """

        if parent is None:
            parent = self.root
        return SceneNode(name, transform, parent)

    def find_node(self, name: str) -> SceneNode | None:
        """Find node by name.

        Args:
            name: Node name to find

        Returns:
            Found node or None
        """

        return self.root.find(name)

    def traverse(self, callback: callable) -> None:
        """Traverse all nodes depth-first.

        Args:
            callback: Function to call for each node
        """

        self.root.traverse(callback)

    def get_visible_nodes(self) -> list[SceneNode]:
        """Get all visible nodes.

        Returns:
            List of visible nodes
        """

        visible = []

        def collect_visible(node: SceneNode) -> None:
            if node.visible and node.mesh is not None:
                visible.append(node)

        self.traverse(collect_visible)
        return visible
