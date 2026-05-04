"""Search Tree Explorer Panel: Real-time MCTS/AAS tree visualization.

Features:
- Real-time MCTS / AAS tree visualization
- Node visit counts, entropy heatmaps, and branch probability visualization
- Mid-search heuristic rewrites and dynamic subtree highlighting
- Zoom, pan, and branch collapse/expand for interactive inspection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from qratum_chess.search.mcts import MCTSNode


class TreeLayout(Enum):
    """Tree visualization layout modes."""

    RADIAL = "radial"  # Nodes arranged in circles
    HIERARCHICAL = "hierarchical"  # Traditional top-down tree
    FORCE_DIRECTED = "force_directed"  # Physics-based layout


class NodeColorMode(Enum):
    """Node coloring modes."""

    VISITS = "visits"  # Color by visit count
    VALUE = "value"  # Color by evaluation value
    ENTROPY = "entropy"  # Color by entropy
    NOVELTY = "novelty"  # Color by novelty score
    POLICY = "policy"  # Color by policy probability


@dataclass
class TreeNode:
    """Visualization data for a search tree node."""

    node_id: int
    parent_id: int | None
    move: str  # UCI move that led to this node
    visits: int
    value: float
    policy_prob: float
    entropy: float
    novelty: float
    children: list[int] = field(default_factory=list)
    is_expanded: bool = True
    is_highlighted: bool = False
    depth: int = 0
    position_fen: str = ""

    # Layout coordinates (computed by layout algorithm)
    x: float = 0.0
    y: float = 0.0


@dataclass
class SearchTreeState:
    """Complete state of the search tree panel."""

    nodes: dict[int, TreeNode] = field(default_factory=dict)
    root_id: int = 0

    # View settings
    layout: TreeLayout = TreeLayout.HIERARCHICAL
    color_mode: NodeColorMode = NodeColorMode.VISITS

    # Navigation
    zoom: float = 1.0
    pan_x: float = 0.0
    pan_y: float = 0.0

    # Filtering
    min_visits: int = 0
    max_depth: int = 10
    show_only_pv: bool = False

    # Highlighting
    highlighted_path: list[int] = field(default_factory=list)
    selected_node: int | None = None

    # Animation
    animate_expansion: bool = True
    animation_speed: float = 0.3

    # Display options
    show_values: bool = True
    show_visits: bool = True
    show_entropy: bool = False
    show_policy: bool = True
    node_size_by_visits: bool = True


class SearchTreePanel:
    """Search Tree Explorer Panel for visualizing MCTS/AAS trees.

    This panel provides:
    - Interactive tree visualization with multiple layout modes
    - Node coloring by visits, value, entropy, or novelty
    - Dynamic filtering and depth control
    - PV line highlighting
    - Zoom and pan controls
    - Mid-search updates with animation
    """

    # Color palettes for different modes
    COLOR_PALETTES = {
        NodeColorMode.VISITS: {
            "low": (0.2, 0.2, 0.6),
            "high": (1.0, 0.8, 0.0),
        },
        NodeColorMode.VALUE: {
            "low": (0.8, 0.2, 0.2),  # Losing - red
            "high": (0.2, 0.8, 0.2),  # Winning - green
        },
        NodeColorMode.ENTROPY: {
            "low": (0.0, 0.2, 0.5),  # Certain - dark blue
            "high": (1.0, 0.5, 0.0),  # Uncertain - orange
        },
        NodeColorMode.NOVELTY: {
            "low": (0.5, 0.5, 0.5),  # Common - gray
            "high": (0.8, 0.0, 1.0),  # Novel - purple
        },
        NodeColorMode.POLICY: {
            "low": (0.2, 0.2, 0.4),
            "high": (0.0, 1.0, 1.0),  # High probability - cyan
        },
    }

    def __init__(self, width: int = 600, height: int = 500) -> None:
        """Initialize search tree panel.

        Args:
            width: Panel width in pixels
            height: Panel height in pixels
        """
        self.width = width
        self.height = height
        self.state = SearchTreeState()

        # Node ID counter
        self._next_id = 0

    def _get_next_id(self) -> int:
        """Get next unique node ID."""
        node_id = self._next_id
        self._next_id += 1
        return node_id

    def clear(self) -> None:
        """Clear the tree."""
        self.state.nodes.clear()
        self.state.root_id = 0
        self.state.highlighted_path.clear()
        self.state.selected_node = None
        self._next_id = 0

    def update_from_mcts(self, root: MCTSNode) -> None:
        """Update tree from MCTS root node.

        Args:
            root: MCTS root node
        """
        self.clear()
        self._add_mcts_node(root, None, 0)
        self._compute_layout()

    def _add_mcts_node(self, node: MCTSNode, parent_id: int | None, depth: int) -> int:
        """Recursively add MCTS node to tree.

        Args:
            node: MCTS node to add
            parent_id: Parent node ID
            depth: Current depth

        Returns:
            ID of added node
        """
        if depth > self.state.max_depth:
            return -1

        node_id = self._get_next_id()

        # Calculate entropy from children
        entropy = 0.0
        if node.children:
            visits = [c.visits for c in node.children if c.visits > 0]
            total = sum(visits)
            if total > 0:
                probs = [v / total for v in visits]
                entropy = -sum(p * np.log(p + 1e-10) for p in probs)

        tree_node = TreeNode(
            node_id=node_id,
            parent_id=parent_id,
            move=node.move.to_uci() if node.move else "root",
            visits=node.visits,
            value=node.value(),
            policy_prob=node.prior,
            entropy=entropy,
            novelty=getattr(node, "novelty", 0.0),
            depth=depth,
        )

        self.state.nodes[node_id] = tree_node

        if node_id == 0:
            self.state.root_id = node_id

        if parent_id is not None:
            self.state.nodes[parent_id].children.append(node_id)

        # Add children (sorted by visits)
        sorted_children = sorted(node.children, key=lambda c: c.visits, reverse=True)
        for child in sorted_children:
            if child.visits >= self.state.min_visits:
                self._add_mcts_node(child, node_id, depth + 1)

        return node_id

    def update_node(
        self, node_id: int, visits: int, value: float, policy_prob: float = 0.0
    ) -> None:
        """Update a specific node's values.

        Args:
            node_id: Node ID to update
            visits: New visit count
            value: New value
            policy_prob: New policy probability
        """
        if node_id in self.state.nodes:
            node = self.state.nodes[node_id]
            node.visits = visits
            node.value = value
            node.policy_prob = policy_prob

    def highlight_pv(self, moves: list[str]) -> None:
        """Highlight principal variation path.

        Args:
            moves: List of UCI moves forming the PV
        """
        self.state.highlighted_path.clear()

        current_id = self.state.root_id
        self.state.highlighted_path.append(current_id)

        for move in moves:
            node = self.state.nodes.get(current_id)
            if not node:
                break

            # Find child with matching move
            for child_id in node.children:
                child = self.state.nodes.get(child_id)
                if child and child.move == move:
                    self.state.highlighted_path.append(child_id)
                    current_id = child_id
                    break

    def toggle_node_expansion(self, node_id: int) -> None:
        """Toggle node expansion state.

        Args:
            node_id: Node ID to toggle
        """
        if node_id in self.state.nodes:
            self.state.nodes[node_id].is_expanded = not self.state.nodes[node_id].is_expanded

    def select_node(self, node_id: int | None) -> None:
        """Select a node for detailed view.

        Args:
            node_id: Node ID to select (None to deselect)
        """
        self.state.selected_node = node_id

    def zoom_in(self, factor: float = 1.2) -> None:
        """Zoom in on the tree."""
        self.state.zoom = min(5.0, self.state.zoom * factor)

    def zoom_out(self, factor: float = 1.2) -> None:
        """Zoom out from the tree."""
        self.state.zoom = max(0.1, self.state.zoom / factor)

    def pan(self, dx: float, dy: float) -> None:
        """Pan the view.

        Args:
            dx: X offset
            dy: Y offset
        """
        self.state.pan_x += dx / self.state.zoom
        self.state.pan_y += dy / self.state.zoom

    def reset_view(self) -> None:
        """Reset zoom and pan to defaults."""
        self.state.zoom = 1.0
        self.state.pan_x = 0.0
        self.state.pan_y = 0.0

    def set_layout(self, layout: TreeLayout) -> None:
        """Set tree layout mode.

        Args:
            layout: Layout mode to use
        """
        self.state.layout = layout
        self._compute_layout()

    def set_color_mode(self, mode: NodeColorMode) -> None:
        """Set node coloring mode.

        Args:
            mode: Color mode to use
        """
        self.state.color_mode = mode

    def _compute_layout(self) -> None:
        """Compute node positions based on layout mode."""
        if not self.state.nodes:
            return

        if self.state.layout == TreeLayout.HIERARCHICAL:
            self._compute_hierarchical_layout()
        elif self.state.layout == TreeLayout.RADIAL:
            self._compute_radial_layout()
        else:
            self._compute_force_directed_layout()

    def _compute_hierarchical_layout(self) -> None:
        """Compute hierarchical (top-down) layout."""
        # First pass: compute subtree widths
        widths: dict[int, int] = {}

        def compute_width(node_id: int) -> int:
            node = self.state.nodes[node_id]
            if not node.children or not node.is_expanded:
                widths[node_id] = 1
                return 1

            total_width = sum(compute_width(c) for c in node.children)
            widths[node_id] = max(1, total_width)
            return widths[node_id]

        compute_width(self.state.root_id)

        # Second pass: assign positions
        def assign_positions(node_id: int, x: float, y: float, width: float) -> None:
            node = self.state.nodes[node_id]
            node.x = x + width / 2
            node.y = y

            if node.children and node.is_expanded:
                child_x = x
                for child_id in node.children:
                    child_width = widths[child_id] * width / widths[node_id]
                    assign_positions(child_id, child_x, y + 60, child_width)
                    child_x += child_width

        total_width = widths[self.state.root_id] * 80
        assign_positions(self.state.root_id, 0, 40, total_width)

    def _compute_radial_layout(self) -> None:
        """Compute radial (circular) layout."""
        center_x = self.width / 2
        center_y = self.height / 2

        def assign_radial(
            node_id: int, angle_start: float, angle_end: float, radius: float
        ) -> None:
            node = self.state.nodes[node_id]
            angle = (angle_start + angle_end) / 2

            if node.depth == 0:
                node.x = center_x
                node.y = center_y
            else:
                node.x = center_x + radius * np.cos(angle)
                node.y = center_y + radius * np.sin(angle)

            if node.children and node.is_expanded:
                angle_span = angle_end - angle_start
                child_angle = angle_start
                for child_id in node.children:
                    child_span = angle_span / len(node.children)
                    assign_radial(child_id, child_angle, child_angle + child_span, radius + 60)
                    child_angle += child_span

        assign_radial(self.state.root_id, 0, 2 * np.pi, 0)

    def _compute_force_directed_layout(self) -> None:
        """Compute force-directed layout (simplified)."""
        # Initialize positions randomly
        for node in self.state.nodes.values():
            node.x = np.random.uniform(50, self.width - 50)
            node.y = 50 + node.depth * 60

        # Simple spring/repulsion simulation
        for _ in range(50):
            forces: dict[int, tuple[float, float]] = dict.fromkeys(self.state.nodes, (0.0, 0.0))

            # Repulsion between all nodes
            node_list = list(self.state.nodes.values())
            for i, n1 in enumerate(node_list):
                for n2 in node_list[i + 1 :]:
                    dx = n1.x - n2.x
                    dy = n1.y - n2.y
                    dist = max(1, np.sqrt(dx * dx + dy * dy))
                    force = 500 / (dist * dist)

                    fx, fy = forces[n1.node_id]
                    forces[n1.node_id] = (fx + force * dx / dist, fy + force * dy / dist)
                    fx, fy = forces[n2.node_id]
                    forces[n2.node_id] = (fx - force * dx / dist, fy - force * dy / dist)

            # Attraction along edges
            for node in self.state.nodes.values():
                if node.parent_id is not None:
                    parent = self.state.nodes[node.parent_id]
                    dx = parent.x - node.x
                    dy = parent.y - node.y
                    dist = max(1, np.sqrt(dx * dx + dy * dy))
                    force = dist / 10

                    fx, fy = forces[node.node_id]
                    forces[node.node_id] = (fx + force * dx / dist, fy + force * dy / dist)

            # Apply forces
            for node_id, (fx, fy) in forces.items():
                node = self.state.nodes[node_id]
                node.x = max(30, min(self.width - 30, node.x + fx * 0.1))
                node.y = max(30, min(self.height - 30, node.y + fy * 0.1))

    def _get_node_color(self, node: TreeNode) -> tuple[float, float, float]:
        """Get node color based on current color mode.

        Args:
            node: Tree node

        Returns:
            RGB color tuple
        """
        palette = self.COLOR_PALETTES[self.state.color_mode]

        # Normalize value to 0-1
        if self.state.color_mode == NodeColorMode.VISITS:
            max_visits = max(n.visits for n in self.state.nodes.values()) or 1
            t = node.visits / max_visits
        elif self.state.color_mode == NodeColorMode.VALUE:
            t = (node.value + 1) / 2  # Map -1,1 to 0,1
        elif self.state.color_mode == NodeColorMode.ENTROPY:
            max_entropy = max(n.entropy for n in self.state.nodes.values()) or 1
            t = node.entropy / max_entropy
        elif self.state.color_mode == NodeColorMode.NOVELTY:
            t = min(1.0, node.novelty)
        else:  # POLICY
            t = node.policy_prob

        # Interpolate between low and high colors
        low = palette["low"]
        high = palette["high"]
        return (
            low[0] + t * (high[0] - low[0]),
            low[1] + t * (high[1] - low[1]),
            low[2] + t * (high[2] - low[2]),
        )

    def get_render_data(self) -> dict[str, Any]:
        """Get all data needed for rendering.

        Returns:
            Dictionary with tree state for visualization
        """
        # Build node data
        nodes_data = []
        for node_id, node in self.state.nodes.items():
            # Skip collapsed children
            if node.parent_id is not None:
                parent = self.state.nodes.get(node.parent_id)
                if parent and not parent.is_expanded:
                    continue

            color = self._get_node_color(node)
            size = (
                10 + (node.visits / max(1, max(n.visits for n in self.state.nodes.values()))) * 20
                if self.state.node_size_by_visits
                else 15
            )

            nodes_data.append(
                {
                    "id": node_id,
                    "parent_id": node.parent_id,
                    "move": node.move,
                    "visits": node.visits,
                    "value": node.value,
                    "policy": node.policy_prob,
                    "entropy": node.entropy,
                    "novelty": node.novelty,
                    "depth": node.depth,
                    "x": node.x,
                    "y": node.y,
                    "color": color,
                    "size": size,
                    "is_expanded": node.is_expanded,
                    "is_highlighted": node_id in self.state.highlighted_path,
                    "is_selected": node_id == self.state.selected_node,
                    "children_count": len(node.children),
                }
            )

        # Build edge data
        edges_data = []
        for node_id, node in self.state.nodes.items():
            if node.parent_id is not None:
                parent = self.state.nodes.get(node.parent_id)
                if parent and parent.is_expanded:
                    is_pv = (
                        node_id in self.state.highlighted_path
                        and node.parent_id in self.state.highlighted_path
                    )
                    edges_data.append(
                        {
                            "from_id": node.parent_id,
                            "to_id": node_id,
                            "from_x": parent.x,
                            "from_y": parent.y,
                            "to_x": node.x,
                            "to_y": node.y,
                            "is_pv": is_pv,
                        }
                    )

        # Selected node details
        selected_details = None
        if self.state.selected_node is not None:
            node = self.state.nodes.get(self.state.selected_node)
            if node:
                selected_details = {
                    "move": node.move,
                    "visits": node.visits,
                    "value": node.value,
                    "policy": node.policy_prob,
                    "entropy": node.entropy,
                    "novelty": node.novelty,
                    "depth": node.depth,
                    "fen": node.position_fen,
                    "children": [
                        {
                            "move": self.state.nodes[c].move,
                            "visits": self.state.nodes[c].visits,
                            "value": self.state.nodes[c].value,
                        }
                        for c in node.children
                        if c in self.state.nodes
                    ],
                }

        return {
            "width": self.width,
            "height": self.height,
            "nodes": nodes_data,
            "edges": edges_data,
            "layout": self.state.layout.value,
            "color_mode": self.state.color_mode.value,
            "view": {
                "zoom": self.state.zoom,
                "pan_x": self.state.pan_x,
                "pan_y": self.state.pan_y,
            },
            "filters": {
                "min_visits": self.state.min_visits,
                "max_depth": self.state.max_depth,
                "show_only_pv": self.state.show_only_pv,
            },
            "highlighted_path": self.state.highlighted_path,
            "selected_node": self.state.selected_node,
            "selected_details": selected_details,
            "display_options": {
                "show_values": self.state.show_values,
                "show_visits": self.state.show_visits,
                "show_entropy": self.state.show_entropy,
                "show_policy": self.state.show_policy,
                "node_size_by_visits": self.state.node_size_by_visits,
                "animate_expansion": self.state.animate_expansion,
            },
            "stats": {
                "total_nodes": len(self.state.nodes),
                "max_depth": max((n.depth for n in self.state.nodes.values()), default=0),
                "total_visits": sum(n.visits for n in self.state.nodes.values()),
            },
        }

    def to_json(self) -> str:
        """Serialize render data to JSON."""
        import json

        return json.dumps(self.get_render_data())
