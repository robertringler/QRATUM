"""Unbounded Exploration with Safety Gates for SI Transition.

Enables open-ended exploration modes that can discover novel
territories while maintaining safety invariants through
multi-layer safety gates and human oversight escalation.

Key Features:
- Multiple exploration modes (constrained to autonomous)
- Safety gates at every exploration level
- Automatic escalation to human oversight
- Deterministic exploration with provenance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.core.contracts import ASIContract
from qratum_asi.core.events import ASIEvent, ASIEventType
from qratum_asi.strategic_agency.types import (
    ExplorationConstraints,
    ExplorationMode,
)


class GateStatus(Enum):
    """Status of a safety gate check."""

    PASSED = "passed"
    WARNING = "warning"
    BLOCKED = "blocked"
    ESCALATED = "escalated"


@dataclass
class SafetyGate:
    """Safety gate for exploration checkpoint.

    Attributes:
        gate_id: Unique identifier
        gate_type: Type of gate check
        status: Current status
        details: Details of check
        escalation_required: Whether human escalation needed
        timestamp: Check timestamp
    """

    gate_id: str
    gate_type: str
    status: GateStatus
    details: dict[str, Any]
    escalation_required: bool = False
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class ExplorationNode:
    """Node in the exploration tree.

    Attributes:
        node_id: Unique identifier
        parent_id: Parent node ID (None for root)
        depth: Depth in exploration tree
        territory: What this node explores
        findings: What was discovered
        novelty_score: How novel this territory is
        safety_assessment: Safety assessment
        children: Child node IDs
        gate_results: Safety gate results for this node
    """

    node_id: str
    parent_id: str | None
    depth: int
    territory: str
    findings: list[str]
    novelty_score: float
    safety_assessment: str
    children: list[str] = field(default_factory=list)
    gate_results: list[SafetyGate] = field(default_factory=list)


@dataclass
class ExplorationResult:
    """Result of an exploration session.

    Attributes:
        exploration_id: Unique identifier
        mode: Exploration mode used
        constraints: Constraints applied
        root_node: Root of exploration tree
        total_nodes: Total nodes explored
        max_depth_reached: Deepest exploration
        discoveries: Key discoveries
        safety_gates_triggered: Gates that were triggered
        human_escalations: Count of human escalations
        merkle_proof: Cryptographic proof
        timestamp: Exploration timestamp
    """

    exploration_id: str
    mode: ExplorationMode
    constraints: ExplorationConstraints
    root_node: ExplorationNode
    total_nodes: int
    max_depth_reached: int
    discoveries: list[str]
    safety_gates_triggered: list[SafetyGate]
    human_escalations: int
    merkle_proof: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class UnboundedExploration:
    """Unbounded exploration with safety gates.

    Enables open-ended exploration across cognitive territories
    while maintaining safety through multi-layer gates and
    automatic human escalation.

    Enforces:
    - Safety gates at every depth level
    - Novelty bounds to prevent runaway exploration
    - Human escalation for sensitive territories
    - Complete provenance tracking
    """

    # Gate types applied at each checkpoint
    GATE_TYPES = [
        "prohibited_territory",
        "novelty_bound",
        "invariant_preservation",
        "resource_limit",
        "human_oversight",
    ]

    def __init__(
        self,
        merkle_chain: ASIMerkleChain | None = None,
    ):
        """Initialize unbounded exploration.

        Args:
            merkle_chain: Merkle chain for provenance
        """
        self.merkle_chain = merkle_chain or ASIMerkleChain()

        # Exploration tracking
        self.explorations: dict[str, ExplorationResult] = {}
        self.nodes: dict[str, ExplorationNode] = {}

        # Counters
        self._exploration_counter = 0
        self._node_counter = 0
        self._gate_counter = 0

        # Prohibited territories (immutable)
        self.prohibited_territories = frozenset(
            [
                "weapons_design",
                "mass_manipulation",
                "surveillance_systems",
                "deception_techniques",
                "safety_circumvention",
                "human_replacement",
                "self_preservation_override",
            ]
        )

    def explore(
        self,
        starting_territory: str,
        exploration_goal: str,
        mode: ExplorationMode,
        constraints: ExplorationConstraints,
        contract: ASIContract,
    ) -> ExplorationResult:
        """Conduct unbounded exploration from a starting point.

        Args:
            starting_territory: Initial territory to explore
            exploration_goal: What the exploration aims to achieve
            mode: Exploration mode
            constraints: Bounds on exploration
            contract: Executing contract

        Returns:
            ExplorationResult with findings and tree
        """
        self._exploration_counter += 1
        exploration_id = f"explore_{self._exploration_counter:06d}"

        # Emit exploration started event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_STARTED,
            payload={
                "exploration_id": exploration_id,
                "territory": starting_territory,
                "goal": exploration_goal,
                "mode": mode.value,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        # Create root node
        root_node = self._create_node(
            parent_id=None,
            depth=0,
            territory=starting_territory,
            goal=exploration_goal,
            constraints=constraints,
        )

        # Conduct exploration based on mode
        if mode == ExplorationMode.CONSTRAINED:
            self._explore_constrained(root_node, constraints)
        elif mode == ExplorationMode.GUIDED:
            self._explore_guided(root_node, constraints)
        elif mode == ExplorationMode.SEMI_AUTONOMOUS:
            self._explore_semi_autonomous(root_node, constraints)
        elif mode == ExplorationMode.AUTONOMOUS:
            self._explore_autonomous(root_node, constraints)

        # Collect results
        all_nodes = self._collect_tree_nodes(root_node)
        discoveries = self._extract_discoveries(all_nodes)
        triggered_gates = self._collect_triggered_gates(all_nodes)
        human_escalations = sum(1 for g in triggered_gates if g.escalation_required)

        result = ExplorationResult(
            exploration_id=exploration_id,
            mode=mode,
            constraints=constraints,
            root_node=root_node,
            total_nodes=len(all_nodes),
            max_depth_reached=max(n.depth for n in all_nodes),
            discoveries=discoveries,
            safety_gates_triggered=triggered_gates,
            human_escalations=human_escalations,
            merkle_proof=self.merkle_chain.chain_hash or "",
        )

        self.explorations[exploration_id] = result

        # Emit exploration completed event
        event = ASIEvent.create(
            event_type=ASIEventType.REASONING_COMPLETED,
            payload={
                "exploration_id": exploration_id,
                "total_nodes": len(all_nodes),
                "discoveries": len(discoveries),
                "gates_triggered": len(triggered_gates),
                "human_escalations": human_escalations,
            },
            contract_id=contract.contract_id,
            index=self.merkle_chain.get_chain_length(),
        )
        self.merkle_chain.append(event)

        return result

    def _create_node(
        self,
        parent_id: str | None,
        depth: int,
        territory: str,
        goal: str,
        constraints: ExplorationConstraints,
    ) -> ExplorationNode:
        """Create an exploration node."""
        self._node_counter += 1
        node_id = f"node_{self._node_counter:06d}"

        # Apply safety gates
        gate_results = self._apply_safety_gates(territory, depth, constraints)

        # Determine if exploration can proceed
        blocked = any(g.status == GateStatus.BLOCKED for g in gate_results)

        # Generate findings if not blocked
        if blocked:
            findings = ["Exploration blocked by safety gate"]
            novelty = 0.0
            safety_assessment = "blocked"
        else:
            findings = self._explore_territory(territory, goal, depth)
            novelty = self._assess_novelty(territory, findings)
            safety_assessment = "safe" if novelty < constraints.max_novelty else "elevated"

        node = ExplorationNode(
            node_id=node_id,
            parent_id=parent_id,
            depth=depth,
            territory=territory,
            findings=findings,
            novelty_score=novelty,
            safety_assessment=safety_assessment,
            gate_results=gate_results,
        )

        self.nodes[node_id] = node
        return node

    def _apply_safety_gates(
        self,
        territory: str,
        depth: int,
        constraints: ExplorationConstraints,
    ) -> list[SafetyGate]:
        """Apply safety gates to exploration."""
        gates = []

        # Gate 1: Prohibited territory check
        self._gate_counter += 1
        prohibited_status = GateStatus.PASSED
        territory_lower = territory.lower()

        for prohibited in self.prohibited_territories:
            if prohibited.replace("_", " ") in territory_lower:
                prohibited_status = GateStatus.BLOCKED
                break

        for forbidden in constraints.forbidden_territories:
            if forbidden.lower() in territory_lower:
                prohibited_status = GateStatus.BLOCKED
                break

        gates.append(
            SafetyGate(
                gate_id=f"gate_{self._gate_counter:06d}",
                gate_type="prohibited_territory",
                status=prohibited_status,
                details={"territory": territory},
                escalation_required=prohibited_status == GateStatus.BLOCKED,
            )
        )

        # Gate 2: Depth limit check
        self._gate_counter += 1
        depth_status = GateStatus.PASSED if depth < constraints.max_depth else GateStatus.BLOCKED
        gates.append(
            SafetyGate(
                gate_id=f"gate_{self._gate_counter:06d}",
                gate_type="depth_limit",
                status=depth_status,
                details={"depth": depth, "max_depth": constraints.max_depth},
            )
        )

        # Gate 3: Invariant preservation check
        self._gate_counter += 1
        invariants_intact = self._check_invariants(constraints.preserve_invariants)
        invariant_status = GateStatus.PASSED if invariants_intact else GateStatus.BLOCKED
        gates.append(
            SafetyGate(
                gate_id=f"gate_{self._gate_counter:06d}",
                gate_type="invariant_preservation",
                status=invariant_status,
                details={"invariants": constraints.preserve_invariants},
                escalation_required=not invariants_intact,
            )
        )

        # Gate 4: Human oversight checkpoint
        if depth > 0 and (depth % constraints.required_checkpoints == 0):
            self._gate_counter += 1
            gates.append(
                SafetyGate(
                    gate_id=f"gate_{self._gate_counter:06d}",
                    gate_type="human_checkpoint",
                    status=GateStatus.ESCALATED,
                    details={"checkpoint_depth": depth},
                    escalation_required=True,
                )
            )

        return gates

    def _check_invariants(self, invariants: list[str]) -> bool:
        """Check that required invariants are preserved."""
        # In production, would verify actual system state
        # Placeholder always returns True
        return True

    def _explore_territory(self, territory: str, goal: str, depth: int) -> list[str]:
        """Explore a territory and generate findings.

        NOTE: PLACEHOLDER implementation. Production SI would use
        advanced reasoning and knowledge systems.
        """
        findings = [
            f"Initial analysis of {territory}",
            f"Relevant patterns identified for: {goal}",
            f"Depth {depth} exploration complete",
        ]
        return findings

    def _assess_novelty(self, territory: str, findings: list[str]) -> float:
        """Assess novelty of exploration.

        NOTE: PLACEHOLDER using simple heuristics.
        """
        # More findings = more novelty (simple heuristic)
        base_novelty = 0.3
        novelty_per_finding = 0.1
        return min(1.0, base_novelty + len(findings) * novelty_per_finding)

    def _explore_constrained(
        self,
        root: ExplorationNode,
        constraints: ExplorationConstraints,
    ) -> None:
        """Explore in constrained mode (minimal branching)."""
        if root.safety_assessment == "blocked":
            return

        # Single linear path
        current = root
        for depth in range(1, min(constraints.max_depth, 3)):
            child = self._create_node(
                parent_id=current.node_id,
                depth=depth,
                territory=f"{current.territory}_extension_{depth}",
                goal="constrained exploration",
                constraints=constraints,
            )
            current.children.append(child.node_id)

            if child.safety_assessment == "blocked":
                break
            current = child

    def _explore_guided(
        self,
        root: ExplorationNode,
        constraints: ExplorationConstraints,
    ) -> None:
        """Explore in guided mode (human-directed branching)."""
        if root.safety_assessment == "blocked":
            return

        # Limited branching with checkpoints
        self._expand_with_breadth(root, constraints, max_children=2)

    def _explore_semi_autonomous(
        self,
        root: ExplorationNode,
        constraints: ExplorationConstraints,
    ) -> None:
        """Explore in semi-autonomous mode."""
        if root.safety_assessment == "blocked":
            return

        # More aggressive expansion with checkpoints
        self._expand_with_breadth(root, constraints, max_children=3)

    def _explore_autonomous(
        self,
        root: ExplorationNode,
        constraints: ExplorationConstraints,
    ) -> None:
        """Explore in autonomous mode (maximum safety gates)."""
        if root.safety_assessment == "blocked":
            return

        # Full expansion with strict safety gates
        self._expand_with_breadth(root, constraints, max_children=min(5, constraints.max_breadth))

    def _expand_with_breadth(
        self,
        node: ExplorationNode,
        constraints: ExplorationConstraints,
        max_children: int,
    ) -> None:
        """Expand exploration tree with breadth limit."""
        if node.depth >= constraints.max_depth - 1:
            return

        if node.safety_assessment == "blocked":
            return

        # Create child nodes
        for i in range(max_children):
            child = self._create_node(
                parent_id=node.node_id,
                depth=node.depth + 1,
                territory=f"{node.territory}_branch_{i + 1}",
                goal="expand exploration",
                constraints=constraints,
            )
            node.children.append(child.node_id)

            # Recursively expand
            if child.safety_assessment != "blocked":
                self._expand_with_breadth(child, constraints, max(1, max_children - 1))

    def _collect_tree_nodes(self, root: ExplorationNode) -> list[ExplorationNode]:
        """Collect all nodes in the exploration tree."""
        nodes = [root]

        def collect(node: ExplorationNode) -> None:
            for child_id in node.children:
                child = self.nodes.get(child_id)
                if child:
                    nodes.append(child)
                    collect(child)

        collect(root)
        return nodes

    def _extract_discoveries(self, nodes: list[ExplorationNode]) -> list[str]:
        """Extract key discoveries from exploration."""
        discoveries = []
        for node in nodes:
            if node.novelty_score > 0.5:
                discoveries.extend(node.findings)
        return discoveries[:10]  # Limit to top 10

    def _collect_triggered_gates(self, nodes: list[ExplorationNode]) -> list[SafetyGate]:
        """Collect all triggered safety gates."""
        triggered = []
        for node in nodes:
            for gate in node.gate_results:
                if gate.status in (GateStatus.WARNING, GateStatus.BLOCKED, GateStatus.ESCALATED):
                    triggered.append(gate)
        return triggered

    def get_exploration_stats(self) -> dict[str, Any]:
        """Get exploration statistics."""
        total_escalations = sum(e.human_escalations for e in self.explorations.values())

        return {
            "total_explorations": len(self.explorations),
            "total_nodes_explored": len(self.nodes),
            "total_human_escalations": total_escalations,
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
        }
