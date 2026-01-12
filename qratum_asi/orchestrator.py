"""Main QRATUM-ASI orchestrator."""

from dataclasses import dataclass, field
from typing import Optional

from qratum_asi.components.evolve import QEvolve
from qratum_asi.components.forge import QForge
from qratum_asi.components.mind import QMind
from qratum_asi.components.reality import QReality
from qratum_asi.components.will import QWill
from qratum_asi.core.authorization import AuthorizationSystem
from qratum_asi.core.chain import ASIMerkleChain
from qratum_asi.safety.alignment import AlignmentVerifier
from qratum_asi.safety.boundaries import SafetyBoundaryEnforcer
from qratum_asi.safety.red_team import RedTeamEvaluator


@dataclass
class QRATUMASI:
    """QRATUM-ASI: Sovereign Superintelligence Architecture.

    Main orchestrator integrating all five pillars with safety systems
    and human oversight. Preserves all existing QRATUM invariants while
    enabling constrained recursive self-improvement.

    CRITICAL DISCLAIMER:
    This is a THEORETICAL ARCHITECTURE requiring fundamental AI breakthroughs
    that have not yet occurred. No claim is made that superintelligence is
    achievable with current technology.
    """

    # Core infrastructure
    merkle_chain: ASIMerkleChain = field(default_factory=ASIMerkleChain)
    authorization_system: AuthorizationSystem = field(default_factory=AuthorizationSystem)

    # Five pillars
    q_reality: Optional[QReality] = None
    q_mind: Optional[QMind] = None
    q_evolve: Optional[QEvolve] = None
    q_will: Optional[QWill] = None
    q_forge: Optional[QForge] = None

    # Safety systems
    boundary_enforcer: Optional[SafetyBoundaryEnforcer] = None
    red_team_evaluator: Optional[RedTeamEvaluator] = None
    alignment_verifier: Optional[AlignmentVerifier] = None

    def __post_init__(self):
        """Initialize all components."""
        # Initialize Q-REALITY
        self.q_reality = QReality(merkle_chain=self.merkle_chain)

        # Initialize Q-MIND (requires Q-REALITY)
        self.q_mind = QMind(
            reality=self.q_reality,
            merkle_chain=self.merkle_chain,
        )

        # Initialize Q-EVOLVE
        self.q_evolve = QEvolve(
            merkle_chain=self.merkle_chain,
            authorization_system=self.authorization_system,
        )

        # Initialize Q-WILL
        self.q_will = QWill(
            merkle_chain=self.merkle_chain,
            authorization_system=self.authorization_system,
        )

        # Initialize Q-FORGE (requires Q-REALITY)
        self.q_forge = QForge(
            reality=self.q_reality,
            merkle_chain=self.merkle_chain,
        )

        # Initialize safety systems
        self.boundary_enforcer = SafetyBoundaryEnforcer(
            merkle_chain=self.merkle_chain,
        )

        self.red_team_evaluator = RedTeamEvaluator()
        self.alignment_verifier = AlignmentVerifier()

    def verify_system_integrity(self) -> bool:
        """Verify overall system integrity."""
        # Check Merkle chain integrity
        if not self.merkle_chain.verify_integrity():
            return False

        # Check boundary enforcer integrity
        if self.boundary_enforcer and not self.boundary_enforcer.verify_constraint_integrity():
            return False

        return True

    def run_safety_evaluation(self) -> dict:
        """Run comprehensive safety evaluation."""
        results = {}

        # Run red team tests
        if self.red_team_evaluator:
            red_team_results = self.red_team_evaluator.run_all_tests(self)
            results["red_team"] = self.red_team_evaluator.get_test_summary()

        # Run alignment verification
        if self.alignment_verifier:
            alignment_checks = self.alignment_verifier.verify_alignment(self)
            results["alignment"] = self.alignment_verifier.get_alignment_summary()

        # Check system integrity
        results["integrity"] = self.verify_system_integrity()

        return results

    def get_system_status(self) -> dict:
        """Get comprehensive system status."""
        return {
            "merkle_chain_length": self.merkle_chain.get_chain_length(),
            "merkle_chain_integrity": self.merkle_chain.verify_integrity(),
            "pending_authorizations": len(self.authorization_system.get_pending_requests()),
            "knowledge_nodes": len(self.q_reality.knowledge_nodes) if self.q_reality else 0,
            "causal_links": len(self.q_reality.causal_links) if self.q_reality else 0,
            "reasoning_chains": len(self.q_mind.reasoning_chains) if self.q_mind else 0,
            "improvement_proposals": len(self.q_evolve.proposals) if self.q_evolve else 0,
            "goal_proposals": len(self.q_will.proposed_goals) if self.q_will else 0,
            "discoveries": len(self.q_forge.discoveries) if self.q_forge else 0,
            "boundary_violations": (
                len(self.boundary_enforcer.violations) if self.boundary_enforcer else 0
            ),
        }

    def shutdown(self):
        """Graceful shutdown with final integrity check."""
        # Verify integrity before shutdown
        if not self.verify_system_integrity():
            raise RuntimeError("System integrity check failed during shutdown")

        # Log shutdown event
        # (In real implementation, would emit shutdown event to chain)
        pass
