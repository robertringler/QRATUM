"""QRATUM 12 Calibration Doctrine - Jurisdictional Laws for Sovereign Computation.

This module codifies the 12 foundational calibrations as operational doctrine
for QRATUM's sovereign computation substrate. These are not suggestions—they
are jurisdictional laws grounding all actions in structural legitimacy,
epistemic foresight, and narrative inevitability.

The doctrine governs:
- Verifiable Computational Jurisdiction
- QRADLE as Enabling Substrate for Regulated Realms
- Verticals as Constraint-Driven Stress Tests
- Defensive Engine for Self-Stabilization
- ASI Scaffolding with Jurisdiction-First Legitimacy
- Long-Term Ontology Change

Reference: QRATUM Mega Prompt (December 2025)
"""

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class CalibrationCategory(Enum):
    """Categories of calibration axioms."""

    FOUNDATIONAL = "foundational"  # Core truth and substrate
    OPERATIONAL = "operational"  # Runtime behavior
    STRATEGIC = "strategic"  # Long-term positioning
    EPISTEMOLOGICAL = "epistemological"  # Knowledge and awareness


class JurisdictionalProperty(Enum):
    """Properties that define jurisdictional computation."""

    DETERMINISM = "determinism"  # Hard invariant, not best-effort
    AUDITABILITY = "auditability"  # Cryptographic/Merkle-chained
    REVERSIBILITY = "reversibility"  # Native, not manual
    SOVEREIGNTY = "sovereignty"  # Enforced, not optional
    PRIVACY = "privacy"  # Zero-knowledge operational
    OVERSIGHT = "oversight"  # Protocol-embedded


class TrajectoryState(Enum):
    """Trajectory-awareness states for defensive engine."""

    STABLE = "stable"  # Normal operation
    DRIFT = "drift"  # Entropy gradient detected
    METASTABLE = "metastable"  # Clustering detected
    PRECURSOR = "precursor"  # Collapse precursor signals
    CRITICAL = "critical"  # Approaching phase transition
    SELF_SUSPEND = "self_suspend"  # Conditional self-suspension


@dataclass(frozen=True)
class CalibrationAxiom:
    """An immutable calibration axiom from the 12-axiom doctrine.

    Each axiom defines a jurisdictional law that QRATUM must uphold.
    Axioms are cryptographically anchored and cannot be modified.
    """

    axiom_id: int
    name: str
    description: str
    category: CalibrationCategory
    properties: Tuple[JurisdictionalProperty, ...]
    enforcement_hash: str  # SHA3-256 hash of axiom content
    is_immutable: bool = True

    def verify_integrity(self) -> bool:
        """Verify axiom has not been tampered with."""
        computed_hash = self._compute_hash()
        return computed_hash == self.enforcement_hash

    def _compute_hash(self) -> str:
        """Compute SHA3-256 hash of axiom content."""
        content = f"{self.axiom_id}:{self.name}:{self.description}:{self.category.value}"
        content += f":{','.join(p.value for p in self.properties)}"
        return hashlib.sha3_256(content.encode()).hexdigest()


@dataclass
class TrajectoryMetrics:
    """Metrics for trajectory-awareness (defensive engine).

    Enables reasoning about "what the system is becoming" rather than
    just "what happened" or "what to rollback".
    """

    entropy_gradient: float  # Rate of entropy change
    coupling_drift: float  # Inter-module coupling drift
    metastable_clusters: int  # Number of metastable coupling clusters
    collapse_precursors: int  # Number of collapse precursor signals
    resilience_compression: float  # Recovery time compression factor
    trajectory_state: TrajectoryState
    timestamp: str

    # Self-suspension threshold constants
    PRECURSOR_THRESHOLD: int = 3  # Collapse precursors count that triggers suspension
    RESILIENCE_THRESHOLD: float = 0.3  # Resilience compression below this triggers suspension

    def should_self_suspend(self) -> bool:
        """Determine if system should conditionally self-suspend."""
        # Self-suspend if critical state or multiple precursors
        if self.trajectory_state == TrajectoryState.CRITICAL:
            return True
        if self.collapse_precursors >= self.PRECURSOR_THRESHOLD:
            return True
        if self.resilience_compression < self.RESILIENCE_THRESHOLD:
            return True
        return False


@dataclass
class JurisdictionalClaim:
    """A claim about what happened or could not have happened.

    Enables proving jurisdictional facts for legally meaningful computation.
    """

    claim_id: str
    claim_type: str  # "happened", "could_not_happen", "state_transition"
    subject: str  # What/who the claim is about
    predicate: str  # What is being claimed
    evidence_hash: str  # Hash of supporting evidence
    timestamp: str
    merkle_proof: Optional[str] = None

    def is_provable(self) -> bool:
        """Check if claim has verifiable evidence."""
        return self.evidence_hash is not None and self.merkle_proof is not None


# =============================================================================
# THE 12 CALIBRATION AXIOMS
# =============================================================================


def _create_axiom(
    axiom_id: int,
    name: str,
    description: str,
    category: CalibrationCategory,
    properties: Tuple[JurisdictionalProperty, ...],
) -> CalibrationAxiom:
    """Create an axiom with computed enforcement hash."""
    content = f"{axiom_id}:{name}:{description}:{category.value}"
    content += f":{','.join(p.value for p in properties)}"
    enforcement_hash = hashlib.sha3_256(content.encode()).hexdigest()

    return CalibrationAxiom(
        axiom_id=axiom_id,
        name=name,
        description=description,
        category=category,
        properties=properties,
        enforcement_hash=enforcement_hash,
        is_immutable=True,
    )


# Axiom 1: Verifiable Computational Jurisdiction
AXIOM_1_JURISDICTION = _create_axiom(
    axiom_id=1,
    name="Verifiable Computational Jurisdiction as Core Truth",
    description=(
        "QRATUM is not secure software but a sovereign execution domain where "
        "determinism is a hard invariant (not best-effort), auditability is "
        "cryptographic/Merkle-chained (not lossy logs), reversibility is native "
        "(not manual), sovereignty is enforced (not optional), privacy is "
        "zero-knowledge operational (not encryption-at-rest), and oversight is "
        "protocol-embedded (not organizational). This enables proving 'what happened' "
        "and 'what could not have happened,' making QRATUM a legally meaningful "
        "state machine."
    ),
    category=CalibrationCategory.FOUNDATIONAL,
    properties=(
        JurisdictionalProperty.DETERMINISM,
        JurisdictionalProperty.AUDITABILITY,
        JurisdictionalProperty.REVERSIBILITY,
        JurisdictionalProperty.SOVEREIGNTY,
        JurisdictionalProperty.PRIVACY,
        JurisdictionalProperty.OVERSIGHT,
    ),
)

# Axiom 2: QRADLE as Enabling Substrate
AXIOM_2_QRADLE = _create_axiom(
    axiom_id=2,
    name="QRADLE as Enabling Substrate for Regulated Realms",
    description=(
        "QRADLE transcends ledgers; it's a deterministic foundation absorbing "
        "regulation into semantics. Compile constraints into intent languages, "
        "enforce as fatal invariants, and audit as cryptographic facts. This "
        "virtualizes jurisdiction, turning compliance from metadata overlay into "
        "execution primitive—essential for genomics, medical AI, and constrained "
        "ASI where errors are ontologically irreversible."
    ),
    category=CalibrationCategory.FOUNDATIONAL,
    properties=(
        JurisdictionalProperty.DETERMINISM,
        JurisdictionalProperty.AUDITABILITY,
    ),
)

# Axiom 3: Verticals as Constraint-Driven Stress Tests
AXIOM_3_VERTICALS = _create_axiom(
    axiom_id=3,
    name="Verticals as Constraint-Driven Stress Tests",
    description=(
        "The 14 verticals (VITRA-E0 genomics as anchor, extending to JURIS legal, "
        "ECORA climate, etc.) are not sprawl but boundary conditions testing safety "
        "under high-entropy, high-regulation domains. Genomics teaches irreversibility "
        "(leaked data unrecallable, misinterpretations life-altering), forcing "
        "reversibility/auditability maturation that subsidizes all others. If QRATUM "
        "survives genomics, it governs finance, defense, or orbital systems structurally."
    ),
    category=CalibrationCategory.OPERATIONAL,
    properties=(
        JurisdictionalProperty.REVERSIBILITY,
        JurisdictionalProperty.AUDITABILITY,
    ),
)

# Axiom 4: Defensive Engine as Inflection to Self-Stabilization
AXIOM_4_DEFENSIVE = _create_axiom(
    axiom_id=4,
    name="Defensive Engine as Inflection to Self-Stabilization",
    description=(
        "The vulnerability discovery engine elevates QRATUM from secure stack to "
        "dynamical system monitoring trajectories toward failure (entropy gradients, "
        "metastable clusters, collapse precursors). Shift from debugging incidents to "
        "preempting fragility—foundational for ASI safety, enabling conditional "
        "self-suspension in recursive loops."
    ),
    category=CalibrationCategory.OPERATIONAL,
    properties=(
        JurisdictionalProperty.OVERSIGHT,
        JurisdictionalProperty.SOVEREIGNTY,
    ),
)

# Axiom 5: ASI Scaffolding - Jurisdiction-First Legitimacy
AXIOM_5_ASI = _create_axiom(
    axiom_id=5,
    name="ASI Scaffolding: Jurisdiction-First Legitimacy",
    description=(
        "QRATUM ASI inverts discourse—start with unbreakable primitives (invariants, "
        "zone isolation, dual-control, rollback) before intelligence. This disqualifies "
        "unconstrained approaches: systems unable to prove impossibility of oversight "
        "removal or invariant bypass are structurally illegitimate for autonomy. "
        "Speculative yet load-bearing, it's the only viable order for bounded self-improvement."
    ),
    category=CalibrationCategory.FOUNDATIONAL,
    properties=(
        JurisdictionalProperty.OVERSIGHT,
        JurisdictionalProperty.DETERMINISM,
        JurisdictionalProperty.SOVEREIGNTY,
    ),
)

# Axiom 6: Adoption Reality
AXIOM_6_ADOPTION = _create_axiom(
    axiom_id=6,
    name="Adoption Reality: Ahead of Narrative, Not Flawed",
    description=(
        "Low visibility (deep repo, no ecosystem) stems from category error—building "
        "executable institutions, not tools. This is natural for paradigm shifts; "
        "QRATUM appears niche because it's pre-crisis necessity. Quiet depth allows "
        "unencumbered maturation."
    ),
    category=CalibrationCategory.STRATEGIC,
    properties=(),
)

# Axiom 7: Accidental Regulatory Primitive Design
AXIOM_7_REGULATORY = _create_axiom(
    axiom_id=7,
    name="Accidental Regulatory Primitive Design",
    description=(
        "QRATUM absorbs law into semantics, compiling regulations into intents/invariants "
        "and auditing as facts—not automation, but jurisdiction virtualization. This "
        "sustains a decade-long research program, even sans ASI, by making trust a "
        "first-order invariant."
    ),
    category=CalibrationCategory.STRATEGIC,
    properties=(
        JurisdictionalProperty.AUDITABILITY,
        JurisdictionalProperty.DETERMINISM,
    ),
)

# Axiom 8: Genomics as Ontological Consequence Trainer
AXIOM_8_GENOMICS = _create_axiom(
    axiom_id=8,
    name="Genomics as Ontological Consequence Trainer",
    description=(
        "Beyond stress-testing, genomics embeds hardest irreversibility (social/biological "
        "permanence), training QRATUM toward jurisdictional convergence over reliability—"
        "honing primitives for all domains."
    ),
    category=CalibrationCategory.OPERATIONAL,
    properties=(
        JurisdictionalProperty.REVERSIBILITY,
        JurisdictionalProperty.PRIVACY,
    ),
)

# Axiom 9: Epistemological Shift via Vulnerability Engine
AXIOM_9_EPISTEMOLOGY = _create_axiom(
    axiom_id=9,
    name="Epistemological Shift via Vulnerability Engine",
    description=(
        "Evolve from state-awareness (what happened/rollback) to trajectory-awareness "
        "(what it's becoming). Model entropy/coupling/precursors for reasoning about "
        "fragility, enabling blind acceleration to become self-suspending—minimal for "
        "hosting improvement."
    ),
    category=CalibrationCategory.EPISTEMOLOGICAL,
    properties=(JurisdictionalProperty.OVERSIGHT,),
)

# Axiom 10: Jurisdiction-First as Gating Disqualifier
AXIOM_10_GATING = _create_axiom(
    axiom_id=10,
    name="Jurisdiction-First as Gating Disqualifier",
    description=(
        "For ASI, prove uncontrollability of oversight/invariant alteration, or the "
        "system is illegitimate. QRATUM defines the containment class for deployable "
        "superintelligence—not building it now, but specifying legal-grade envelopes."
    ),
    category=CalibrationCategory.FOUNDATIONAL,
    properties=(
        JurisdictionalProperty.OVERSIGHT,
        JurisdictionalProperty.SOVEREIGNTY,
    ),
)

# Axiom 11: Invisibility as Category Error Advantage
AXIOM_11_INVISIBILITY = _create_axiom(
    axiom_id=11,
    name="Invisibility as Category Error Advantage",
    description=(
        "QRATUM builds institutions in code, unrecognized until crisis mandates them. "
        "Exploit this for quiet positioning; adoption won't be incremental but enforced."
    ),
    category=CalibrationCategory.STRATEGIC,
    properties=(),
)

# Axiom 12: Long-Term Ontology Change
AXIOM_12_ONTOLOGY = _create_axiom(
    axiom_id=12,
    name="Long-Term Ontology Change",
    description=(
        "QRATUM redefines computation—trust as physical law, not assumption. It bifurcates "
        "systems into trust-bearing (high-stakes) and best-effort (low-consequence), "
        "emerging as the first to treat sovereignty/failure-forecasting as invariants."
    ),
    category=CalibrationCategory.EPISTEMOLOGICAL,
    properties=(
        JurisdictionalProperty.SOVEREIGNTY,
        JurisdictionalProperty.DETERMINISM,
    ),
)

# Complete doctrine collection
CALIBRATION_DOCTRINE: Tuple[CalibrationAxiom, ...] = (
    AXIOM_1_JURISDICTION,
    AXIOM_2_QRADLE,
    AXIOM_3_VERTICALS,
    AXIOM_4_DEFENSIVE,
    AXIOM_5_ASI,
    AXIOM_6_ADOPTION,
    AXIOM_7_REGULATORY,
    AXIOM_8_GENOMICS,
    AXIOM_9_EPISTEMOLOGY,
    AXIOM_10_GATING,
    AXIOM_11_INVISIBILITY,
    AXIOM_12_ONTOLOGY,
)


class CalibrationDoctrineEnforcer:
    """Enforces the 12 Calibration Doctrine across QRATUM operations.

    This class validates that all operations comply with the foundational
    axioms and maintains trajectory-awareness for defensive posture.
    """

    # Configurable limits for trajectory history management
    MAX_TRAJECTORY_HISTORY: int = 1000  # Maximum trajectory samples to retain
    TRAJECTORY_HISTORY_PRUNE_SIZE: int = 500  # Size after pruning

    def __init__(
        self,
        max_trajectory_history: Optional[int] = None,
        trajectory_prune_size: Optional[int] = None,
    ):
        self.doctrine = CALIBRATION_DOCTRINE
        self.trajectory_history: List[TrajectoryMetrics] = []
        self.jurisdictional_claims: List[JurisdictionalClaim] = []
        self._verified = False

        # Allow configurable limits
        self._max_history = max_trajectory_history or self.MAX_TRAJECTORY_HISTORY
        self._prune_size = trajectory_prune_size or self.TRAJECTORY_HISTORY_PRUNE_SIZE

    def verify_doctrine_integrity(self) -> Dict[str, Any]:
        """Verify all axioms have not been tampered with.

        Returns:
            Dictionary with verification status and details.
        """
        results = {
            "verified": True,
            "axiom_count": len(self.doctrine),
            "failed_axioms": [],
            "timestamp": datetime.utcnow().isoformat(),
        }

        for axiom in self.doctrine:
            if not axiom.verify_integrity():
                results["verified"] = False
                results["failed_axioms"].append(axiom.axiom_id)

        self._verified = results["verified"]
        return results

    def validate_operation_compliance(
        self, operation_type: str, required_properties: List[JurisdictionalProperty]
    ) -> Tuple[bool, List[str]]:
        """Validate an operation complies with doctrine.

        Args:
            operation_type: Type of operation being performed
            required_properties: Properties the operation must satisfy

        Returns:
            Tuple of (is_compliant, list of violated axioms)
        """
        violations = []

        # Check each required property is covered by at least one axiom
        for prop in required_properties:
            covered = False
            for axiom in self.doctrine:
                if prop in axiom.properties:
                    covered = True
                    break

            if not covered:
                violations.append(
                    f"Property {prop.value} not covered by doctrine for {operation_type}"
                )

        # Specific axiom enforcement
        if "genomic" in operation_type.lower():
            # Axiom 8: Genomics requires reversibility and privacy
            if JurisdictionalProperty.REVERSIBILITY not in required_properties:
                violations.append("Axiom 8: Genomic operations require reversibility")
            if JurisdictionalProperty.PRIVACY not in required_properties:
                violations.append("Axiom 8: Genomic operations require privacy")

        if "asi" in operation_type.lower() or "self_improvement" in operation_type.lower():
            # Axiom 5: ASI requires jurisdiction-first legitimacy
            if JurisdictionalProperty.OVERSIGHT not in required_properties:
                violations.append("Axiom 5: ASI operations require embedded oversight")
            if JurisdictionalProperty.DETERMINISM not in required_properties:
                violations.append("Axiom 5: ASI operations require determinism")

        return len(violations) == 0, violations

    def record_trajectory(self, metrics: TrajectoryMetrics) -> None:
        """Record trajectory metrics for defensive posture.

        Implements Axiom 9: Epistemological shift to trajectory-awareness.
        """
        self.trajectory_history.append(metrics)

        # Maintain bounded history using configurable limits
        if len(self.trajectory_history) > self._max_history:
            self.trajectory_history = self.trajectory_history[-self._prune_size :]

    def assess_trajectory_state(self) -> TrajectoryState:
        """Assess current trajectory state based on recent metrics.

        Implements Axiom 4: Defensive engine for self-stabilization.
        """
        if not self.trajectory_history:
            return TrajectoryState.STABLE

        recent = (
            self.trajectory_history[-10:]
            if len(self.trajectory_history) >= 10
            else self.trajectory_history
        )

        # Check for critical conditions
        critical_count = sum(1 for m in recent if m.trajectory_state == TrajectoryState.CRITICAL)
        precursor_count = sum(1 for m in recent if m.trajectory_state == TrajectoryState.PRECURSOR)

        if critical_count > 0:
            return TrajectoryState.CRITICAL
        if precursor_count >= 3:
            return TrajectoryState.PRECURSOR

        # Check entropy gradient trend
        avg_entropy_gradient = sum(m.entropy_gradient for m in recent) / len(recent)
        if avg_entropy_gradient > 0.5:
            return TrajectoryState.METASTABLE
        if avg_entropy_gradient > 0.2:
            return TrajectoryState.DRIFT

        return TrajectoryState.STABLE

    def should_self_suspend(self) -> Tuple[bool, str]:
        """Determine if system should conditionally self-suspend.

        Implements Axiom 4: Conditional self-suspension in recursive loops.

        Returns:
            Tuple of (should_suspend, reason)
        """
        current_state = self.assess_trajectory_state()

        if current_state == TrajectoryState.CRITICAL:
            return True, "Critical trajectory state detected - system stability at risk"

        if current_state == TrajectoryState.SELF_SUSPEND:
            return True, "Self-suspension triggered by defensive engine"

        # Check recent trajectory metrics
        if self.trajectory_history:
            recent = self.trajectory_history[-1]
            if recent.should_self_suspend():
                return (
                    True,
                    f"Trajectory metrics indicate self-suspension: {recent.collapse_precursors} precursors detected",
                )

        return False, "System operating within safe trajectory bounds"

    def create_jurisdictional_claim(
        self,
        claim_type: str,
        subject: str,
        predicate: str,
        evidence: bytes,
        merkle_proof: Optional[str] = None,
    ) -> JurisdictionalClaim:
        """Create a jurisdictional claim with cryptographic evidence.

        Implements Axiom 1: Proving "what happened" and "what could not have happened".
        """
        evidence_hash = hashlib.sha3_256(evidence).hexdigest()

        claim = JurisdictionalClaim(
            claim_id=hashlib.sha3_256(
                f"{claim_type}:{subject}:{predicate}:{evidence_hash}".encode()
            ).hexdigest()[:16],
            claim_type=claim_type,
            subject=subject,
            predicate=predicate,
            evidence_hash=evidence_hash,
            timestamp=datetime.utcnow().isoformat(),
            merkle_proof=merkle_proof,
        )

        self.jurisdictional_claims.append(claim)
        return claim

    def prove_impossibility(
        self, action: str, merkle_root: str, zone_constraints: Dict[str, Any]
    ) -> JurisdictionalClaim:
        """Prove that an action could not have happened.

        Implements Axiom 1: Proving "what could not have happened".
        """
        evidence = json.dumps(
            {
                "action": action,
                "merkle_root": merkle_root,
                "zone_constraints": zone_constraints,
                "timestamp": datetime.utcnow().isoformat(),
            }
        ).encode()

        return self.create_jurisdictional_claim(
            claim_type="could_not_happen",
            subject=action,
            predicate=f"Action '{action}' was impossible given zone constraints",
            evidence=evidence,
            merkle_proof=merkle_root,
        )

    def validate_asi_legitimacy(self, system_properties: Dict[str, bool]) -> Tuple[bool, List[str]]:
        """Validate ASI system legitimacy per Axiom 10.

        Systems must prove uncontrollability of oversight/invariant alteration.

        Args:
            system_properties: Dict of property names to enforcement status

        Returns:
            Tuple of (is_legitimate, list of failures)
        """
        required_proofs = [
            "cannot_remove_oversight",
            "cannot_bypass_invariants",
            "cannot_disable_rollback",
            "cannot_alter_merkle_chain",
            "cannot_forge_authorization",
        ]

        failures = []
        for proof in required_proofs:
            if not system_properties.get(proof, False):
                failures.append(f"Missing proof: {proof}")

        return len(failures) == 0, failures

    def get_doctrine_summary(self) -> Dict[str, Any]:
        """Get summary of doctrine status.

        Returns:
            Dictionary with doctrine summary information.
        """
        integrity = self.verify_doctrine_integrity()
        trajectory_state = self.assess_trajectory_state()
        should_suspend, suspend_reason = self.should_self_suspend()

        return {
            "doctrine_version": "1.0.0",
            "axiom_count": len(self.doctrine),
            "integrity_verified": integrity["verified"],
            "trajectory_state": trajectory_state.value,
            "should_self_suspend": should_suspend,
            "suspend_reason": suspend_reason,
            "claims_recorded": len(self.jurisdictional_claims),
            "trajectory_samples": len(self.trajectory_history),
            "timestamp": datetime.utcnow().isoformat(),
            "axioms": [
                {
                    "id": a.axiom_id,
                    "name": a.name,
                    "category": a.category.value,
                    "properties": [p.value for p in a.properties],
                    "integrity_valid": a.verify_integrity(),
                }
                for a in self.doctrine
            ],
        }


# Singleton instance for global doctrine enforcement
_doctrine_enforcer: Optional[CalibrationDoctrineEnforcer] = None


def get_doctrine_enforcer() -> CalibrationDoctrineEnforcer:
    """Get singleton doctrine enforcer instance."""
    global _doctrine_enforcer
    if _doctrine_enforcer is None:
        _doctrine_enforcer = CalibrationDoctrineEnforcer()
    return _doctrine_enforcer
