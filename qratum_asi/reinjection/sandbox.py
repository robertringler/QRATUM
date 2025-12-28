"""Sandbox Orchestration for Reinjection Testing.

Implements Z1 sandbox environment for safe testing of discovery reinjections
before Z2 commitment. Includes rollback testing and side effect detection.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from qradle.merkle import MerkleChain
from qratum_asi.reinjection.mapper import DiscoveryPriorMapper, MappingResult
from qratum_asi.reinjection.types import (
    ReinjectionCandidate,
    SandboxResult,
)


@dataclass
class SandboxState:
    """State of a sandbox session.

    Attributes:
        sandbox_id: Unique sandbox identifier
        candidate_id: Candidate being tested
        state_snapshot: Snapshot of prior state before testing
        applied_mappings: Mappings applied in sandbox
        detected_side_effects: Side effects detected during testing
        is_active: Whether sandbox is active
    """

    sandbox_id: str
    candidate_id: str
    state_snapshot: dict[str, Any]
    applied_mappings: list[str]
    detected_side_effects: list[str]
    is_active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize sandbox state."""
        return {
            "sandbox_id": self.sandbox_id,
            "candidate_id": self.candidate_id,
            "state_snapshot_hash": hashlib.sha3_256(
                json.dumps(self.state_snapshot, sort_keys=True).encode()
            ).hexdigest(),
            "applied_mappings": self.applied_mappings,
            "detected_side_effects": self.detected_side_effects,
            "is_active": self.is_active,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


@dataclass
class RollbackTest:
    """Result of a rollback test.

    Attributes:
        test_id: Unique test identifier
        sandbox_id: Sandbox where test was performed
        rollback_successful: Whether rollback succeeded
        state_restored: Whether original state was fully restored
        restore_fidelity: Fidelity of state restoration (0-1)
        execution_time_ms: Time to perform rollback
    """

    test_id: str
    sandbox_id: str
    rollback_successful: bool
    state_restored: bool
    restore_fidelity: float
    execution_time_ms: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize rollback test."""
        return {
            "test_id": self.test_id,
            "sandbox_id": self.sandbox_id,
            "rollback_successful": self.rollback_successful,
            "state_restored": self.state_restored,
            "restore_fidelity": self.restore_fidelity,
            "execution_time_ms": self.execution_time_ms,
            "timestamp": self.timestamp,
            "errors": self.errors,
        }


class SandboxOrchestrator:
    """Orchestrates sandbox testing for reinjection candidates.

    Provides Z1 sandbox environment with:
    - Isolated state management
    - Mapping application testing
    - Side effect detection
    - Rollback verification
    - Fidelity measurement
    """

    def __init__(self, merkle_chain: MerkleChain | None = None):
        """Initialize sandbox orchestrator.

        Args:
            merkle_chain: Optional Merkle chain for provenance
        """
        self.merkle_chain = merkle_chain or MerkleChain()
        self.active_sandboxes: dict[str, SandboxState] = {}
        self.completed_sandboxes: list[SandboxState] = []
        self._sandbox_counter = 0

        # Create isolated mapper for sandbox testing
        self._sandbox_mapper = DiscoveryPriorMapper(merkle_chain=MerkleChain())

    def create_sandbox(self, candidate: ReinjectionCandidate) -> SandboxState:
        """Create a new sandbox session for testing.

        Args:
            candidate: Candidate to test

        Returns:
            Created SandboxState
        """
        self._sandbox_counter += 1
        sandbox_id = f"sb_{candidate.candidate_id}_{self._sandbox_counter:04d}"

        # Snapshot current state
        state_snapshot = self._capture_state_snapshot()

        sandbox = SandboxState(
            sandbox_id=sandbox_id,
            candidate_id=candidate.candidate_id,
            state_snapshot=state_snapshot,
            applied_mappings=[],
            detected_side_effects=[],
        )

        self.active_sandboxes[sandbox_id] = sandbox

        # Log sandbox creation
        self.merkle_chain.add_event(
            "sandbox_created",
            {
                "sandbox_id": sandbox_id,
                "candidate_id": candidate.candidate_id,
            },
        )

        return sandbox

    def run_sandbox_test(
        self,
        candidate: ReinjectionCandidate,
        mapping_result: MappingResult,
    ) -> SandboxResult:
        """Run a complete sandbox test for a candidate.

        Args:
            candidate: Candidate to test
            mapping_result: Mapping result to apply

        Returns:
            SandboxResult with test outcomes
        """
        start_time = time.perf_counter()

        # Create sandbox
        sandbox = self.create_sandbox(candidate)
        side_effects: list[str] = []

        try:
            # Apply mapping in sandbox
            success = self._apply_mapping_in_sandbox(sandbox, mapping_result)

            if not success:
                side_effects.append("Mapping application failed")

            # Detect side effects
            detected_effects = self._detect_side_effects(sandbox, mapping_result)
            side_effects.extend(detected_effects)
            sandbox.detected_side_effects = side_effects

            # Measure fidelity improvement
            fidelity_score = self._measure_fidelity(sandbox, mapping_result)

            # Test rollback capability
            rollback_test = self._test_rollback(sandbox)
            rollback_tested = rollback_test.rollback_successful and rollback_test.state_restored

            # Close sandbox
            self._close_sandbox(sandbox.sandbox_id)

            execution_time_ms = (time.perf_counter() - start_time) * 1000

            result = SandboxResult(
                sandbox_id=sandbox.sandbox_id,
                candidate_id=candidate.candidate_id,
                success=success and len(side_effects) == 0,
                fidelity_score=fidelity_score,
                rollback_tested=rollback_tested,
                side_effects=side_effects,
                execution_time_ms=execution_time_ms,
                metrics={
                    "rollback_fidelity": rollback_test.restore_fidelity,
                    "priors_tested": len(mapping_result.prior_updates),
                },
            )

            # Log test completion
            self.merkle_chain.add_event(
                "sandbox_test_completed",
                {
                    "sandbox_id": sandbox.sandbox_id,
                    "success": result.success,
                    "fidelity_score": fidelity_score,
                },
            )

            return result

        except Exception as e:
            # Ensure sandbox is closed on error
            self._close_sandbox(sandbox.sandbox_id)
            execution_time_ms = (time.perf_counter() - start_time) * 1000

            return SandboxResult(
                sandbox_id=sandbox.sandbox_id,
                candidate_id=candidate.candidate_id,
                success=False,
                fidelity_score=0.0,
                rollback_tested=False,
                side_effects=[f"Sandbox error: {str(e)}"],
                execution_time_ms=execution_time_ms,
            )

    def _apply_mapping_in_sandbox(
        self,
        sandbox: SandboxState,
        mapping_result: MappingResult,
    ) -> bool:
        """Apply mapping in sandbox environment."""
        try:
            success = self._sandbox_mapper.apply_mapping(mapping_result)
            if success:
                sandbox.applied_mappings.append(mapping_result.mapping_id)
            return success
        except Exception:
            return False

    def _detect_side_effects(
        self,
        sandbox: SandboxState,
        mapping_result: MappingResult,
    ) -> list[str]:
        """Detect any side effects from applying the mapping."""
        side_effects: list[str] = []

        # Check for unexpected prior changes
        for update in mapping_result.prior_updates:
            # Check if change is within expected bounds
            if update.relative_change > 0.5:  # More than 50% change
                side_effects.append(
                    f"Large prior change detected: {update.prior_id} "
                    f"changed by {update.relative_change:.1%}"
                )

        # Check cross-vertical impacts
        for vertical, impact in mapping_result.cross_vertical_impacts.items():
            if impact > 0.8:  # High cross-vertical impact
                side_effects.append(f"High cross-vertical impact on {vertical}: {impact:.2f}")

        return side_effects

    def _measure_fidelity(
        self,
        sandbox: SandboxState,
        mapping_result: MappingResult,
    ) -> float:
        """Measure fidelity improvement from mapping."""
        if not mapping_result.prior_updates:
            return 0.0

        # Fidelity is based on confidence improvement and update consistency
        avg_confidence = mapping_result.average_confidence_improvement

        # Normalize to 0-1 scale
        # Base fidelity is proportional to confidence improvement
        base_fidelity = min(avg_confidence / 0.1, 1.0)  # 10% improvement = 1.0 fidelity

        # Penalize for side effects
        penalty = len(sandbox.detected_side_effects) * 0.1
        fidelity = max(0.0, base_fidelity - penalty)

        return round(fidelity, 4)

    def _test_rollback(self, sandbox: SandboxState) -> RollbackTest:
        """Test rollback capability in sandbox."""
        test_id = f"rb_test_{sandbox.sandbox_id}"
        start_time = time.perf_counter()
        errors: list[str] = []

        # Attempt to rollback all applied mappings
        rollback_successful = True
        for mapping_id in sandbox.applied_mappings:
            if not self._sandbox_mapper.rollback_mapping(mapping_id):
                rollback_successful = False
                errors.append(f"Failed to rollback mapping: {mapping_id}")

        # Verify state restoration
        current_state = self._capture_state_snapshot()
        state_restored, restore_fidelity = self._compare_states(
            sandbox.state_snapshot, current_state
        )

        execution_time_ms = (time.perf_counter() - start_time) * 1000

        return RollbackTest(
            test_id=test_id,
            sandbox_id=sandbox.sandbox_id,
            rollback_successful=rollback_successful,
            state_restored=state_restored,
            restore_fidelity=restore_fidelity,
            execution_time_ms=execution_time_ms,
            errors=errors,
        )

    def _capture_state_snapshot(self) -> dict[str, Any]:
        """Capture current state snapshot."""
        # In a real implementation, this would capture the complete state
        # For now, return mapper stats as proxy
        return {
            "mapper_stats": self._sandbox_mapper.get_mapping_stats(),
            "prior_database": dict(self._sandbox_mapper._prior_database),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _compare_states(
        self,
        original: dict[str, Any],
        current: dict[str, Any],
    ) -> tuple[bool, float]:
        """Compare two state snapshots.

        Returns:
            Tuple of (states_match, fidelity_score)
        """
        # Compare prior databases
        original_priors = original.get("prior_database", {})
        current_priors = current.get("prior_database", {})

        if not original_priors and not current_priors:
            return True, 1.0

        if set(original_priors.keys()) != set(current_priors.keys()):
            # Different keys = imperfect restoration
            key_match = len(set(original_priors.keys()) & set(current_priors.keys())) / max(
                len(original_priors), len(current_priors), 1
            )
            return False, key_match

        # Compare values
        total_diff = 0.0
        for key in original_priors:
            orig_val = original_priors[key].get("value", 0)
            curr_val = current_priors.get(key, {}).get("value", 0)
            total_diff += abs(orig_val - curr_val)

        avg_diff = total_diff / len(original_priors) if original_priors else 0
        fidelity = max(0.0, 1.0 - avg_diff)

        return fidelity > 0.99, fidelity

    def _close_sandbox(self, sandbox_id: str) -> bool:
        """Close and archive a sandbox session."""
        if sandbox_id not in self.active_sandboxes:
            return False

        sandbox = self.active_sandboxes.pop(sandbox_id)
        sandbox.is_active = False
        sandbox.completed_at = datetime.now(timezone.utc).isoformat()
        self.completed_sandboxes.append(sandbox)

        # Reset sandbox mapper
        self._sandbox_mapper = DiscoveryPriorMapper(merkle_chain=MerkleChain())

        # Log closure
        self.merkle_chain.add_event(
            "sandbox_closed",
            {
                "sandbox_id": sandbox_id,
                "side_effects_detected": len(sandbox.detected_side_effects),
            },
        )

        return True

    def get_sandbox_stats(self) -> dict[str, Any]:
        """Get sandbox statistics."""
        return {
            "active_sandboxes": len(self.active_sandboxes),
            "completed_sandboxes": len(self.completed_sandboxes),
            "total_sessions": self._sandbox_counter,
        }
