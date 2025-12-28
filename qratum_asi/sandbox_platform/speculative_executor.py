"""Speculative Executor for Pre-running High-Likelihood Proposals.

Implements speculative execution to pre-run proposals that are likely
to be approved, reducing latency when approval comes.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from qradle.merkle import MerkleChain

from qratum_asi.sandbox_platform.types import SandboxProposal, SandboxEvaluationResult


class SpeculativeStatus(Enum):
    """Status of speculative execution."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    VALIDATED = "validated"
    DISCARDED = "discarded"
    FAILED = "failed"


@dataclass
class LikelihoodEstimate:
    """Estimate of proposal approval likelihood.

    Attributes:
        estimate_id: Unique estimate identifier
        proposal_id: Estimated proposal ID
        likelihood: Estimated likelihood (0-1)
        confidence: Confidence in estimate (0-1)
        factors: Factors contributing to estimate
    """

    estimate_id: str
    proposal_id: str
    likelihood: float
    confidence: float
    factors: dict[str, float]
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize estimate."""
        return {
            "estimate_id": self.estimate_id,
            "proposal_id": self.proposal_id,
            "likelihood": self.likelihood,
            "confidence": self.confidence,
            "factors": self.factors,
            "timestamp": self.timestamp,
        }


@dataclass
class SpeculativeResult:
    """Result of speculative execution.

    Attributes:
        result_id: Unique result identifier
        proposal_id: Executed proposal ID
        status: Speculative execution status
        evaluation_result: Evaluation result if completed
        execution_time_ms: Execution time
        was_used: Whether result was actually used
        time_saved_ms: Time saved by speculative execution
    """

    result_id: str
    proposal_id: str
    status: SpeculativeStatus
    evaluation_result: SandboxEvaluationResult | None = None
    execution_time_ms: float = 0.0
    was_used: bool = False
    time_saved_ms: float = 0.0
    likelihood_estimate: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    discarded_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize speculative result."""
        return {
            "result_id": self.result_id,
            "proposal_id": self.proposal_id,
            "status": self.status.value,
            "has_evaluation_result": self.evaluation_result is not None,
            "execution_time_ms": self.execution_time_ms,
            "was_used": self.was_used,
            "time_saved_ms": self.time_saved_ms,
            "likelihood_estimate": self.likelihood_estimate,
            "timestamp": self.timestamp,
            "discarded_reason": self.discarded_reason,
        }


class LikelihoodEstimator:
    """Estimator for proposal approval likelihood.

    Uses historical data and proposal features to estimate
    the likelihood that a proposal will be approved.
    """

    def __init__(
        self,
        estimator_id: str = "likelihood",
        default_threshold: float = 0.7,
    ):
        """Initialize likelihood estimator.

        Args:
            estimator_id: Unique estimator identifier
            default_threshold: Default threshold for speculative execution
        """
        self.estimator_id = estimator_id
        self.default_threshold = default_threshold

        # Historical data
        self._approval_history: dict[str, bool] = {}  # proposal_type -> approved
        self._estimate_counter = 0

        # Feature weights
        self._weights = {
            "priority": 0.25,
            "impact": 0.20,
            "historical": 0.30,
            "source_reputation": 0.25,
        }

    def estimate(self, proposal: SandboxProposal) -> LikelihoodEstimate:
        """Estimate approval likelihood for a proposal.

        Args:
            proposal: Proposal to estimate

        Returns:
            LikelihoodEstimate
        """
        self._estimate_counter += 1
        estimate_id = f"est_{self.estimator_id}_{self._estimate_counter:08d}"

        factors: dict[str, float] = {}

        # Factor: Priority
        priority_likelihood = {
            "critical": 0.95,
            "high": 0.85,
            "normal": 0.70,
            "low": 0.50,
        }
        factors["priority"] = priority_likelihood.get(proposal.priority.value, 0.70)

        # Factor: Impact (higher impact = more scrutiny = lower likelihood)
        factors["impact"] = 1.0 - (proposal.estimated_impact * 0.3)

        # Factor: Historical approval rate for proposal type
        historical_rate = self._get_historical_rate(proposal.proposal_type)
        factors["historical"] = historical_rate

        # Factor: Source reputation (simplified)
        factors["source_reputation"] = 0.8  # Default good reputation

        # Compute weighted likelihood
        likelihood = sum(
            factors[k] * self._weights[k] for k in factors
        )
        likelihood = max(0.0, min(1.0, likelihood))

        # Confidence based on historical data availability
        confidence = 0.7 if proposal.proposal_type in self._approval_history else 0.5

        return LikelihoodEstimate(
            estimate_id=estimate_id,
            proposal_id=proposal.proposal_id,
            likelihood=likelihood,
            confidence=confidence,
            factors=factors,
        )

    def _get_historical_rate(self, proposal_type: str) -> float:
        """Get historical approval rate for proposal type."""
        if proposal_type in self._approval_history:
            return 0.9 if self._approval_history[proposal_type] else 0.3
        return 0.7  # Default rate

    def record_outcome(self, proposal_type: str, approved: bool) -> None:
        """Record the outcome of a proposal for learning.

        Args:
            proposal_type: Type of proposal
            approved: Whether it was approved
        """
        self._approval_history[proposal_type] = approved


class SpeculativeExecutor:
    """Executor that speculatively pre-runs high-likelihood proposals.

    Pre-executes proposals that are likely to be approved,
    caching results to reduce latency when approval comes.
    """

    def __init__(
        self,
        executor_id: str = "speculative",
        likelihood_threshold: float = 0.7,
        max_speculative: int = 100,
        merkle_chain: MerkleChain | None = None,
    ):
        """Initialize speculative executor.

        Args:
            executor_id: Unique executor identifier
            likelihood_threshold: Minimum likelihood for speculative execution
            max_speculative: Maximum cached speculative results
            merkle_chain: Merkle chain for audit trail
        """
        self.executor_id = executor_id
        self.likelihood_threshold = likelihood_threshold
        self.max_speculative = max_speculative
        self.merkle_chain = merkle_chain or MerkleChain()

        # Likelihood estimator
        self.estimator = LikelihoodEstimator()

        # Speculative results cache
        self._cache: dict[str, SpeculativeResult] = {}
        self._result_counter = 0
        self._lock = threading.RLock()

        # Background execution
        self._pending_proposals: list[SandboxProposal] = []
        self._executor_func: Callable[[SandboxProposal], SandboxEvaluationResult] | None = None

        # Statistics
        self._total_speculative = 0
        self._used_results = 0
        self._discarded_results = 0
        self._total_time_saved_ms = 0.0

        # Log initialization
        self.merkle_chain.add_event(
            "speculative_executor_initialized",
            {
                "executor_id": executor_id,
                "likelihood_threshold": likelihood_threshold,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
        )

    def set_executor(
        self,
        executor: Callable[[SandboxProposal], SandboxEvaluationResult],
    ) -> None:
        """Set the proposal executor function.

        Args:
            executor: Function to execute proposals
        """
        self._executor_func = executor

    def submit_speculative(
        self,
        proposal: SandboxProposal,
    ) -> SpeculativeResult | None:
        """Submit a proposal for speculative execution.

        Args:
            proposal: Proposal to speculatively execute

        Returns:
            SpeculativeResult if submitted, None if below threshold
        """
        # Estimate likelihood
        estimate = self.estimator.estimate(proposal)

        if estimate.likelihood < self.likelihood_threshold:
            return None

        # Check cache limit
        with self._lock:
            if len(self._cache) >= self.max_speculative:
                self._evict_oldest()

            self._result_counter += 1
            result_id = f"spec_{self.executor_id}_{self._result_counter:08d}"

            result = SpeculativeResult(
                result_id=result_id,
                proposal_id=proposal.proposal_id,
                status=SpeculativeStatus.PENDING,
                likelihood_estimate=estimate.likelihood,
            )

            self._cache[proposal.proposal_id] = result
            self._total_speculative += 1

        # Execute speculatively
        self._execute_speculative(proposal, result)

        return result

    def _execute_speculative(
        self,
        proposal: SandboxProposal,
        result: SpeculativeResult,
    ) -> None:
        """Execute proposal speculatively."""
        if self._executor_func is None:
            result.status = SpeculativeStatus.FAILED
            return

        result.status = SpeculativeStatus.RUNNING
        start_time = time.perf_counter()

        try:
            eval_result = self._executor_func(proposal)
            result.evaluation_result = eval_result
            result.status = SpeculativeStatus.COMPLETED
            result.execution_time_ms = (time.perf_counter() - start_time) * 1000

            # Log speculative execution
            self.merkle_chain.add_event(
                "speculative_execution_completed",
                {
                    "result_id": result.result_id,
                    "proposal_id": proposal.proposal_id,
                    "likelihood": result.likelihood_estimate,
                },
            )

        except Exception as e:
            result.status = SpeculativeStatus.FAILED
            result.discarded_reason = str(e)

    def get_speculative_result(
        self,
        proposal_id: str,
    ) -> SpeculativeResult | None:
        """Get cached speculative result for a proposal.

        Args:
            proposal_id: Proposal ID to look up

        Returns:
            SpeculativeResult if available
        """
        return self._cache.get(proposal_id)

    def validate_result(
        self,
        proposal_id: str,
    ) -> SandboxEvaluationResult | None:
        """Validate and retrieve a speculative result.

        Called when proposal is actually approved to use the
        pre-computed result.

        Args:
            proposal_id: Proposal ID

        Returns:
            Evaluation result if available and valid
        """
        with self._lock:
            if proposal_id not in self._cache:
                return None

            result = self._cache[proposal_id]

            if result.status != SpeculativeStatus.COMPLETED:
                return None

            # Mark as used
            result.status = SpeculativeStatus.VALIDATED
            result.was_used = True
            result.time_saved_ms = result.execution_time_ms
            self._used_results += 1
            self._total_time_saved_ms += result.time_saved_ms

            # Record outcome for learning
            self.estimator.record_outcome(
                result.evaluation_result.proposal_id if result.evaluation_result else "",
                True,
            )

            # Log validation
            self.merkle_chain.add_event(
                "speculative_result_validated",
                {
                    "result_id": result.result_id,
                    "proposal_id": proposal_id,
                    "time_saved_ms": result.time_saved_ms,
                },
            )

            return result.evaluation_result

    def discard_result(
        self,
        proposal_id: str,
        reason: str = "proposal_rejected",
    ) -> bool:
        """Discard a speculative result.

        Called when proposal is rejected, discarding the
        pre-computed result.

        Args:
            proposal_id: Proposal ID
            reason: Reason for discarding

        Returns:
            True if result was discarded
        """
        with self._lock:
            if proposal_id not in self._cache:
                return False

            result = self._cache[proposal_id]
            result.status = SpeculativeStatus.DISCARDED
            result.discarded_reason = reason
            self._discarded_results += 1

            # Record outcome for learning
            self.estimator.record_outcome(proposal_id, False)

            # Log discard
            self.merkle_chain.add_event(
                "speculative_result_discarded",
                {
                    "result_id": result.result_id,
                    "proposal_id": proposal_id,
                    "reason": reason,
                },
            )

            return True

    def _evict_oldest(self) -> None:
        """Evict oldest cached result to make room."""
        if not self._cache:
            return

        # Find oldest non-validated result
        oldest_key = None
        oldest_time = None

        for key, result in self._cache.items():
            if result.status in (SpeculativeStatus.VALIDATED, SpeculativeStatus.RUNNING):
                continue

            result_time = datetime.fromisoformat(result.timestamp)
            if oldest_time is None or result_time < oldest_time:
                oldest_key = key
                oldest_time = result_time

        if oldest_key:
            del self._cache[oldest_key]

    def get_executor_stats(self) -> dict[str, Any]:
        """Get executor statistics."""
        hit_rate = (
            self._used_results / self._total_speculative
            if self._total_speculative > 0
            else 0
        )

        status_counts: dict[str, int] = {}
        for result in self._cache.values():
            status_counts[result.status.value] = status_counts.get(result.status.value, 0) + 1

        return {
            "executor_id": self.executor_id,
            "likelihood_threshold": self.likelihood_threshold,
            "max_speculative": self.max_speculative,
            "cache_size": len(self._cache),
            "total_speculative": self._total_speculative,
            "used_results": self._used_results,
            "discarded_results": self._discarded_results,
            "hit_rate": hit_rate,
            "total_time_saved_ms": self._total_time_saved_ms,
            "status_counts": status_counts,
        }
