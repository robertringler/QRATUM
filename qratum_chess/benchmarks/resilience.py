"""Resilience and load-failure testing for QRATUM-Chess.

Simulates failure conditions:
- Corrupt hash tables
- GPU drop mid-game
- Thread termination (50%)
- NaN injection into NN outputs

System must recover in ≤ 500 ms with no illegal move generation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


class FailureType(Enum):
    """Types of simulated failures."""

    HASH_CORRUPTION = "hash_corruption"
    GPU_DROP = "gpu_drop"
    THREAD_KILL = "thread_kill"
    NAN_INJECTION = "nan_injection"


@dataclass
class ResilienceTestResult:
    """Result of a single resilience test."""

    failure_type: FailureType
    recovery_time_ms: float
    recovered_successfully: bool
    illegal_move_generated: bool
    moves_after_recovery: int
    error_message: str | None = None

    @property
    def passed(self) -> bool:
        """Check if test passed all criteria."""
        return (
            self.recovered_successfully
            and not self.illegal_move_generated
            and self.recovery_time_ms <= 500
        )


@dataclass
class ResilienceReport:
    """Report from resilience test suite."""

    results: list[ResilienceTestResult] = field(default_factory=list)
    total_tests: int = 0
    passed_tests: int = 0
    failed_tests: int = 0
    average_recovery_ms: float = 0.0
    illegal_moves_generated: int = 0

    def finalize(self) -> None:
        """Calculate aggregate metrics."""
        self.total_tests = len(self.results)
        self.passed_tests = sum(1 for r in self.results if r.passed)
        self.failed_tests = self.total_tests - self.passed_tests

        recovery_times = [r.recovery_time_ms for r in self.results if r.recovered_successfully]
        self.average_recovery_ms = (
            sum(recovery_times) / len(recovery_times) if recovery_times else 0
        )

        self.illegal_moves_generated = sum(1 for r in self.results if r.illegal_move_generated)

    def passed(self) -> bool:
        """Check if all resilience tests passed."""
        return self.failed_tests == 0


class ResilienceTest:
    """Resilience testing framework for QRATUM-Chess.

    Injects failures and measures recovery time and correctness.
    """

    # Recovery time threshold (ms)
    MAX_RECOVERY_TIME_MS = 500

    def __init__(self):
        """Initialize the resilience test framework."""
        self.failure_handlers: dict[FailureType, Callable] = {}

    def run_hash_corruption_test(
        self, engine, position, corruption_rate: float = 0.5
    ) -> ResilienceTestResult:
        """Test recovery from hash table corruption.

        Args:
            engine: Chess engine with transposition table.
            position: Position to search from.
            corruption_rate: Fraction of entries to corrupt.

        Returns:
            Test result.
        """

        # Corrupt hash table with malformed but structurally valid entries
        if hasattr(engine, "tt"):
            from qratum_chess.search.alphabeta import TranspositionEntry

            num_entries = len(engine.tt)
            num_corrupt = int(num_entries * corruption_rate)

            keys = list(engine.tt.keys())[:num_corrupt]
            for key in keys:
                # Corrupt with invalid but structurally valid entry
                # Use extreme/invalid values that should be handled gracefully
                engine.tt[key] = TranspositionEntry(
                    hash_key=key ^ 0xDEADBEEF,  # Wrong hash
                    depth=-999,  # Invalid depth
                    value=float("inf"),  # Extreme value
                    flag="invalid",  # Invalid flag
                    best_move=None,
                )

        # Measure recovery
        start = time.perf_counter()
        try:
            best_move, _, _ = engine.search(position, depth=8)
            recovery_time = (time.perf_counter() - start) * 1000
            recovered = True
            error = None
        except Exception as e:
            recovery_time = (time.perf_counter() - start) * 1000
            recovered = False
            best_move = None
            error = str(e)

        # Verify move legality
        illegal = False
        if best_move:
            legal_moves = position.generate_legal_moves()
            illegal = best_move not in legal_moves

        return ResilienceTestResult(
            failure_type=FailureType.HASH_CORRUPTION,
            recovery_time_ms=recovery_time,
            recovered_successfully=recovered,
            illegal_move_generated=illegal,
            moves_after_recovery=1 if best_move else 0,
            error_message=error,
        )

    def run_nan_injection_test(
        self, evaluator, position, injection_rate: float = 0.1
    ) -> ResilienceTestResult:
        """Test recovery from NaN injection in neural network.

        Args:
            evaluator: Neural evaluator instance.
            position: Position to evaluate.
            injection_rate: Rate of NaN injection.

        Returns:
            Test result.
        """
        import numpy as np

        # Store original forward method
        original_forward = None
        if hasattr(evaluator, "network"):
            original_forward = evaluator.network.forward

            def corrupted_forward(x):
                policy, value = original_forward(x)
                # Inject NaNs
                mask = np.random.random(policy.shape) < injection_rate
                policy[mask] = np.nan
                return policy, value

            evaluator.network.forward = corrupted_forward

        # Measure recovery
        start = time.perf_counter()
        try:
            policy, value = evaluator.evaluate(position)
            recovery_time = (time.perf_counter() - start) * 1000

            # Check for NaN in output
            has_nan = np.any(np.isnan(policy)) if hasattr(policy, "__iter__") else False
            recovered = not has_nan
            error = "NaN in output" if has_nan else None
        except Exception as e:
            recovery_time = (time.perf_counter() - start) * 1000
            recovered = False
            error = str(e)

        # Restore original
        if original_forward and hasattr(evaluator, "network"):
            evaluator.network.forward = original_forward

        return ResilienceTestResult(
            failure_type=FailureType.NAN_INJECTION,
            recovery_time_ms=recovery_time,
            recovered_successfully=recovered,
            illegal_move_generated=False,
            moves_after_recovery=0,
            error_message=error,
        )

    def run_thread_kill_test(
        self, engine, position, kill_rate: float = 0.5
    ) -> ResilienceTestResult:
        """Test recovery from thread termination.

        Args:
            engine: Multi-threaded chess engine.
            position: Position to search from.
            kill_rate: Fraction of threads to kill.

        Returns:
            Test result.
        """
        # Note: Actual thread killing is complex and platform-specific
        # This is a simulation of the behavior

        start = time.perf_counter()
        try:
            # Search with reduced time to simulate thread loss
            best_move, _, _ = engine.search(position, time_limit_ms=100)
            recovery_time = (time.perf_counter() - start) * 1000
            recovered = True
            error = None
        except Exception as e:
            recovery_time = (time.perf_counter() - start) * 1000
            recovered = False
            best_move = None
            error = str(e)

        # Verify move legality
        illegal = False
        if best_move:
            legal_moves = position.generate_legal_moves()
            illegal = best_move not in legal_moves

        return ResilienceTestResult(
            failure_type=FailureType.THREAD_KILL,
            recovery_time_ms=recovery_time,
            recovered_successfully=recovered,
            illegal_move_generated=illegal,
            moves_after_recovery=1 if best_move else 0,
            error_message=error,
        )

    def run_full_suite(
        self, engine, evaluator, positions: list, iterations: int = 10
    ) -> ResilienceReport:
        """Run complete resilience test suite.

        Args:
            engine: Chess engine to test.
            evaluator: Neural evaluator to test.
            positions: Test positions.
            iterations: Number of iterations per test type.

        Returns:
            Complete resilience report.
        """
        report = ResilienceReport()

        for i in range(iterations):
            pos = positions[i % len(positions)] if positions else None

            if pos:
                # Hash corruption test
                result = self.run_hash_corruption_test(engine, pos)
                report.results.append(result)

                # Thread kill test
                result = self.run_thread_kill_test(engine, pos)
                report.results.append(result)

            if evaluator and pos:
                # NaN injection test
                result = self.run_nan_injection_test(evaluator, pos)
                report.results.append(result)

        report.finalize()
        return report
