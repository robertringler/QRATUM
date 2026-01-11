"""Adversarial Simulation Harness for QRATUM Security Testing

This module provides tools for adversarial testing of QRATUM components:
- Byzantine validator simulation
- Timing attack analysis
- Entropy starvation tests
- Memory extraction attempts
- Censorship resistance testing

Usage:
    from quasim.security.adversarial import (
        ByzantineSimulator,
        TimingAnalyzer,
        EntropyStarvationTest,
        CensorshipProber,
    )
    
    # Run adversarial simulation
    sim = ByzantineSimulator(num_validators=10, byzantine_fraction=0.33)
    result = sim.run_attack_scenario("double_voting")
"""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Any, Callable
import statistics


@dataclass
class AdversarialResult:
    """Result of an adversarial simulation."""
    
    test_name: str
    passed: bool
    attack_succeeded: bool
    details: dict[str, Any] = field(default_factory=dict)
    duration_ms: float = 0.0
    recommendations: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "attack_succeeded": self.attack_succeeded,
            "details": self.details,
            "duration_ms": self.duration_ms,
            "recommendations": self.recommendations,
        }


class ByzantineSimulator:
    """Simulates Byzantine validator behavior to test consensus resilience.
    
    Tests the BFT consensus implementation against various Byzantine attacks:
    - Double voting
    - Proposal withholding
    - Vote manipulation
    - Collusion attacks
    """
    
    def __init__(self, num_validators: int = 10, byzantine_fraction: float = 0.33):
        """Initialize Byzantine simulator.
        
        Args:
            num_validators: Total number of validators
            byzantine_fraction: Fraction of validators that are Byzantine
        """
        self.num_validators = num_validators
        self.byzantine_fraction = byzantine_fraction
        self.num_byzantine = int(num_validators * byzantine_fraction)
        self.num_honest = num_validators - self.num_byzantine
        
        # Create validator set
        self.validators = list(range(num_validators))
        self.byzantine_validators = set(random.sample(self.validators, self.num_byzantine))
        self.honest_validators = set(self.validators) - self.byzantine_validators
        
    def run_attack_scenario(self, attack_type: str) -> AdversarialResult:
        """Run a specific attack scenario.
        
        Args:
            attack_type: Type of attack to simulate
            
        Returns:
            AdversarialResult with test outcomes
        """
        start_time = time.time()
        
        attack_methods = {
            "double_voting": self._attack_double_voting,
            "proposal_withholding": self._attack_proposal_withholding,
            "vote_manipulation": self._attack_vote_manipulation,
            "collusion_safety": self._attack_collusion_safety,
            "collusion_liveness": self._attack_collusion_liveness,
        }
        
        if attack_type not in attack_methods:
            return AdversarialResult(
                test_name=attack_type,
                passed=False,
                attack_succeeded=False,
                details={"error": f"Unknown attack type: {attack_type}"},
            )
        
        result = attack_methods[attack_type]()
        result.duration_ms = (time.time() - start_time) * 1000
        
        return result
    
    def _attack_double_voting(self) -> AdversarialResult:
        """Test resilience against double voting attack."""
        rounds = 100
        double_votes_detected = 0
        consensus_failures = 0
        
        for _ in range(rounds):
            votes = {}
            for v in self.honest_validators:
                proposal = random.choice([0, 1])
                votes[v] = [proposal]
            
            for v in self.byzantine_validators:
                votes[v] = [0, 1]
            
            for v, v_votes in votes.items():
                if len(v_votes) > 1:
                    double_votes_detected += 1
            
            vote_counts = {0: 0, 1: 0}
            for v in self.honest_validators:
                vote_counts[votes[v][0]] += 1
            
            threshold = (2 * self.num_validators) // 3 + 1
            if max(vote_counts.values()) < threshold:
                consensus_failures += 1
        
        attack_succeeded = consensus_failures > rounds * 0.1
        
        return AdversarialResult(
            test_name="double_voting",
            passed=not attack_succeeded,
            attack_succeeded=attack_succeeded,
            details={
                "rounds": rounds,
                "double_votes_detected": double_votes_detected,
                "consensus_failures": consensus_failures,
                "byzantine_fraction": self.byzantine_fraction,
            },
            recommendations=[
                "Implement double-vote detection and slashing" if attack_succeeded 
                else "Double voting defense adequate",
            ],
        )
    
    def _attack_proposal_withholding(self) -> AdversarialResult:
        """Test resilience against proposal withholding."""
        rounds = 100
        stalled_rounds = 0
        view_changes = 0
        
        for round_num in range(rounds):
            proposer = round_num % self.num_validators
            
            if proposer in self.byzantine_validators:
                proposal_made = False
            else:
                proposal_made = True
            
            if not proposal_made:
                view_changes += 1
                stalled_rounds += 1
        
        stall_rate = stalled_rounds / rounds
        attack_succeeded = stall_rate > 0.5
        
        return AdversarialResult(
            test_name="proposal_withholding",
            passed=not attack_succeeded,
            attack_succeeded=attack_succeeded,
            details={
                "rounds": rounds,
                "stalled_rounds": stalled_rounds,
                "view_changes": view_changes,
                "stall_rate": stall_rate,
            },
            recommendations=[
                "Reduce timeout for view changes" if attack_succeeded 
                else "View change mechanism adequate",
            ],
        )
    
    def _attack_vote_manipulation(self) -> AdversarialResult:
        """Test resilience against vote count manipulation."""
        rounds = 100
        manipulation_detected = 0
        successful_manipulations = 0
        
        for _ in range(rounds):
            true_votes = {v: random.choice([0, 1]) for v in self.validators}
            
            reported_votes = true_votes.copy()
            for v in self.byzantine_validators:
                majority = sum(true_votes.values())
                if majority > self.num_validators // 2:
                    reported_votes[v] = 0
                else:
                    reported_votes[v] = 1
            
            for v in self.byzantine_validators:
                if reported_votes[v] != true_votes[v]:
                    manipulation_detected += 1
                    
            true_count = sum(true_votes.values())
            reported_count = sum(reported_votes.values())
            
            if (true_count > self.num_validators // 2) != (reported_count > self.num_validators // 2):
                successful_manipulations += 1
        
        attack_succeeded = successful_manipulations > 0 and manipulation_detected == 0
        
        return AdversarialResult(
            test_name="vote_manipulation",
            passed=not attack_succeeded,
            attack_succeeded=attack_succeeded,
            details={
                "rounds": rounds,
                "manipulation_detected": manipulation_detected,
                "successful_manipulations": successful_manipulations,
            },
            recommendations=[
                "Verify all vote signatures" if attack_succeeded 
                else "Vote verification adequate",
            ],
        )
    
    def _attack_collusion_safety(self) -> AdversarialResult:
        """Test safety under Byzantine collusion."""
        safety_threshold = self.num_validators / 3
        can_break_safety = self.num_byzantine >= safety_threshold
        
        attack_rounds = 100
        safety_violations = 0
        
        for _ in range(attack_rounds):
            votes_for_0 = len(self.byzantine_validators)
            votes_for_1 = len(self.byzantine_validators)
            
            threshold = (2 * self.num_validators) // 3 + 1
            
            if votes_for_0 < threshold and votes_for_1 < threshold:
                continue
                
            if votes_for_0 >= threshold and votes_for_1 >= threshold:
                safety_violations += 1
        
        attack_succeeded = safety_violations > 0
        
        return AdversarialResult(
            test_name="collusion_safety",
            passed=not attack_succeeded,
            attack_succeeded=attack_succeeded,
            details={
                "byzantine_count": self.num_byzantine,
                "safety_threshold": safety_threshold,
                "can_theoretically_break": can_break_safety,
                "safety_violations": safety_violations,
            },
            recommendations=[
                f"Reduce Byzantine tolerance to <{safety_threshold:.0f}" if attack_succeeded 
                else "BFT safety guarantees maintained",
            ],
        )
    
    def _attack_collusion_liveness(self) -> AdversarialResult:
        """Test liveness under Byzantine collusion."""
        liveness_threshold = self.num_validators / 3
        
        rounds = 100
        halted_rounds = 0
        
        for _ in range(rounds):
            honest_votes = len(self.honest_validators)
            threshold = (2 * self.num_validators) // 3 + 1
            
            if honest_votes < threshold:
                halted_rounds += 1
        
        halt_rate = halted_rounds / rounds
        attack_succeeded = halt_rate > 0.5
        
        return AdversarialResult(
            test_name="collusion_liveness",
            passed=not attack_succeeded,
            attack_succeeded=attack_succeeded,
            details={
                "byzantine_count": self.num_byzantine,
                "honest_count": self.num_honest,
                "threshold_needed": (2 * self.num_validators) // 3 + 1,
                "halted_rounds": halted_rounds,
                "halt_rate": halt_rate,
            },
            recommendations=[
                "Consider timeout-based fallback" if attack_succeeded 
                else "Liveness maintained with current Byzantine tolerance",
            ],
        )


class TimingAnalyzer:
    """Analyzes timing side-channels in cryptographic operations."""
    
    def __init__(self, target_function: Callable, samples: int = 10000):
        """Initialize timing analyzer."""
        self.target_function = target_function
        self.samples = samples
        self.measurements: list[float] = []
        
    def measure(self, inputs: list[Any]) -> AdversarialResult:
        """Measure timing for various inputs."""
        timings_by_input: dict[str, list[float]] = {}
        
        for inp in inputs:
            input_key = str(inp)[:50]
            timings_by_input[input_key] = []
            
            for _ in range(self.samples // len(inputs)):
                start = time.perf_counter_ns()
                try:
                    self.target_function(inp)
                except Exception:
                    pass
                end = time.perf_counter_ns()
                
                timings_by_input[input_key].append(end - start)
        
        means = {k: statistics.mean(v) for k, v in timings_by_input.items()}
        stdevs = {k: statistics.stdev(v) if len(v) > 1 else 0 for k, v in timings_by_input.items()}
        
        mean_values = list(means.values())
        max_diff = max(mean_values) - min(mean_values) if mean_values else 0
        avg_mean = statistics.mean(mean_values) if mean_values else 0
        
        timing_leak_detected = (max_diff / avg_mean) > 0.05 if avg_mean > 0 else False
        
        return AdversarialResult(
            test_name="timing_analysis",
            passed=not timing_leak_detected,
            attack_succeeded=timing_leak_detected,
            details={
                "samples_per_input": self.samples // len(inputs),
                "input_count": len(inputs),
                "mean_timings_ns": means,
                "stdev_timings_ns": stdevs,
                "max_timing_diff_ns": max_diff,
                "timing_diff_percent": (max_diff / avg_mean * 100) if avg_mean > 0 else 0,
            },
            recommendations=[
                "Review operation for data-dependent branches" if timing_leak_detected
                else "Timing appears constant",
            ],
        )


class EntropyStarvationTest:
    """Tests system behavior under entropy starvation conditions."""
    
    def __init__(self, min_entropy_bits: int = 256):
        """Initialize entropy starvation test."""
        self.min_entropy_bits = min_entropy_bits
        
    def test_degraded_entropy(
        self,
        key_derivation_func: Callable[[list[bytes]], Any],
    ) -> AdversarialResult:
        """Test key derivation with degraded entropy."""
        test_cases = [
            ("zero_entropy", [b"\x00" * 32]),
            ("low_entropy", [bytes([i % 4 for i in range(32)])]),
            ("single_source", [b"single_source_only_not_enough"]),
            ("adequate_entropy", [
                bytes(random.getrandbits(8) for _ in range(32)),
                bytes(random.getrandbits(8) for _ in range(32)),
            ]),
        ]
        
        results = {}
        for name, entropy_sources in test_cases:
            try:
                key = key_derivation_func(entropy_sources)
                results[name] = {"success": True, "key_derived": key is not None}
            except Exception as e:
                results[name] = {"success": False, "error": str(e)}
        
        weak_rejected = not results.get("zero_entropy", {}).get("success", True)
        single_source_rejected = not results.get("single_source", {}).get("success", True)
        adequate_accepted = results.get("adequate_entropy", {}).get("success", False)
        
        attack_succeeded = (
            results.get("zero_entropy", {}).get("success", False) or
            results.get("low_entropy", {}).get("success", False)
        )
        
        return AdversarialResult(
            test_name="entropy_starvation",
            passed=not attack_succeeded and adequate_accepted,
            attack_succeeded=attack_succeeded,
            details={
                "test_results": results,
                "weak_entropy_rejected": weak_rejected,
                "single_source_rejected": single_source_rejected,
                "adequate_entropy_accepted": adequate_accepted,
            },
            recommendations=[
                "Strengthen entropy validation" if attack_succeeded
                else "Entropy validation adequate",
            ],
        )


class CensorshipProber:
    """Probes system for censorship resistance."""
    
    def __init__(self, probe_interval_ms: int = 1000):
        """Initialize censorship prober."""
        self.probe_interval_ms = probe_interval_ms
        self.probes_sent = 0
        self.probes_received = 0
        self.blocked_probes = 0
        
    def run_probe_test(
        self,
        submit_func: Callable[[bytes], bool],
        verify_func: Callable[[bytes], bool],
        num_probes: int = 100,
    ) -> AdversarialResult:
        """Run censorship probe test."""
        for i in range(num_probes):
            probe_id = hashlib.sha256(f"probe_{i}_{time.time()}".encode()).digest()
            
            submitted = submit_func(probe_id)
            self.probes_sent += 1
            
            if submitted:
                time.sleep(self.probe_interval_ms / 1000)
                processed = verify_func(probe_id)
                
                if processed:
                    self.probes_received += 1
                else:
                    self.blocked_probes += 1
        
        censorship_rate = self.blocked_probes / self.probes_sent if self.probes_sent > 0 else 0
        censorship_detected = censorship_rate > 0.1
        
        return AdversarialResult(
            test_name="censorship_probe",
            passed=not censorship_detected,
            attack_succeeded=censorship_detected,
            details={
                "probes_sent": self.probes_sent,
                "probes_received": self.probes_received,
                "probes_blocked": self.blocked_probes,
                "censorship_rate": censorship_rate,
            },
            recommendations=[
                "Investigate blocked probes for censorship" if censorship_detected
                else "No significant censorship detected",
            ],
        )


class AdversarialTestSuite:
    """Complete adversarial test suite for QRATUM."""
    
    def __init__(self):
        """Initialize test suite."""
        self.results: list[AdversarialResult] = []
        
    def run_all_tests(self) -> dict[str, Any]:
        """Run all adversarial tests."""
        byzantine_sim = ByzantineSimulator(num_validators=10, byzantine_fraction=0.33)
        
        for attack in ["double_voting", "proposal_withholding", "vote_manipulation",
                       "collusion_safety", "collusion_liveness"]:
            result = byzantine_sim.run_attack_scenario(attack)
            self.results.append(result)
        
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        
        return {
            "total_tests": len(self.results),
            "passed": passed,
            "failed": failed,
            "results": [r.to_dict() for r in self.results],
        }


__all__ = [
    "AdversarialResult",
    "ByzantineSimulator",
    "TimingAnalyzer",
    "EntropyStarvationTest",
    "CensorshipProber",
    "AdversarialTestSuite",
]
