"""
Physical and Computational Constants for QRATUM Temporal Computing

This module defines fundamental constants used throughout the temporal computing
framework, including physical constants from the universe and computational
constants specific to QRATUM's exascale architecture.
"""

# Physical Constants
SPEED_OF_LIGHT = 299_792_458  # meters per second (exact definition)
PLANCK_TIME = 5.391247e-44  # seconds (smallest meaningful time interval in physics)
PLANCK_LENGTH = 1.616255e-35  # meters

# Time Constants
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
SECONDS_PER_YEAR = 31_557_600  # Julian year
SECONDS_PER_COSMIC_YEAR = 7.1e15  # Galactic year (~225 million years)

# QRATUM Computational Constants
QRATUM_PEAK_FLOPS = 2.5e18  # 2.5 ExaFLOPS target performance
QRATUM_SUSTAINED_FLOPS = 2.0e18  # Sustained performance
OPERATIONS_PER_LIGHT_METER = QRATUM_PEAK_FLOPS / SPEED_OF_LIGHT  # ~8.3 billion ops/meter

# Temporal Compression Ratios
TARGET_COMPRESSION_RATIO = 1e9  # 1 billion simulated seconds per wall second
MAX_COMPRESSION_RATIO = 1e12  # Maximum theoretical compression

# State Management
MERKLE_HASH_SIZE = 32  # bytes (SHA-256)
DEFAULT_STATE_CHUNK_SIZE = 4096  # bytes
MAX_TIMELINE_BRANCHES = 10000  # Maximum parallel branches

# Verification Thresholds
STATE_VERIFICATION_THROUGHPUT = 1_000_000  # states per second
PARADOX_DETECTION_LATENCY = 100e-6  # 100 microseconds
TIMELINE_BRANCH_LATENCY = 1e-3  # 1 millisecond

# Memory Efficiency
TEMPORAL_STATE_OVERHEAD = 1024  # bytes per state (target: <1KB)

# Effective Velocity Multipliers
MIN_FTL_MULTIPLIER = 1.0  # Minimum speed-of-light multiple to be considered FTL
INFINITE_VELOCITY = float('inf')  # For instantaneous lookups

# Paradox Resolution
class ParadoxStrategy:
    """Enumeration of paradox resolution strategies"""
    NOVIKOV = "novikov"  # Self-consistency principle
    MANY_WORLDS = "many_worlds"  # Branch on paradox
    CHRONOLOGY_PROTECTION = "chronology_protection"  # Reject paradox

# Entropy Reversal Strategies
class EntropyStrategy:
    """Enumeration of entropy reversal strategies"""
    RECONSTRUCT = "reconstruct"  # Deterministic reconstruction
    PROBABILISTIC = "probabilistic"  # Probabilistic inference
    MERKLE_RECOVER = "merkle_recover"  # Recover from Merkle chain

# Temporal Resolution Levels
class TemporalResolution:
    """Pre-defined temporal resolution levels"""
    PLANCK = PLANCK_TIME
    NANOSECOND = 1e-9
    MICROSECOND = 1e-6
    MILLISECOND = 1e-3
    SECOND = 1.0
    MINUTE = SECONDS_PER_MINUTE
    HOUR = SECONDS_PER_HOUR
    DAY = SECONDS_PER_DAY
    YEAR = SECONDS_PER_YEAR
    COSMIC = SECONDS_PER_COSMIC_YEAR

# FTL Computation Mechanisms
class FTLMechanism:
    """Mechanisms for achieving effective FTL computation"""
    PRECOMPUTATION = "precomputation"  # Compute all possibilities in advance
    PREDICTIVE_MODELING = "predictive_modeling"  # Model futures faster than they unfold
    PARALLEL_BRANCHING = "parallel_branching"  # Explore all timelines simultaneously
    TEMPORAL_COMPRESSION = "temporal_compression"  # Compress simulated time vs wall time
    INVERSE_CAUSALITY = "inverse_causality"  # Compute backward from desired outcome
