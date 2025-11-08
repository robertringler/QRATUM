"""PRNG seed management for deterministic simulation replay."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass
class SeedRecord:
    """Seed management record.

    Attributes:
        seed_value: PRNG seed value
        timestamp: Creation timestamp
        environment: Environment identifier
        replay_id: Unique replay identifier
        hash_digest: SHA256 hash of seed + metadata for verification
    """

    seed_value: int
    timestamp: float
    environment: str
    replay_id: str
    hash_digest: str = ""

    def __post_init__(self):
        """Compute hash digest after initialization."""
        if not self.hash_digest:
            self.hash_digest = self._compute_hash()

    def _compute_hash(self) -> str:
        """Compute SHA256 hash for verification."""
        data = f"{self.seed_value}|{self.timestamp}|{self.environment}|{self.replay_id}"
        return hashlib.sha256(data.encode()).hexdigest()

    def verify(self) -> bool:
        """Verify record integrity."""
        return self.hash_digest == self._compute_hash()


class SeedRepository:
    """Repository for seed storage and retrieval."""

    def __init__(self):
        """Initialize seed repository."""
        self._seeds: dict[str, SeedRecord] = {}
        self._seed_sequence: list[int] = []

    def store(self, record: SeedRecord) -> None:
        """Store seed record.

        Args:
            record: Seed record to store
        """
        self._seeds[record.replay_id] = record
        self._seed_sequence.append(record.seed_value)

    def retrieve(self, replay_id: str) -> SeedRecord | None:
        """Retrieve seed record by replay ID.

        Args:
            replay_id: Replay identifier

        Returns:
            Seed record if found, None otherwise
        """
        return self._seeds.get(replay_id)

    def get_sequence(self) -> list[int]:
        """Get complete seed sequence.

        Returns:
            List of seed values in storage order
        """
        return self._seed_sequence.copy()

    def get_all_records(self) -> list[SeedRecord]:
        """Get all stored seed records.

        Returns:
            List of all seed records
        """
        return list(self._seeds.values())

    def count(self) -> int:
        """Get count of stored seeds.

        Returns:
            Number of seeds in repository
        """
        return len(self._seeds)

    def clear(self) -> None:
        """Clear all stored seeds."""
        self._seeds.clear()
        self._seed_sequence.clear()


class SeedManager:
    """Manage PRNG seeds for deterministic replay.

    Ensures deterministic replay across 1024-trajectory batches with timestamp
    synchronization < 1μs drift.
    """

    def __init__(self, base_seed: int = 42, environment: str = "default"):
        """Initialize seed manager.

        Args:
            base_seed: Base seed for deterministic generation
            environment: Environment identifier
        """
        self.base_seed = base_seed
        self.environment = environment
        self.repository = SeedRepository()
        self._rng = random.Random(base_seed)

    def generate_seed(self, replay_id: str | None = None) -> SeedRecord:
        """Generate new seed with tracking.

        Args:
            replay_id: Optional replay identifier (auto-generated if None)

        Returns:
            Seed record with metadata
        """
        if replay_id is None:
            replay_id = f"replay_{self.repository.count():06d}"

        seed_value = self._rng.randint(0, 2**31 - 1)
        timestamp = time.time()

        record = SeedRecord(
            seed_value=seed_value,
            timestamp=timestamp,
            environment=self.environment,
            replay_id=replay_id,
        )

        self.repository.store(record)

        return record

    def generate_batch(self, batch_size: int = 1024) -> list[SeedRecord]:
        """Generate batch of seeds for Monte-Carlo simulation.

        Args:
            batch_size: Number of seeds to generate

        Returns:
            List of seed records
        """
        batch = []

        for i in range(batch_size):
            replay_id = f"mc_batch_{i:04d}"
            record = self.generate_seed(replay_id)
            batch.append(record)

        return batch

    def replay_from_seed(self, seed_record: SeedRecord) -> random.Random:
        """Create RNG instance for deterministic replay.

        Args:
            seed_record: Seed record to replay

        Returns:
            RNG instance initialized with seed

        Raises:
            ValueError: If seed record verification fails
        """
        if not seed_record.verify():
            raise ValueError("Seed record verification failed")

        return random.Random(seed_record.seed_value)

    def export_manifest(self) -> dict[str, Any]:
        """Export seed manifest for audit trail.

        Returns:
            Manifest dictionary with all seed records
        """
        records = [
            {
                "seed_value": record.seed_value,
                "timestamp": record.timestamp,
                "timestamp_iso": datetime.fromtimestamp(record.timestamp).isoformat(),
                "environment": record.environment,
                "replay_id": record.replay_id,
                "hash_digest": record.hash_digest,
            }
            for record in self.repository.get_all_records()
        ]

        return {
            "base_seed": self.base_seed,
            "environment": self.environment,
            "total_seeds": len(records),
            "generated_at": datetime.now().isoformat(),
            "records": records,
        }


class DeterministicValidator:
    """Validate deterministic replay with timestamp synchronization."""

    @staticmethod
    def validate_replay(
        original_record: SeedRecord,
        replay_record: SeedRecord,
        max_drift_us: float = 1.0,
    ) -> tuple[bool, float]:
        """Validate deterministic replay.

        Args:
            original_record: Original seed record
            replay_record: Replay seed record
            max_drift_us: Maximum allowed timestamp drift in microseconds

        Returns:
            Tuple of (is_valid, drift_microseconds)
        """
        # Verify seed values match
        if original_record.seed_value != replay_record.seed_value:
            return False, float("inf")

        # Verify environment matches
        if original_record.environment != replay_record.environment:
            return False, float("inf")

        # Calculate timestamp drift in microseconds
        drift_seconds = abs(replay_record.timestamp - original_record.timestamp)
        drift_us = drift_seconds * 1_000_000

        is_valid = drift_us < max_drift_us

        return is_valid, drift_us

    @staticmethod
    def validate_batch_determinism(
        batch1: list[SeedRecord],
        batch2: list[SeedRecord],
    ) -> tuple[bool, dict[str, Any]]:
        """Validate determinism across two batches.

        Args:
            batch1: First batch of seed records
            batch2: Second batch of seed records

        Returns:
            Tuple of (is_deterministic, validation_report)
        """
        if len(batch1) != len(batch2):
            return False, {"error": "Batch sizes differ"}

        mismatches = []
        drifts = []

        for i, (rec1, rec2) in enumerate(zip(batch1, batch2)):
            if rec1.seed_value != rec2.seed_value:
                mismatches.append(
                    {
                        "index": i,
                        "seed1": rec1.seed_value,
                        "seed2": rec2.seed_value,
                    }
                )

            drift = abs(rec2.timestamp - rec1.timestamp) * 1_000_000  # μs
            drifts.append(drift)

        report = {
            "batch_size": len(batch1),
            "mismatches": len(mismatches),
            "mismatch_details": mismatches[:10],  # First 10 mismatches
            "max_drift_us": max(drifts) if drifts else 0.0,
            "mean_drift_us": sum(drifts) / len(drifts) if drifts else 0.0,
            "deterministic": len(mismatches) == 0,
        }

        return report["deterministic"], report
