"""Federated learning aggregator for cross-deployment performance data."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass
class AnonymizedTelemetry:
    """Anonymized telemetry record for federated learning."""

    deployment_hash: str  # Anonymized deployment ID
    kernel_config_hash: str  # Anonymized kernel configuration
    latency_ms: float
    energy_j: float
    throughput_gops: float
    timestamp: float

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "deployment_hash": self.deployment_hash,
            "kernel_config_hash": self.kernel_config_hash,
            "latency_ms": self.latency_ms,
            "energy_j": self.energy_j,
            "throughput_gops": self.throughput_gops,
            "timestamp": self.timestamp,
        }


@dataclass
class AggregatedPerformance:
    """Aggregated performance statistics."""

    kernel_config_hash: str
    sample_count: int
    avg_latency_ms: float
    avg_energy_j: float
    avg_throughput_gops: float
    std_latency_ms: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "kernel_config_hash": self.kernel_config_hash,
            "sample_count": self.sample_count,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_energy_j": self.avg_energy_j,
            "avg_throughput_gops": self.avg_throughput_gops,
            "std_latency_ms": self.std_latency_ms,
        }


class FederatedAggregator:
    """
    Aggregates anonymized performance data across deployments.
    Implements secure aggregation without sharing raw data.
    """

    def __init__(self, aggregation_dir: str = "federated/aggregated"):
        self.aggregation_dir = Path(aggregation_dir)
        self.aggregation_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry: List[AnonymizedTelemetry] = []
        self.aggregated: Dict[str, AggregatedPerformance] = {}

    def anonymize_deployment_id(self, deployment_id: str) -> str:
        """Anonymize deployment ID using hash."""
        return hashlib.sha256(deployment_id.encode()).hexdigest()[:16]

    def anonymize_kernel_config(self, config: Dict) -> str:
        """Anonymize kernel configuration using hash."""
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]

    def add_telemetry(
        self,
        deployment_id: str,
        kernel_config: Dict,
        latency_ms: float,
        energy_j: float,
        throughput_gops: float,
        timestamp: float,
    ) -> None:
        """Add anonymized telemetry record."""
        telemetry = AnonymizedTelemetry(
            deployment_hash=self.anonymize_deployment_id(deployment_id),
            kernel_config_hash=self.anonymize_kernel_config(kernel_config),
            latency_ms=latency_ms,
            energy_j=energy_j,
            throughput_gops=throughput_gops,
            timestamp=timestamp,
        )
        self.telemetry.append(telemetry)

    def aggregate_performance(self) -> Dict[str, AggregatedPerformance]:
        """
        Aggregate performance data by kernel configuration.
        Computes mean and standard deviation.
        """
        # Group by kernel config hash
        grouped: Dict[str, List[AnonymizedTelemetry]] = {}

        for record in self.telemetry:
            config_hash = record.kernel_config_hash
            if config_hash not in grouped:
                grouped[config_hash] = []
            grouped[config_hash].append(record)

        # Compute aggregated statistics
        self.aggregated = {}

        for config_hash, records in grouped.items():
            latencies = [r.latency_ms for r in records]
            energies = [r.energy_j for r in records]
            throughputs = [r.throughput_gops for r in records]

            avg_latency = sum(latencies) / len(latencies)
            avg_energy = sum(energies) / len(energies)
            avg_throughput = sum(throughputs) / len(throughputs)

            # Compute standard deviation for latency
            variance = sum((x - avg_latency) ** 2 for x in latencies) / len(latencies)
            std_latency = variance**0.5

            aggregated = AggregatedPerformance(
                kernel_config_hash=config_hash,
                sample_count=len(records),
                avg_latency_ms=avg_latency,
                avg_energy_j=avg_energy,
                avg_throughput_gops=avg_throughput,
                std_latency_ms=std_latency,
            )

            self.aggregated[config_hash] = aggregated

        return self.aggregated

    def predict_performance(self, kernel_config: Dict) -> AggregatedPerformance | None:
        """
        Predict performance for a kernel configuration using aggregated data.
        Returns None if no data available for this configuration.
        """
        config_hash = self.anonymize_kernel_config(kernel_config)
        return self.aggregated.get(config_hash)

    def save_aggregated_data(self) -> Path:
        """Save aggregated data to disk."""
        output_path = self.aggregation_dir / "aggregated_performance.json"

        data = {
            "aggregated": {k: v.to_dict() for k, v in self.aggregated.items()},
            "total_records": len(self.telemetry),
            "unique_configs": len(self.aggregated),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def export_telemetry_schema(self) -> Dict:
        """Export telemetry schema for documentation."""
        return {
            "schema_version": "1.0",
            "fields": {
                "deployment_hash": "string (16 chars) - anonymized deployment ID",
                "kernel_config_hash": "string (16 chars) - anonymized kernel config",
                "latency_ms": "float - execution latency in milliseconds",
                "energy_j": "float - energy consumption in joules",
                "throughput_gops": "float - throughput in GFLOP/s",
                "timestamp": "float - Unix timestamp",
            },
            "privacy": "All personally identifiable information is hashed using SHA-256",
        }
