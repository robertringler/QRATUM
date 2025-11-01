"""Federated kernel intelligence system.

Implements anonymized telemetry aggregation and shared performance prediction
across deployments without sharing raw data.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class AnonymizedTelemetry:
    """Anonymized performance telemetry from a deployment."""
    
    deployment_id: str  # Hashed deployment identifier
    kernel_family: str  # Generic kernel type (e.g., "matmul", "conv2d")
    parameter_hash: str  # Hash of kernel parameters
    avg_latency_ms: float
    avg_throughput: float
    hardware_class: str  # Generic hardware category
    timestamp: float = field(default_factory=time.time)
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "deployment_id": self.deployment_id,
            "kernel_family": self.kernel_family,
            "parameter_hash": self.parameter_hash,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_throughput": self.avg_throughput,
            "hardware_class": self.hardware_class,
            "timestamp": self.timestamp
        }


@dataclass
class PerformancePredictor:
    """Global performance predictor trained on federated data."""
    
    predictor_id: str
    kernel_family: str
    model_weights: List[float] = field(default_factory=list)
    training_samples: int = 0
    last_updated: float = field(default_factory=time.time)
    
    def predict(self, parameter_hash: str, hardware_class: str) -> float:
        """Predict latency for given configuration.
        
        Args:
            parameter_hash: Hash of kernel parameters
            hardware_class: Hardware category
            
        Returns:
            Predicted latency in milliseconds
        """
        # Simple hash-based prediction for demonstration
        combined = f"{parameter_hash}_{hardware_class}"
        hash_val = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
        
        # Use model weights to modulate prediction
        base_latency = 5.0 + (hash_val % 1000) / 100.0
        if self.model_weights:
            weight_factor = sum(self.model_weights) / len(self.model_weights)
            base_latency *= (1.0 + weight_factor)
        
        return base_latency
    
    def update(self, telemetry: AnonymizedTelemetry, learning_rate: float = 0.01) -> None:
        """Update predictor with new telemetry (federated learning).
        
        Args:
            telemetry: New telemetry data
            learning_rate: Learning rate for updates
        """
        # Initialize weights if empty
        if not self.model_weights:
            self.model_weights = [random.uniform(-0.1, 0.1) for _ in range(4)]
        
        # Compute prediction error
        predicted = self.predict(telemetry.parameter_hash, telemetry.hardware_class)
        error = telemetry.avg_latency_ms - predicted
        
        # Simple gradient update
        for i in range(len(self.model_weights)):
            self.model_weights[i] += learning_rate * error * 0.1
        
        self.training_samples += 1
        self.last_updated = time.time()


class FederatedIntelligence:
    """Federated intelligence system for kernel optimization."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize federated intelligence system.
        
        Args:
            data_dir: Directory for telemetry and models
        """
        self.data_dir = data_dir or Path(__file__).parent
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.telemetry_buffer: List[AnonymizedTelemetry] = []
        self.predictors: Dict[str, PerformancePredictor] = {}
        
    def anonymize_deployment(self, deployment_name: str) -> str:
        """Create anonymized deployment ID.
        
        Args:
            deployment_name: Original deployment name
            
        Returns:
            Hashed deployment ID
        """
        return hashlib.sha256(deployment_name.encode()).hexdigest()[:16]
    
    def hash_parameters(self, params: Dict) -> str:
        """Hash kernel parameters for privacy.
        
        Args:
            params: Kernel parameters
            
        Returns:
            Parameter hash
        """
        param_str = json.dumps(params, sort_keys=True)
        return hashlib.md5(param_str.encode()).hexdigest()[:16]
    
    def submit_telemetry(self, deployment_name: str, kernel_family: str,
                        params: Dict, latency_ms: float, throughput: float,
                        hardware_class: str) -> AnonymizedTelemetry:
        """Submit anonymized telemetry.
        
        Args:
            deployment_name: Deployment identifier
            kernel_family: Kernel family/type
            params: Kernel parameters
            latency_ms: Measured latency
            throughput: Measured throughput
            hardware_class: Hardware category
            
        Returns:
            Anonymized telemetry record
        """
        telemetry = AnonymizedTelemetry(
            deployment_id=self.anonymize_deployment(deployment_name),
            kernel_family=kernel_family,
            parameter_hash=self.hash_parameters(params),
            avg_latency_ms=latency_ms,
            avg_throughput=throughput,
            hardware_class=hardware_class
        )
        
        self.telemetry_buffer.append(telemetry)
        return telemetry
    
    def aggregate_telemetry(self) -> Dict[str, List[AnonymizedTelemetry]]:
        """Aggregate telemetry by kernel family.
        
        Returns:
            Dictionary mapping kernel families to telemetry lists
        """
        aggregated: Dict[str, List[AnonymizedTelemetry]] = {}
        
        for telemetry in self.telemetry_buffer:
            family = telemetry.kernel_family
            if family not in aggregated:
                aggregated[family] = []
            aggregated[family].append(telemetry)
        
        return aggregated
    
    def train_predictor(self, kernel_family: str) -> PerformancePredictor:
        """Train or update performance predictor for a kernel family.
        
        Args:
            kernel_family: Kernel family to train predictor for
            
        Returns:
            Trained predictor
        """
        # Get or create predictor
        if kernel_family not in self.predictors:
            self.predictors[kernel_family] = PerformancePredictor(
                predictor_id=f"predictor_{kernel_family}",
                kernel_family=kernel_family
            )
        
        predictor = self.predictors[kernel_family]
        
        # Train on relevant telemetry
        for telemetry in self.telemetry_buffer:
            if telemetry.kernel_family == kernel_family:
                predictor.update(telemetry)
        
        return predictor
    
    def query_predictor(self, kernel_family: str, params: Dict, 
                       hardware_class: str) -> Optional[float]:
        """Query performance predictor.
        
        Args:
            kernel_family: Kernel family
            params: Kernel parameters
            hardware_class: Hardware category
            
        Returns:
            Predicted latency or None if no predictor available
        """
        if kernel_family not in self.predictors:
            return None
        
        parameter_hash = self.hash_parameters(params)
        predictor = self.predictors[kernel_family]
        return predictor.predict(parameter_hash, hardware_class)
    
    def save_telemetry(self) -> Path:
        """Save telemetry buffer to disk.
        
        Returns:
            Path to saved telemetry
        """
        telemetry_path = self.data_dir / "telemetry.json"
        
        telemetry_data = {
            "count": len(self.telemetry_buffer),
            "timestamp": time.time(),
            "telemetry": [t.to_dict() for t in self.telemetry_buffer]
        }
        
        with open(telemetry_path, 'w') as f:
            json.dump(telemetry_data, f, indent=2)
        
        return telemetry_path
    
    def save_predictors(self) -> Path:
        """Save all predictors to disk.
        
        Returns:
            Path to saved predictors
        """
        predictors_path = self.data_dir / "predictors.json"
        
        predictors_data = {
            predictor_id: {
                "kernel_family": p.kernel_family,
                "model_weights": p.model_weights,
                "training_samples": p.training_samples,
                "last_updated": p.last_updated
            }
            for predictor_id, p in self.predictors.items()
        }
        
        with open(predictors_path, 'w') as f:
            json.dump(predictors_data, f, indent=2)
        
        return predictors_path
    
    def get_statistics(self) -> Dict[str, any]:
        """Get system statistics.
        
        Returns:
            Dictionary of statistics
        """
        aggregated = self.aggregate_telemetry()
        
        return {
            "total_telemetry": len(self.telemetry_buffer),
            "kernel_families": len(aggregated),
            "trained_predictors": len(self.predictors),
            "families": {
                family: len(telemetry_list)
                for family, telemetry_list in aggregated.items()
            }
        }


if __name__ == "__main__":
    print("Federated Intelligence Demo")
    print("=" * 70)
    
    # Initialize system
    intel = FederatedIntelligence()
    
    # Simulate telemetry from multiple deployments
    print("\nSubmitting telemetry from deployments...")
    
    deployments = ["deployment_alpha", "deployment_beta", "deployment_gamma"]
    kernel_families = ["matmul", "conv2d", "attention"]
    
    for i in range(20):
        deployment = random.choice(deployments)
        family = random.choice(kernel_families)
        params = {
            "tile_size": random.choice([16, 32, 64]),
            "warp_count": random.choice([8, 16, 32])
        }
        latency = random.uniform(5.0, 15.0)
        throughput = random.uniform(100.0, 500.0)
        hardware = random.choice(["gpu_high", "gpu_medium", "cpu"])
        
        intel.submit_telemetry(deployment, family, params, latency, throughput, hardware)
    
    # Get statistics
    stats = intel.get_statistics()
    print(f"\nTelemetry collected: {stats['total_telemetry']} samples")
    print(f"Kernel families: {stats['kernel_families']}")
    
    # Train predictors
    print("\nTraining performance predictors...")
    for family in kernel_families:
        predictor = intel.train_predictor(family)
        print(f"  {family}: {predictor.training_samples} samples")
    
    # Test prediction
    print("\nTesting prediction...")
    test_params = {"tile_size": 32, "warp_count": 16}
    predicted = intel.query_predictor("matmul", test_params, "gpu_high")
    print(f"  Predicted latency for matmul: {predicted:.2f} ms")
    
    # Save data
    telemetry_path = intel.save_telemetry()
    predictors_path = intel.save_predictors()
    print(f"\nTelemetry saved to: {telemetry_path}")
    print(f"Predictors saved to: {predictors_path}")
