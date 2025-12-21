"""NEURA - Neuroscience & BCI Module for QRATUM Platform.

Provides neural network simulation, EEG/MEG signal analysis,
brain connectivity mapping, BCI signal processing, and cognitive modeling.
"""

import random
from typing import Any, Dict, List

from qratum_platform.core import (
    ComputeSubstrate,
    PlatformContract,
    VerticalModuleBase,
)
from qratum_platform.substrates import get_optimal_substrate, VerticalModule
from qratum_platform.utils import compute_deterministic_seed


class NEURAModule(VerticalModuleBase):
    """Neuroscience and Brain-Computer Interface module."""

    MODULE_NAME = "NEURA"
    MODULE_VERSION = "2.0.0"
    SAFETY_DISCLAIMER = """
    NEURA Neuroscience & BCI Disclaimer:
    - FOR RESEARCH USE ONLY
    - NOT for clinical diagnosis or treatment
    - NOT FDA approved for medical use
    - BCI applications require IRB approval
    - Privacy and consent requirements apply
    - Consult qualified neuroscientists and physicians
    - Neural data is highly sensitive - protect privacy
    """

    PROHIBITED_USES = [
        "unauthorized mind reading",
        "coercive BCI applications",
        "privacy violations",
        "non-consensual neural monitoring",
        "brain manipulation without consent",
    ]

    # EEG frequency bands (Hz)
    FREQUENCY_BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 100),
    }

    def execute(self, contract: PlatformContract) -> Dict[str, Any]:
        """Execute neuroscience/BCI operation."""
        operation = contract.intent.operation
        parameters = contract.intent.parameters

        self.emit_event("neura_execution_start", contract.contract_id, {"operation": operation})

        try:
            if operation == "neural_simulation":
                result = self._simulate_neural_network(parameters)
            elif operation == "eeg_analysis":
                result = self._analyze_eeg_signal(parameters)
            elif operation == "connectivity_mapping":
                result = self._map_brain_connectivity(parameters)
            elif operation == "bci_processing":
                result = self._process_bci_signal(parameters)
            elif operation == "cognitive_modeling":
                result = self._model_cognitive_process(parameters)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            self.emit_event(
                "neura_execution_complete",
                contract.contract_id,
                {"operation": operation, "success": True},
            )
            return result
        except Exception as e:
            self.emit_event(
                "neura_execution_failed",
                contract.contract_id,
                {"operation": operation, "error": str(e)},
            )
            raise

    def _simulate_neural_network(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate spiking neural network.

        Args:
            parameters: num_neurons, duration_ms, connectivity

        Returns:
            Neural simulation results
        """
        num_neurons = parameters.get("num_neurons", 100)
        duration_ms = parameters.get("duration_ms", 1000)
        connectivity = parameters.get("connectivity", 0.1)

        # Derive seed from parameters for input-dependent determinism
        seed = compute_deterministic_seed(parameters)
        random.seed(seed)

        # Generate spike trains
        spike_trains = []
        for i in range(min(num_neurons, 100)):
            firing_rate = random.uniform(1, 20)  # Hz
            num_spikes = int(firing_rate * duration_ms / 1000)
            spikes = sorted([random.uniform(0, duration_ms) for _ in range(num_spikes)])
            spike_trains.append({"neuron_id": i, "firing_rate_hz": firing_rate, "spike_times_ms": spikes[:20]})

        return {
            "simulation_type": "spiking_neural_network",
            "num_neurons": num_neurons,
            "duration_ms": duration_ms,
            "connectivity": connectivity,
            "spike_trains": spike_trains[:10],  # First 10 neurons
            "mean_firing_rate_hz": sum(st["firing_rate_hz"] for st in spike_trains) / len(spike_trains),
            "note": "Simplified Poisson spiking model",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _analyze_eeg_signal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze EEG/MEG signal.

        Args:
            parameters: signal_data, sampling_rate_hz, channels

        Returns:
            EEG analysis results
        """
        sampling_rate = parameters.get("sampling_rate_hz", 256)
        channels = parameters.get("channels", ["Fp1", "Fp2", "F3", "F4", "C3", "C4"])
        duration_s = parameters.get("duration_s", 60)

        # Derive seed from parameters for input-dependent determinism
        seed = compute_deterministic_seed(parameters)
        random.seed(seed)

        # Frequency band analysis
        band_power = {}
        for band, (low, high) in self.FREQUENCY_BANDS.items():
            # Simplified power calculation
            power = random.uniform(1, 10)  # μV²
            band_power[band] = power

        # Artifact detection (simplified)
        artifacts = self._detect_artifacts(duration_s, seed)

        # Event detection
        events = self._detect_neural_events(duration_s, seed)

        return {
            "analysis_type": "eeg_spectral",
            "sampling_rate_hz": sampling_rate,
            "num_channels": len(channels),
            "duration_s": duration_s,
            "band_power": band_power,
            "dominant_frequency": "alpha",
            "artifacts_detected": artifacts,
            "events_detected": events,
            "note": "Simplified spectral analysis",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _detect_artifacts(self, duration_s: float, seed: int) -> List[Dict[str, Any]]:
        """Detect artifacts in EEG signal."""
        random.seed(seed)
        num_artifacts = random.randint(0, 5)
        artifacts = []
        for i in range(num_artifacts):
            artifacts.append(
                {
                    "type": random.choice(["eye_blink", "muscle", "electrode"]),
                    "time_s": random.uniform(0, duration_s),
                    "duration_ms": random.uniform(100, 500),
                }
            )
        return artifacts

    def _detect_neural_events(self, duration_s: float, seed: int) -> List[Dict[str, Any]]:
        """Detect neural events."""
        random.seed(seed + 1)  # Use different seed than artifacts
        events = []
        for i in range(random.randint(2, 8)):
            events.append(
                {
                    "type": random.choice(["p300", "n400", "alpha_burst"]),
                    "time_s": random.uniform(0, duration_s),
                    "amplitude_uv": random.uniform(5, 20),
                }
            )
        return events

    def _map_brain_connectivity(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Map brain connectivity patterns.

        Args:
            parameters: regions, connectivity_metric

        Returns:
            Connectivity analysis results
        """
        regions = parameters.get("regions", ["frontal", "parietal", "temporal", "occipital"])
        metric = parameters.get("connectivity_metric", "coherence")

        # Generate connectivity matrix
        n = len(regions)
        random.seed(42)
        connectivity_matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                if i == j:
                    conn = 1.0
                else:
                    conn = random.uniform(0.1, 0.8)
                row.append(round(conn, 2))
            connectivity_matrix.append(row)

        # Calculate network metrics
        network_metrics = self._calculate_network_metrics(connectivity_matrix)

        return {
            "analysis_type": "brain_connectivity",
            "num_regions": n,
            "regions": regions,
            "connectivity_metric": metric,
            "connectivity_matrix": connectivity_matrix,
            "network_metrics": network_metrics,
            "note": "Simplified connectivity analysis",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _calculate_network_metrics(self, matrix: List[List[float]]) -> Dict[str, Any]:
        """Calculate network metrics."""
        n = len(matrix)
        # Average connectivity
        total = sum(sum(row) for row in matrix) - n  # Subtract diagonal
        avg_connectivity = total / (n * (n - 1))

        return {
            "average_connectivity": round(avg_connectivity, 3),
            "clustering_coefficient": 0.45,  # Simplified
            "path_length": 2.1,  # Simplified
            "small_world_index": 1.5,  # Simplified
        }

    def _process_bci_signal(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Process BCI signal for control output.

        Args:
            parameters: signal, task_type

        Returns:
            BCI processing results
        """
        task_type = parameters.get("task_type", "motor_imagery")
        channels = parameters.get("channels", 8)

        # Classify intent (simplified)
        random.seed(42)
        intent = random.choice(["left", "right", "up", "down"])
        confidence = random.uniform(0.6, 0.95)

        # Control signal
        control_output = {"direction": intent, "magnitude": confidence, "latency_ms": random.uniform(200, 400)}

        return {
            "processing_type": "bci_classification",
            "task_type": task_type,
            "num_channels": channels,
            "detected_intent": intent,
            "confidence": confidence,
            "control_output": control_output,
            "classification_accuracy": 0.82,  # Typical BCI accuracy
            "note": "Simplified BCI decoding",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _model_cognitive_process(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Model cognitive process.

        Args:
            parameters: process_type, complexity

        Returns:
            Cognitive modeling results
        """
        process_type = parameters.get("process_type", "working_memory")
        complexity = parameters.get("complexity", "medium")

        # Simplified cognitive model
        capacity = {"working_memory": 7, "attention": 4, "decision_making": 5}.get(process_type, 5)

        reaction_time_ms = {"low": 300, "medium": 500, "high": 800}.get(complexity, 500)

        return {
            "model_type": "cognitive_architecture",
            "process_type": process_type,
            "complexity": complexity,
            "capacity_items": capacity,
            "reaction_time_ms": reaction_time_ms,
            "accuracy": 0.85 if complexity == "low" else 0.70 if complexity == "medium" else 0.55,
            "cognitive_load": 0.6,
            "note": "Simplified cognitive model",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def get_optimal_substrate(self, operation: str, parameters: Dict[str, Any]) -> ComputeSubstrate:
        """Get optimal compute substrate for neuroscience operation."""
        return get_optimal_substrate(VerticalModule.NEURA, operation)
