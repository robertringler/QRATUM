"""Autotuning module with Bayesian optimization for kernel configurations.

Implements energy-aware scheduling with power metrics from NVML/ROCm SMI.
"""
from __future__ import annotations

from .bayesian_tuner import BayesianTuner, TuningConfig
from .energy_monitor import EnergyMonitor, PowerMetrics

__all__ = [
    "BayesianTuner",
    "TuningConfig",
    "EnergyMonitor",
    "PowerMetrics",
]
