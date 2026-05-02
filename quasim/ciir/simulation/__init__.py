"""CIIR → QuASIM → QRATUM 3D Interactive Simulation.

This package provides a fully interactive simulation of the CIIR → QuASIM → QRATUM
cognitive pipeline with 3D visualization, real-time parameter control, tensor
exploration, performance dashboards, and export capabilities.

Modules
-------
engine : Core simulation engine orchestrating the full pipeline
ciir_module : CIIR state manifolds, observer perspectives, constraint visualization
quasim_module : QuASIM tensor runtime, gradient evolution, loss landscape visualization
qratum_module : QRATUM hardware abstraction, execution scheduling, performance metrics
renderer : 3D rendering with camera control, mesh generation, and animation
dashboard : Real-time metrics: loss convergence, constraint violations, gradient norms
interactive : Interactive controls for parameter tweaking and manifold navigation
export : Export utilities for screenshots, video, tensor logs, and CSV metrics
"""

__version__ = "0.1.0"

from quasim.ciir.simulation.engine import (CIIRSimulationEngine,
                                           SimulationConfig,
                                           run_and_capture_simulation)

__all__ = [
    "CIIRSimulationEngine",
    "SimulationConfig",
    "run_and_capture_simulation",
]
