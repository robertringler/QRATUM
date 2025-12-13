"""QuASIM Ansys Integration SDK.

This package provides Python-first integration for Ansys Mechanical via PyMAPDL,
enabling GPU-accelerated nonlinear elastomer mechanics with deterministic
reproducibility and aerospace-grade quality assurance.
"""

from .quasim_ansys_adapter import (
    # Main adapter class
    QuasimAnsysAdapter,
    
    # Enumerations
    SolverMode,
    DeviceType,
    MaterialModel,
    
    # Data structures
    MeshData,
    StateVector,
    MaterialParameters,
    SolverConfig,
    PerformanceMetrics,
    
    # Exceptions
    QuasimError,
    MeshImportError,
    MaterialParameterError,
    ConvergenceError,
    GPUMemoryError,
    GPUDriverError,
    
    # Utilities
    test_installation,
)

__version__ = "1.0.0"
__all__ = [
    # Main class
    "QuasimAnsysAdapter",
    
    # Enumerations
    "SolverMode",
    "DeviceType",
    "MaterialModel",
    
    # Data structures
    "MeshData",
    "StateVector",
    "MaterialParameters",
    "SolverConfig",
    "PerformanceMetrics",
    
    # Exceptions
    "QuasimError",
    "MeshImportError",
    "MaterialParameterError",
    "ConvergenceError",
    "GPUMemoryError",
    "GPUDriverError",
    
    # Utilities
    "test_installation",
]
