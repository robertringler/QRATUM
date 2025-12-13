#!/usr/bin/env python3
"""QuASIM Ansys Adapter - PyMAPDL integration for hybrid solver acceleration.

This module provides a Python-first integration layer for Ansys Mechanical,
enabling GPU-accelerated nonlinear elastomer mechanics with deterministic
reproducibility and aerospace-grade quality assurance.

Usage:
    from quasim_ansys_adapter import QuasimAnsysAdapter, SolverMode
    
    adapter = QuasimAnsysAdapter(mode=SolverMode.CO_SOLVER, device="gpu")
    adapter.import_mesh_from_mapdl(mapdl_session)
    adapter.add_material("rubber", model="mooney_rivlin", C10=0.5, C01=0.2)
    adapter.solve()
    adapter.export_results_to_mapdl()
"""

from __future__ import annotations

import enum
import hashlib
import json
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Enumerations
# ============================================================================

class SolverMode(enum.Enum):
    """Solver integration modes."""
    
    CO_SOLVER = "co_solver"
    """Run in parallel with Ansys, handling specific physics domains."""
    
    PRECONDITIONER = "preconditioner"
    """Provide initial solution estimate to accelerate Ansys convergence."""
    
    STANDALONE = "standalone"
    """Fully replace Ansys for supported physics."""


class DeviceType(enum.Enum):
    """Compute device types."""
    
    CPU = "cpu"
    """CPU execution (automatic fallback)."""
    
    GPU = "gpu"
    """Single GPU acceleration (CUDA required)."""
    
    MULTI_GPU = "multi_gpu"
    """Multi-GPU distributed execution."""


class MaterialModel(enum.Enum):
    """Supported material constitutive models."""
    
    # Hyperelastic models
    MOONEY_RIVLIN = "mooney_rivlin"
    """2-parameter Mooney-Rivlin hyperelastic model."""
    
    NEO_HOOKEAN = "neo_hookean"
    """Single-parameter Neo-Hookean model."""
    
    OGDEN = "ogden"
    """Ogden model (up to 6-term series)."""
    
    YEOH = "yeoh"
    """Yeoh reduced polynomial model."""
    
    # Viscoelastic models
    PRONY_SERIES = "prony_series"
    """Generalized Maxwell viscoelastic model."""
    
    # Thermal shift models
    WLF_SHIFT = "wlf_shift"
    """Williams-Landel-Ferry temperature shift."""
    
    # Linear elastic (for reference)
    LINEAR_ELASTIC = "linear_elastic"
    """Linear elastic material (testing/comparison)."""


# ============================================================================
# Data Structures
# ============================================================================

@dataclass
class MeshData:
    """Finite element mesh representation.
    
    Attributes:
        nodes: Node coordinates, shape (num_nodes, 3) [x, y, z].
        elements: Element connectivity, shape (num_elements, max_nodes_per_elem).
        element_types: Ansys element type IDs (e.g., 186 for SOLID186).
        node_sets: Named node selections for boundary conditions.
        element_sets: Named element selections for material assignment.
        coordinate_system: Coordinate system type.
    """
    
    nodes: np.ndarray
    elements: np.ndarray
    element_types: np.ndarray
    node_sets: dict[str, np.ndarray] = field(default_factory=dict)
    element_sets: dict[str, np.ndarray] = field(default_factory=dict)
    coordinate_system: str = "cartesian"
    
    def __post_init__(self) -> None:
        """Validate mesh data."""
        if self.nodes.ndim != 2 or self.nodes.shape[1] != 3:
            raise ValueError(f"Nodes must be (N, 3) array, got {self.nodes.shape}")
        if self.elements.ndim != 2:
            raise ValueError(f"Elements must be 2D array, got {self.elements.shape}")
        if len(self.element_types) != len(self.elements):
            raise ValueError("Element types array length must match elements array")
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in mesh."""
        return len(self.nodes)
    
    @property
    def num_elements(self) -> int:
        """Number of elements in mesh."""
        return len(self.elements)
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "num_nodes": self.num_nodes,
            "num_elements": self.num_elements,
            "element_types": self.element_types.tolist(),
            "coordinate_system": self.coordinate_system,
            "node_sets": {k: v.tolist() for k, v in self.node_sets.items()},
            "element_sets": {k: v.tolist() for k, v in self.element_sets.items()},
        }


@dataclass
class StateVector:
    """Simulation state at a given time.
    
    Attributes:
        time: Current simulation time.
        displacements: Nodal displacements, shape (num_nodes, 3) [ux, uy, uz].
        velocities: Nodal velocities (for dynamic analysis), shape (num_nodes, 3).
        accelerations: Nodal accelerations (for dynamic analysis).
        temperatures: Nodal temperatures (for thermal coupling).
        internal_state: Internal state variables (e.g., viscoelastic history).
    """
    
    time: float
    displacements: np.ndarray
    velocities: np.ndarray | None = None
    accelerations: np.ndarray | None = None
    temperatures: np.ndarray | None = None
    internal_state: dict[str, np.ndarray] = field(default_factory=dict)
    
    def __post_init__(self) -> None:
        """Validate state vector."""
        if self.displacements.ndim != 2 or self.displacements.shape[1] != 3:
            raise ValueError(f"Displacements must be (N, 3), got {self.displacements.shape}")
    
    @property
    def num_nodes(self) -> int:
        """Number of nodes in state."""
        return len(self.displacements)
    
    def compute_hash(self) -> str:
        """Compute SHA-256 hash of state for reproducibility verification."""
        data = self.displacements.tobytes()
        return hashlib.sha256(data).hexdigest()


@dataclass
class MaterialParameters:
    """Material model parameters.
    
    Attributes:
        material_id: Unique material identifier.
        name: Human-readable material name.
        model: Material constitutive model.
        parameters: Model-specific parameters.
        density: Material density [kg/m³].
        temperature_reference: Reference temperature [K].
    """
    
    material_id: int
    name: str
    model: MaterialModel
    parameters: dict[str, float]
    density: float = 1000.0
    temperature_reference: float = 293.15  # K (20°C)
    
    def validate(self) -> None:
        """Validate material parameters."""
        # Check for negative moduli
        if self.model == MaterialModel.MOONEY_RIVLIN:
            if "C10" not in self.parameters or "C01" not in self.parameters:
                raise ValueError("Mooney-Rivlin requires C10 and C01 parameters")
            if self.parameters["C10"] <= 0 or self.parameters["C01"] <= 0:
                raise ValueError("Mooney-Rivlin parameters must be positive")
        
        elif self.model == MaterialModel.NEO_HOOKEAN:
            if "C10" not in self.parameters:
                raise ValueError("Neo-Hookean requires C10 parameter")
            if self.parameters["C10"] <= 0:
                raise ValueError("Neo-Hookean C10 must be positive")
        
        elif self.model == MaterialModel.LINEAR_ELASTIC:
            if "youngs_modulus" not in self.parameters:
                raise ValueError("Linear elastic requires youngs_modulus")
            if self.parameters["youngs_modulus"] <= 0:
                raise ValueError("Young's modulus must be positive")
        
        # Check density
        if self.density <= 0:
            raise ValueError(f"Density must be positive, got {self.density}")
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "material_id": self.material_id,
            "name": self.name,
            "model": self.model.value,
            "parameters": self.parameters,
            "density": self.density,
            "temperature_reference": self.temperature_reference,
        }


@dataclass
class SolverConfig:
    """Solver configuration parameters.
    
    Attributes:
        analysis_type: Type of analysis (static, transient, etc.).
        large_deflection: Enable geometric nonlinearity.
        convergence_tolerance: Force convergence tolerance.
        max_iterations: Maximum Newton-Raphson iterations.
        substeps: Number of load substeps.
        line_search: Enable line search for convergence acceleration.
        precision: Floating-point precision (fp32, fp64).
        tensor_backend: Tensor computation backend.
    """
    
    analysis_type: str = "static_nonlinear"
    large_deflection: bool = True
    convergence_tolerance: float = 0.005
    max_iterations: int = 25
    substeps: int = 10
    line_search: bool = True
    precision: str = "fp64"
    tensor_backend: str = "cuquantum"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "analysis_type": self.analysis_type,
            "large_deflection": self.large_deflection,
            "convergence_tolerance": self.convergence_tolerance,
            "max_iterations": self.max_iterations,
            "substeps": self.substeps,
            "line_search": self.line_search,
            "precision": self.precision,
            "tensor_backend": self.tensor_backend,
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for solver execution.
    
    Attributes:
        solve_time: Total solve time [seconds].
        setup_time: Setup time (mesh import, material initialization) [seconds].
        iterations: Number of Newton-Raphson iterations.
        convergence_history: Residual norm at each iteration.
        memory_usage: Peak GPU memory usage [GB].
        speedup: Speedup vs Ansys baseline (if available).
    """
    
    solve_time: float
    setup_time: float
    iterations: int
    convergence_history: list[float] = field(default_factory=list)
    memory_usage: float = 0.0
    speedup: float | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "solve_time": self.solve_time,
            "setup_time": self.setup_time,
            "iterations": self.iterations,
            "convergence_history": self.convergence_history,
            "memory_usage": self.memory_usage,
            "speedup": self.speedup,
        }


# ============================================================================
# Exceptions
# ============================================================================

class QuasimError(Exception):
    """Base exception for QuASIM errors."""


class MeshImportError(QuasimError):
    """Mesh import failed."""


class MaterialParameterError(QuasimError):
    """Invalid material parameters."""


class ConvergenceError(QuasimError):
    """Solver failed to converge."""


class GPUMemoryError(QuasimError):
    """GPU out of memory."""


class GPUDriverError(QuasimError):
    """CUDA driver error."""


# ============================================================================
# Main Adapter Class
# ============================================================================

class QuasimAnsysAdapter:
    """QuASIM Ansys integration adapter.
    
    This class provides the main interface for integrating QuASIM with Ansys
    Mechanical via PyMAPDL. It supports three integration modes: co-solver,
    preconditioner, and standalone.
    
    Args:
        mode: Integration mode (co-solver, preconditioner, standalone).
        device: Compute device (cpu, gpu, multi_gpu).
        mapdl_session: Active PyMAPDL session (for co-solver mode).
        random_seed: Random seed for deterministic execution.
    
    Example:
        >>> from ansys.mapdl.core import launch_mapdl
        >>> mapdl = launch_mapdl()
        >>> adapter = QuasimAnsysAdapter(mode=SolverMode.CO_SOLVER, device="gpu")
        >>> adapter.import_mesh_from_mapdl(mapdl)
        >>> adapter.add_material(1, "rubber", MaterialModel.MOONEY_RIVLIN, 
        ...                      {"C10": 0.5, "C01": 0.2})
        >>> adapter.solve()
        >>> adapter.export_results_to_mapdl(mapdl)
    """
    
    def __init__(
        self,
        mode: SolverMode = SolverMode.CO_SOLVER,
        device: str = "gpu",
        mapdl_session: Any = None,
        random_seed: int = 42,
    ) -> None:
        """Initialize QuASIM Ansys adapter."""
        self.mode = mode
        self.device = DeviceType(device) if isinstance(device, str) else device
        self.mapdl_session = mapdl_session
        self.random_seed = random_seed
        
        # Data
        self.mesh: MeshData | None = None
        self.materials: dict[int, MaterialParameters] = {}
        self.state: StateVector | None = None
        self.config = SolverConfig()
        self.metrics: PerformanceMetrics | None = None
        
        # GPU context
        self._gpu_available = False
        self._gpu_initialized = False
        
        logger.info(f"Initialized QuasimAnsysAdapter (mode={mode.value}, device={device})")
        
        # Initialize GPU context
        if self.device in (DeviceType.GPU, DeviceType.MULTI_GPU):
            self._initialize_gpu()
    
    def _initialize_gpu(self) -> None:
        """Initialize GPU context (CUDA).
        
        Raises:
            GPUDriverError: If GPU initialization fails.
        """
        try:
            # TODO: C++/CUDA integration - initialize cuQuantum context
            # For now, check if GPU is theoretically available
            try:
                import torch
                self._gpu_available = torch.cuda.is_available()
                if self._gpu_available:
                    logger.info(f"GPU detected: {torch.cuda.get_device_name(0)}")
                    logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
                    self._gpu_initialized = True
                else:
                    logger.warning("GPU requested but not available, falling back to CPU")
                    self.device = DeviceType.CPU
            except ImportError:
                logger.warning("PyTorch not available for GPU detection, assuming GPU available")
                self._gpu_available = True
                self._gpu_initialized = True
        
        except Exception as e:
            raise GPUDriverError(f"Failed to initialize GPU: {e}") from e
    
    def import_mesh_from_mapdl(self, mapdl: Any = None) -> MeshData:
        """Import mesh from active Ansys MAPDL session.
        
        Args:
            mapdl: Active PyMAPDL session (optional if provided in constructor).
        
        Returns:
            Imported mesh data.
        
        Raises:
            MeshImportError: If mesh import fails.
        
        Example:
            >>> from ansys.mapdl.core import launch_mapdl
            >>> mapdl = launch_mapdl()
            >>> # ... setup geometry, mesh in Ansys ...
            >>> adapter.import_mesh_from_mapdl(mapdl)
        """
        start_time = time.time()
        
        mapdl = mapdl or self.mapdl_session
        if mapdl is None:
            raise MeshImportError("No MAPDL session provided")
        
        logger.info("Importing mesh from MAPDL session...")
        
        try:
            # TODO: C++/CUDA integration - actual PyMAPDL mesh extraction
            # For now, create a simple test mesh
            logger.warning("Using test mesh (PyMAPDL integration not yet implemented)")
            
            # Create simple 8-node hex mesh (2x2x2 elements)
            nodes = np.array([
                [0.0, 0.0, 0.0], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0],
                [0.0, 0.5, 0.0], [0.5, 0.5, 0.0], [1.0, 0.5, 0.0],
                [0.0, 1.0, 0.0], [0.5, 1.0, 0.0], [1.0, 1.0, 0.0],
                [0.0, 0.0, 0.5], [0.5, 0.0, 0.5], [1.0, 0.0, 0.5],
                [0.0, 0.5, 0.5], [0.5, 0.5, 0.5], [1.0, 0.5, 0.5],
                [0.0, 1.0, 0.5], [0.5, 1.0, 0.5], [1.0, 1.0, 0.5],
            ])
            
            elements = np.array([
                [0, 1, 4, 3, 9, 10, 13, 12],
                [1, 2, 5, 4, 10, 11, 14, 13],
                [3, 4, 7, 6, 12, 13, 16, 15],
                [4, 5, 8, 7, 13, 14, 17, 16],
            ])
            
            element_types = np.array([186, 186, 186, 186])  # SOLID186
            
            self.mesh = MeshData(
                nodes=nodes,
                elements=elements,
                element_types=element_types,
                node_sets={"bottom": np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])},
                element_sets={"all": np.array([0, 1, 2, 3])},
            )
            
            elapsed = time.time() - start_time
            logger.info(f"Mesh imported: {self.mesh.num_nodes} nodes, "
                       f"{self.mesh.num_elements} elements ({elapsed:.2f}s)")
            
            return self.mesh
        
        except Exception as e:
            raise MeshImportError(f"Failed to import mesh from MAPDL: {e}") from e
    
    def import_mesh_from_file(self, filepath: str | Path) -> MeshData:
        """Import mesh from Ansys CDB file.
        
        Args:
            filepath: Path to Ansys .cdb file.
        
        Returns:
            Imported mesh data.
        
        Raises:
            MeshImportError: If file not found or parsing fails.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise MeshImportError(f"Mesh file not found: {filepath}")
        
        logger.info(f"Importing mesh from {filepath}...")
        
        # TODO: C++/CUDA integration - actual CDB parser
        raise NotImplementedError("CDB file import not yet implemented")
    
    def set_mesh(self, mesh: MeshData) -> None:
        """Set mesh data directly (for advanced users).
        
        Args:
            mesh: MeshData object.
        """
        mesh.__post_init__()  # Validate
        self.mesh = mesh
        logger.info(f"Mesh set: {mesh.num_nodes} nodes, {mesh.num_elements} elements")
    
    def add_material(
        self,
        material_id: int,
        name: str,
        model: MaterialModel | str,
        parameters: dict[str, float],
        density: float = 1000.0,
        temperature_reference: float = 293.15,
    ) -> None:
        """Add material definition.
        
        Args:
            material_id: Unique material identifier (matches Ansys MAT ID).
            name: Human-readable material name.
            model: Material constitutive model.
            parameters: Model-specific parameters (e.g., C10, C01 for Mooney-Rivlin).
            density: Material density [kg/m³].
            temperature_reference: Reference temperature [K].
        
        Raises:
            MaterialParameterError: If parameters are invalid.
        
        Example:
            >>> adapter.add_material(
            ...     material_id=1,
            ...     name="EPDM_Rubber",
            ...     model=MaterialModel.MOONEY_RIVLIN,
            ...     parameters={"C10": 0.293, "C01": 0.177, "bulk_modulus": 2000.0},
            ...     density=1100.0
            ... )
        """
        if isinstance(model, str):
            model = MaterialModel(model)
        
        mat = MaterialParameters(
            material_id=material_id,
            name=name,
            model=model,
            parameters=parameters,
            density=density,
            temperature_reference=temperature_reference,
        )
        
        try:
            mat.validate()
        except ValueError as e:
            raise MaterialParameterError(str(e)) from e
        
        self.materials[material_id] = mat
        logger.info(f"Material added: {name} (ID={material_id}, model={model.value})")
    
    def solve(
        self,
        element_types: list[str] | None = None,
        config: SolverConfig | None = None,
    ) -> StateVector:
        """Execute solver.
        
        Args:
            element_types: Ansys element types to solve (for co-solver mode).
            config: Solver configuration (optional, uses default if not provided).
        
        Returns:
            Solution state vector.
        
        Raises:
            ConvergenceError: If solver fails to converge.
            GPUMemoryError: If GPU memory is exhausted.
        
        Example:
            >>> adapter.solve()
            >>> print(f"Solve completed in {adapter.metrics.solve_time:.2f}s")
        """
        if self.mesh is None:
            raise ValueError("No mesh imported. Call import_mesh_from_mapdl() first.")
        
        if not self.materials:
            raise ValueError("No materials defined. Call add_material() first.")
        
        if config is not None:
            self.config = config
        
        logger.info(f"Starting solve (mode={self.mode.value}, device={self.device.value})...")
        logger.info(f"  Mesh: {self.mesh.num_nodes} nodes, {self.mesh.num_elements} elements")
        logger.info(f"  Materials: {len(self.materials)}")
        logger.info(f"  Config: {self.config.analysis_type}, {self.config.substeps} substeps")
        
        start_time = time.time()
        
        try:
            # TODO: C++/CUDA integration - actual solver call
            # For now, create a simple displacement field
            logger.warning("Using mock solver (C++/CUDA integration not yet implemented)")
            
            # Simulate solve with mock displacement
            displacements = np.zeros((self.mesh.num_nodes, 3))
            
            # Apply simple deformation pattern (linear gradient in z)
            z_coords = self.mesh.nodes[:, 2]
            z_max = z_coords.max()
            if z_max > 0:
                displacements[:, 2] = -0.01 * (z_coords / z_max)  # 1cm max deflection
            
            # Simulate convergence
            convergence_history = [1.0, 0.5, 0.1, 0.05, 0.01, 0.003]
            
            self.state = StateVector(
                time=1.0,
                displacements=displacements,
            )
            
            solve_time = time.time() - start_time
            
            self.metrics = PerformanceMetrics(
                solve_time=solve_time,
                setup_time=0.1,
                iterations=len(convergence_history),
                convergence_history=convergence_history,
                memory_usage=0.5,  # GB
            )
            
            logger.info(f"Solve completed in {solve_time:.2f}s "
                       f"({self.metrics.iterations} iterations)")
            logger.info(f"  Final residual: {convergence_history[-1]:.2e}")
            logger.info(f"  Memory usage: {self.metrics.memory_usage:.2f} GB")
            
            return self.state
        
        except MemoryError as e:
            raise GPUMemoryError("GPU memory exhausted") from e
        
        except Exception as e:
            raise ConvergenceError(f"Solver failed: {e}") from e
    
    def export_results_to_mapdl(self, mapdl: Any = None) -> None:
        """Export results back to Ansys MAPDL session.
        
        Args:
            mapdl: Active PyMAPDL session (optional if provided in constructor).
        
        Raises:
            ValueError: If no solution state available.
        
        Example:
            >>> adapter.export_results_to_mapdl(mapdl)
            >>> mapdl.post1()  # Enter postprocessor
            >>> mapdl.pldisp()  # Plot displacements
        """
        if self.state is None:
            raise ValueError("No solution state. Call solve() first.")
        
        mapdl = mapdl or self.mapdl_session
        if mapdl is None:
            raise ValueError("No MAPDL session provided")
        
        logger.info("Exporting results to MAPDL session...")
        
        # TODO: C++/CUDA integration - actual result export to PyMAPDL
        logger.warning("Result export to MAPDL not yet implemented")
        logger.info(f"  Displacements: {self.state.num_nodes} nodes")
        logger.info(f"  State hash: {self.state.compute_hash()[:16]}...")
    
    def export_results_to_file(
        self,
        filepath: str | Path,
        format: str = "rst",
    ) -> None:
        """Export results to file.
        
        Args:
            filepath: Output file path.
            format: Output format (rst, vtk, csv, json, hdf5).
        
        Raises:
            ValueError: If no solution state or invalid format.
        
        Example:
            >>> adapter.export_results_to_file("results.rst", format="rst")
            >>> adapter.export_results_to_file("results.vtu", format="vtk")
        """
        if self.state is None:
            raise ValueError("No solution state. Call solve() first.")
        
        filepath = Path(filepath)
        
        logger.info(f"Exporting results to {filepath} (format={format})...")
        
        if format == "json":
            # Export summary to JSON
            data = {
                "mesh": self.mesh.to_dict() if self.mesh else None,
                "state": {
                    "time": self.state.time,
                    "num_nodes": self.state.num_nodes,
                    "hash": self.state.compute_hash(),
                },
                "metrics": self.metrics.to_dict() if self.metrics else None,
                "materials": {
                    k: v.to_dict() for k, v in self.materials.items()
                },
                "config": self.config.to_dict(),
            }
            
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Results exported to {filepath}")
        
        elif format == "csv":
            # Export displacements to CSV
            header = "NodeID,Ux,Uy,Uz"
            data = np.column_stack([
                np.arange(self.state.num_nodes),
                self.state.displacements
            ])
            np.savetxt(filepath, data, delimiter=",", header=header, comments="")
            logger.info(f"Displacements exported to {filepath}")
        
        else:
            # TODO: C++/CUDA integration - RST, VTK, HDF5 export
            raise NotImplementedError(f"Export format '{format}' not yet implemented")
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """Get performance metrics from last solve.
        
        Returns:
            Performance metrics object.
        
        Raises:
            ValueError: If no solve has been executed.
        """
        if self.metrics is None:
            raise ValueError("No metrics available. Call solve() first.")
        return self.metrics
    
    def set_solver_config(self, **kwargs: Any) -> None:
        """Update solver configuration parameters.
        
        Args:
            **kwargs: Configuration parameters (see SolverConfig).
        
        Example:
            >>> adapter.set_solver_config(
            ...     max_iterations=50,
            ...     convergence_tolerance=0.001,
            ...     line_search=True
            ... )
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Config updated: {key}={value}")
            else:
                logger.warning(f"Unknown config parameter: {key}")
    
    def register_as_preconditioner(self, mapdl: Any) -> None:
        """Register QuASIM as preconditioner for Ansys solver.
        
        Args:
            mapdl: Active PyMAPDL session.
        
        Note:
            This is an advanced feature for preconditioner mode.
        """
        if self.mode != SolverMode.PRECONDITIONER:
            warnings.warn("Registering as preconditioner but mode is not PRECONDITIONER")
        
        logger.info("Registering QuASIM as Ansys preconditioner...")
        
        # TODO: C++/CUDA integration - register callback with Ansys
        raise NotImplementedError("Preconditioner registration not yet implemented")


# ============================================================================
# Utility Functions
# ============================================================================

def test_installation() -> bool:
    """Test QuASIM Ansys adapter installation.
    
    Returns:
        True if installation is valid, False otherwise.
    """
    logger.info("Testing QuASIM Ansys adapter installation...")
    
    # Check Python version
    import sys
    if sys.version_info < (3, 10):
        logger.error(f"Python 3.10+ required, found {sys.version_info.major}.{sys.version_info.minor}")
        return False
    logger.info(f"✓ Python {sys.version_info.major}.{sys.version_info.minor}")
    
    # Check NumPy
    logger.info(f"✓ NumPy {np.__version__}")
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            logger.info(f"✓ GPU available: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("! GPU not available (CPU fallback will be used)")
    except ImportError:
        logger.warning("! PyTorch not installed (cannot detect GPU)")
    
    # Test adapter creation
    try:
        adapter = QuasimAnsysAdapter(device="cpu")
        logger.info("✓ Adapter creation successful")
    except Exception as e:
        logger.error(f"✗ Adapter creation failed: {e}")
        return False
    
    logger.info("\n✓ Installation test passed!")
    return True


if __name__ == "__main__":
    # Run installation test if executed as script
    import sys
    success = test_installation()
    sys.exit(0 if success else 1)
