"""

    ██████╗ ██████╗  █████╗ ████████╗██╗   ██╗███╗   ███╗
   ██╔═══██╗██╔══██╗██╔══██╗╚══██╔══╝██║   ██║████╗ ████║
   ██║   ██║██████╔╝███████║   ██║   ██║   ██║██╔████╔██║
   ██║▄▄ ██║██╔══██╗██╔══██║   ██║   ██║   ██║██║╚██╔╝██║
   ╚██████╔╝██║  ██║██║  ██║   ██║   ╚██████╔╝██║ ╚═╝ ██║
    ╚══▀▀═╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝

QRATUM: Quantum Resource Allocation, Tensor Analysis, and Unified Modeling

High-performance quantum simulation for modern GPU clusters.
Formerly known as QuASIM.

Classification: UNCLASSIFIED // CUI
Website: https://qratum.io
GitHub: https://github.com/robertringler/QRATUM
Docs: https://qratum.io/docs
"""

from qratum.config import QRATUMConfig, get_config, reset_config, set_config
from qratum.core import gates
from qratum.core.circuit import Circuit
from qratum.core.densitymatrix import DensityMatrix
from qratum.core.measurement import Measurement, Result

# Core imports
from qratum.core.simulator import Simulator
from qratum.core.statevector import StateVector
from qratum.version import (
    __author__,
    __classification__,
    __docs__,
    __github__,
    __legacy_name__,
    __license__,
    __rebrand_date__,
    __url__,
    __version__,
)

# Quantum integration
try:
    from qratum.quantum.integration import QuantumModuleAdapter

    QUANTUM_INTEGRATION_AVAILABLE = True
except ImportError:
    QUANTUM_INTEGRATION_AVAILABLE = False
    QuantumModuleAdapter = None
# Platform integration layer (Task 1)
from qratum.core.platform import QRATUMPlatform, create_platform
from qratum.core.platform_config import PlatformConfig

__all__ = [
    # Version and metadata
    "__version__",
    "__author__",
    "__legacy_name__",
    "__rebrand_date__",
    "__license__",
    "__url__",
    "__github__",
    "__docs__",
    "__classification__",
    # Configuration
    "QRATUMConfig",
    "get_config",
    "set_config",
    "reset_config",
    # Core classes
    "Simulator",
    "Circuit",
    "StateVector",
    "Measurement",
    "Result",
    "DensityMatrix",
    "gates",
    # Platform integration
    # Platform integration (Task 1)
    "PlatformConfig",
    "QRATUMPlatform",
    "create_platform",
]


def create_platform(quantum_backend: str = "simulator", seed: int = 42, **kwargs):
    """Create QRATUM platform instance with quantum capabilities.

    Args:
        quantum_backend: Backend type ('simulator', 'ibmq')
        seed: Random seed for reproducibility
        **kwargs: Additional configuration (shots, ibmq_token, device_name)

    Returns:
        QuantumModuleAdapter instance for executing quantum algorithms

    Raises:
        ImportError: If quantum modules are not available

    Example:
        >>> platform = create_platform(quantum_backend="simulator", seed=42)
        >>> result = platform.execute_vqe("H2", 0.735)
    """
    if not QUANTUM_INTEGRATION_AVAILABLE:
        raise ImportError(
            "Quantum integration modules not available. "
            "Install with: pip install qiskit qiskit-aer qiskit-nature pyscf"
        )

    platform_config = {
        "quantum_backend": quantum_backend,
        "seed": seed,
        "shots": kwargs.get("shots", 1024),
        "ibmq_token": kwargs.get("ibmq_token"),
        "device_name": kwargs.get("device_name"),
    }

    return QuantumModuleAdapter(platform_config)


def print_banner():
    """Print QRATUM ASCII banner."""

    banner = f"""

    ██████╗ ██████╗  █████╗ ████████╗██╗   ██╗███╗   ███╗
   ██╔═══██╗██╔══██╗██╔══██╗╚══██╔══╝██║   ██║████╗ ████║
   ██║   ██║██████╔╝███████║   ██║   ██║   ██║██╔████╔██║
   ██║▄▄ ██║██╔══██╗██╔══██║   ██║   ██║   ██║██║╚██╔╝██║
   ╚██████╔╝██║  ██║██║  ██║   ██║   ╚██████╔╝██║ ╚═╝ ██║
    ╚══▀▀═╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝

    Quantum Resource Allocation, Tensor Analysis, and Unified Modeling
    Version {__version__} | Formerly QuASIM
    https://qratum.io
    """

    print(banner)


if __name__ == "__main__":
    print_banner()
