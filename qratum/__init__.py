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

# Core imports
from qratum.core.simulator import Simulator
from qratum.core.circuit import Circuit
from qratum.core.statevector import StateVector
from qratum.core.measurement import Measurement, Result
from qratum.core.densitymatrix import DensityMatrix
from qratum.core import gates

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
]


def print_banner():
    """Print QRATUM ASCII banner."""
    banner = """
    ██████╗ ██████╗  █████╗ ████████╗██╗   ██╗███╗   ███╗
   ██╔═══██╗██╔══██╗██╔══██╗╚══██╔══╝██║   ██║████╗ ████║
   ██║   ██║██████╔╝███████║   ██║   ██║   ██║██╔████╔██║
   ██║▄▄ ██║██╔══██╗██╔══██║   ██║   ██║   ██║██║╚██╔╝██║
   ╚██████╔╝██║  ██║██║  ██║   ██║   ╚██████╔╝██║ ╚═╝ ██║
    ╚══▀▀═╝ ╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝     ╚═╝

    Quantum Resource Allocation, Tensor Analysis, and Unified Modeling
    Version {version} | Formerly QuASIM
    https://qratum.io
    """.format(
        version=__version__
    )
    print(banner)


if __name__ == "__main__":
    print_banner()
