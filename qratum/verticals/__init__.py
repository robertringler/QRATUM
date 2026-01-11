"""
QRATUM Vertical Modules

Domain-specific AI engines running on the QRATUM platform.
All vertical modules extend VerticalModuleBase and implement
domain-specific tasks with full determinism and auditability.
"""

from .base import VerticalModuleBase
from .capra import CapraModule
from .chrona import ChronaModule
from .cohora import CohoraModule
from .ecora import EcoraModule
from .fluxa import FluxaModule
from .fusia import FusiaModule
from .geona import GeonaModule
from .juris import JurisModule
from .neura import NeuraModule
from .orbia import OrbiaModule
from .sentra import SentraModule
from .strata import StrataModule
from .vexor import VexorModule
from .vitra import VitraModule

__all__ = [
    "VerticalModuleBase",
    "JurisModule",
    "VitraModule",
    "EcoraModule",
    "CapraModule",
    "SentraModule",
    "NeuraModule",
    "FluxaModule",
    "ChronaModule",
    "GeonaModule",
    "FusiaModule",
    "StrataModule",
    "VexorModule",
    "CohoraModule",
    "OrbiaModule",
]
