"""
QRATUM Vertical Modules

Domain-specific AI engines running on the QRATUM platform.
All vertical modules extend VerticalModuleBase and implement
domain-specific tasks with full determinism and auditability.
"""

from .base import VerticalModuleBase
from .juris import JurisModule
from .vitra import VitraModule
from .ecora import EcoraModule
from .capra import CapraModule
from .sentra import SentraModule
from .neura import NeuraModule
from .fluxa import FluxaModule
from .chrona import ChronaModule
from .geona import GeonaModule
from .fusia import FusiaModule
from .strata import StrataModule
from .vexor import VexorModule
from .cohora import CohoraModule
from .orbia import OrbiaModule

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
