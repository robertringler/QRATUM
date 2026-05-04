"""QuASIC Visualization Module.

A production-grade visualization platform featuring:
- Multi-GPU rendering cluster
- Multi-user real-time WebGPU dashboards
- VR/AR streaming adapters
- Historical replay and time-series analytics
- CAD export pipelines (OBJ, glTF, FBX, STEP)
- Cloud-native Docker/Kubernetes deployment scaffolding
"""

from .adapters import TireDataAdapter
from .dashboards import TimeSeriesAnalytics, create_dashboard_app
from .engines import ARAdapter, MultiGPURenderer, TireMeshGenerator
from .exporters import CADExporter

__version__ = "0.1.0"

__all__ = [
    "MultiGPURenderer",
    "ARAdapter",
    "TireMeshGenerator",
    "TireDataAdapter",
    "CADExporter",
    "TimeSeriesAnalytics",
    "create_dashboard_app",
]
