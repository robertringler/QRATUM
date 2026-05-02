"""Reality adapters bringing external data into Q-Stack deterministically."""

from qreal.base_adapter import AdapterOutput, BaseAdapter
from qreal.grid_adapter import GridAdapter
from qreal.market_adapter import MarketAdapter
from qreal.normalizers import NormalizationChain
from qreal.provenance import ProvenanceRecord, compute_provenance
from qreal.science_adapter import ScienceAdapter
from qreal.telemetry_adapter import TelemetryAdapter
from qreal.transport_adapter import TransportAdapter

__all__ = [
    "BaseAdapter",
    "AdapterOutput",
    "MarketAdapter",
    "TelemetryAdapter",
    "GridAdapter",
    "TransportAdapter",
    "ScienceAdapter",
    "ProvenanceRecord",
    "compute_provenance",
    "NormalizationChain",
]
