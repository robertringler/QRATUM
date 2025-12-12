"""Data adapters for external simulation results."""

from .quasim_adapter import QuASIMDataAdapter
from .tire_data_adapter import TireDataAdapter

__all__ = ["QuASIMDataAdapter", "TireDataAdapter"]
