"""XENON simulation adapter for visualization pipeline integration."""

from __future__ import annotations

from typing import Any

from qubic.visualization.adapters.base import SimulationAdapter
from qubic.visualization.core.data_model import VisualizationData
from xenon.adapters.bio_mechanism_adapter import BioMechanismAdapter
from xenon.adapters.molecular_state_adapter import MolecularStateAdapter
from xenon.adapters.transition_adapter import TransitionAdapter
from xenon.core.mechanism import BioMechanism, MolecularState, Transition


class XenonSimulationAdapter(SimulationAdapter):
    """Adapter for XENON bio-mechanism simulations.

    Integrates XENON simulation outputs into the Qubic visualization pipeline,
    supporting BioMechanism, MolecularState, and Transition data types.
    """

    def __init__(self, layout: str = "spring", scale: float = 10.0):
        """Initialize XENON adapter.

        Args:
            layout: Network layout algorithm ('spring', 'circular', 'hierarchical')
            scale: Spatial scale factor for visualization
        """
        self.layout = layout
        self.scale = scale

    def load_data(self, source: Any) -> VisualizationData:
        """Load XENON simulation data and convert to visualization format.

        Args:
            source: XENON simulation object (BioMechanism, MolecularState, or Transition)

        Returns:
            VisualizationData object ready for rendering

        Raises:
            ValueError: If source type is not supported
        """
        if isinstance(source, BioMechanism):
            adapter = BioMechanismAdapter(source)
            return adapter.to_3d_network(layout=self.layout, scale=self.scale)
        elif isinstance(source, MolecularState):
            adapter = MolecularStateAdapter(source)
            return adapter.to_energy_surface()
        elif isinstance(source, Transition):
            # For standalone transition, create arrow between origin and destination
            # (requires position information, using default positions here)
            import numpy as np

            source_pos = np.array([0.0, 0.0, 0.0])
            target_pos = np.array([self.scale, 0.0, 0.0])
            adapter = TransitionAdapter(source)
            return adapter.to_arrow(source_pos, target_pos)
        else:
            raise ValueError(
                f"Unsupported XENON source type: {type(source)}. "
                "Expected BioMechanism, MolecularState, or Transition."
            )

    def validate_source(self, source: Any) -> bool:
        """Validate that the source can be loaded by this adapter.

        Args:
            source: Data source to validate

        Returns:
            True if source is valid, False otherwise
        """
        return isinstance(source, (BioMechanism, MolecularState, Transition))

    def load_mechanism_timeseries(self, mechanisms: list[BioMechanism]) -> list[VisualizationData]:
        """Load a time series of bio-mechanism snapshots.

        Useful for streaming or time-series visualization of mechanism evolution.

        Args:
            mechanisms: List of BioMechanism snapshots over time

        Returns:
            List of VisualizationData objects, one per timestep
        """
        viz_data_list = []
        for mechanism in mechanisms:
            adapter = BioMechanismAdapter(mechanism)
            viz_data = adapter.to_3d_network(layout=self.layout, scale=self.scale)
            viz_data_list.append(viz_data)
        return viz_data_list
