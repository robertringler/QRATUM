"""Q-Core Capability Resolution Engine.

This module implements capability resolution for intents,
mapping high-level capabilities to specific hardware resources.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qil.ast import Capability, Intent


@dataclass
class ResolvedCapability:
    """Represents a resolved capability.

    Attributes:
        name: Capability name
        cluster_type: Resolved cluster type
        resource_requirements: Required resources
        metadata: Additional metadata
    """

    name: str
    cluster_type: str
    resource_requirements: dict[str, Any]
    metadata: dict[str, Any]


class CapabilityResolutionError(Exception):
    """Exception raised for capability resolution failures (FATAL)."""

    pass


class CapabilityResolver:
    """Resolves capabilities to hardware bindings."""

    def __init__(self, capability_map: dict[str, list[str]] | None = None) -> None:
        """Initialize capability resolver.

        Args:
            capability_map: Optional mapping of capabilities to cluster types
        """
        self.capability_map = capability_map or self._default_capability_map()

    def _default_capability_map(self) -> dict[str, list[str]]:
        """Get default capability to cluster type mapping.

        Returns:
            Dictionary mapping capability names to cluster types
        """
        return {
            # AI/ML capabilities
            "llm_training": ["GB200", "MI300X", "GAUDI3"],
            "llm_inference": ["GB200", "MI300X", "GAUDI3", "CPU"],
            "neural_network": ["GB200", "MI300X", "IPU", "GAUDI3"],
            # Quantum capabilities
            "quantum_optimizer": ["QPU"],
            "quantum_simulation": ["QPU", "GB200", "MI300X"],
            "vqe": ["QPU"],
            "qaoa": ["QPU"],
            # Graph/combinatorial
            "graph_optimization": ["IPU", "GB200", "CPU"],
            "maxcut": ["QPU", "IPU", "GB200"],
            # Scientific computing
            "molecular_simulation": ["QPU", "GB200", "MI300X"],
            "chemistry": ["QPU", "GB200", "MI300X"],
            "physics_simulation": ["GB200", "MI300X", "CPU"],
            # Large-scale compute
            "wafer_scale": ["CEREBRAS"],
            "distributed_training": ["GB200", "MI300X", "GAUDI3"],
            # General purpose
            "general_compute": ["CPU", "GB200", "MI300X"],
        }

    def resolve_capabilities(self, intent: Intent) -> list[ResolvedCapability]:
        """Resolve capabilities for an intent.

        Args:
            intent: Intent with capabilities to resolve

        Returns:
            List of ResolvedCapability objects

        Raises:
            CapabilityResolutionError: If resolution fails
        """
        resolved: list[ResolvedCapability] = []

        # If no capabilities specified, use general_compute
        if not intent.capabilities:
            capabilities = [Capability(name="general_compute")]
        else:
            capabilities = intent.capabilities

        for capability in capabilities:
            # Get possible cluster types for this capability
            cluster_types = self.capability_map.get(capability.name, [])

            if not cluster_types:
                raise CapabilityResolutionError(
                    f"No cluster types available for capability '{capability.name}'"
                )

            # Filter by hardware constraints
            if intent.hardware:
                cluster_types = self._filter_by_hardware_spec(cluster_types, intent.hardware)

            if not cluster_types:
                raise CapabilityResolutionError(
                    f"No cluster types satisfy hardware constraints for '{capability.name}'"
                )

            # Select best cluster type (first in list for now)
            selected_cluster = cluster_types[0]

            # Build resource requirements
            requirements = self._compute_resource_requirements(
                capability.name, selected_cluster, intent
            )

            resolved.append(
                ResolvedCapability(
                    name=capability.name,
                    cluster_type=selected_cluster,
                    resource_requirements=requirements,
                    metadata={
                        "alternative_clusters": cluster_types[1:],
                        "intent_name": intent.name,
                    },
                )
            )

        return resolved

    def _filter_by_hardware_spec(self, cluster_types: list[str], hardware_spec: Any) -> list[str]:
        """Filter cluster types by hardware specification.

        Args:
            cluster_types: List of cluster types
            hardware_spec: Hardware specification from intent

        Returns:
            Filtered list of cluster types
        """
        filtered = cluster_types[:]

        # Apply ONLY clause
        if hardware_spec.only_clusters:
            filtered = [c for c in filtered if c in hardware_spec.only_clusters]

        # Apply NOT clause
        if hardware_spec.not_clusters:
            filtered = [c for c in filtered if c not in hardware_spec.not_clusters]

        return filtered

    def _compute_resource_requirements(
        self, capability_name: str, cluster_type: str, intent: Intent
    ) -> dict[str, Any]:
        """Compute resource requirements for a capability.

        Args:
            capability_name: Name of capability
            cluster_type: Selected cluster type
            intent: Original intent

        Returns:
            Dictionary of resource requirements
        """
        requirements: dict[str, Any] = {
            "capability": capability_name,
            "cluster_type": cluster_type,
        }

        # Extract constraint-based requirements
        for constraint in intent.constraints:
            if constraint.name.endswith("_VRAM"):
                requirements["vram_gb"] = constraint.value
            elif constraint.name.endswith("_MEMORY"):
                requirements["memory_gb"] = constraint.value
            elif constraint.name.endswith("_CORES"):
                requirements["cores"] = constraint.value
            elif constraint.name.endswith("_NODES"):
                requirements["nodes"] = constraint.value

        # Set defaults based on cluster type
        if cluster_type == "GB200":
            requirements.setdefault("vram_gb", 192)  # HBM3e per GPU
            requirements.setdefault("nodes", 1)
        elif cluster_type == "MI300X":
            requirements.setdefault("vram_gb", 192)  # HBM3 per GPU
            requirements.setdefault("nodes", 1)
        elif cluster_type == "QPU":
            requirements.setdefault("qubits", 100)
            requirements.setdefault("depth", 100)
        elif cluster_type == "CEREBRAS":
            requirements.setdefault("wafer_scale", True)
        elif cluster_type == "IPU":
            requirements.setdefault("ipus", 4)
        elif cluster_type == "GAUDI3":
            requirements.setdefault("accelerators", 8)
        elif cluster_type == "CPU":
            requirements.setdefault("cores", 64)

        return requirements

    def add_capability_mapping(self, capability: str, cluster_types: list[str]) -> None:
        """Add or update a capability mapping.

        Args:
            capability: Capability name
            cluster_types: List of supporting cluster types
        """
        self.capability_map[capability] = cluster_types

    def get_supported_clusters(self, capability: str) -> list[str]:
        """Get supported cluster types for a capability.

        Args:
            capability: Capability name

        Returns:
            List of supporting cluster types
        """
        return self.capability_map.get(capability, [])
