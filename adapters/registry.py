"""Adapter Registry - Registry for Substrate Adapters.

This module provides a registry for cluster adapters,
allowing dynamic adapter lookup and registration.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

from adapters.base import BaseAdapter
from adapters.cerebras import CerebrasAdapter
from adapters.cpu import CPUAdapter
from adapters.gaudi import Gaudi3Adapter
from adapters.gb200 import GB200Adapter
from adapters.ipu import IPUAdapter
from adapters.mi300x import MI300XAdapter
from adapters.qpu import QPUAdapter


class AdapterRegistry:
    """Registry for substrate cluster adapters."""

    def __init__(self) -> None:
        """Initialize adapter registry with default adapters."""
        self._adapters: dict[str, BaseAdapter] = {}
        self._adapter_classes: dict[str, type[BaseAdapter]] = {}

        # Register default adapters
        self._register_defaults()

    def _register_defaults(self) -> None:
        """Register default cluster adapters."""
        self.register_adapter_class("CEREBRAS", CerebrasAdapter)
        self.register_adapter_class("GB200", GB200Adapter)
        self.register_adapter_class("MI300X", MI300XAdapter)
        self.register_adapter_class("QPU", QPUAdapter)
        self.register_adapter_class("IPU", IPUAdapter)
        self.register_adapter_class("GAUDI3", Gaudi3Adapter)
        self.register_adapter_class("CPU", CPUAdapter)

    def register_adapter_class(self, cluster_type: str, adapter_class: type[BaseAdapter]) -> None:
        """Register an adapter class for a cluster type.

        Args:
            cluster_type: Cluster type identifier
            adapter_class: Adapter class to register
        """
        self._adapter_classes[cluster_type] = adapter_class

    def register_adapter(self, cluster_type: str, adapter: BaseAdapter) -> None:
        """Register a pre-instantiated adapter.

        Args:
            cluster_type: Cluster type identifier
            adapter: Adapter instance to register
        """
        self._adapters[cluster_type] = adapter

    def get_adapter(self, cluster_type: str) -> BaseAdapter:
        """Get adapter for a cluster type.

        Args:
            cluster_type: Cluster type identifier

        Returns:
            Adapter instance

        Raises:
            ValueError: If no adapter registered for cluster type
        """
        # Return existing adapter if available
        if cluster_type in self._adapters:
            return self._adapters[cluster_type]

        # Create new adapter from class
        if cluster_type in self._adapter_classes:
            adapter_class = self._adapter_classes[cluster_type]
            adapter = adapter_class()
            self._adapters[cluster_type] = adapter
            return adapter

        raise ValueError(f"No adapter registered for cluster type: {cluster_type}")

    def has_adapter(self, cluster_type: str) -> bool:
        """Check if adapter is registered for cluster type.

        Args:
            cluster_type: Cluster type identifier

        Returns:
            True if adapter is registered, False otherwise
        """
        return cluster_type in self._adapters or cluster_type in self._adapter_classes

    def list_cluster_types(self) -> list[str]:
        """List all registered cluster types.

        Returns:
            List of cluster type identifiers
        """
        return sorted(set(self._adapters.keys()) | set(self._adapter_classes.keys()))

    def get_cluster_info(self) -> dict[str, dict]:
        """Get information about all registered clusters.

        Returns:
            Dictionary mapping cluster types to info
        """
        info = {}
        for cluster_type in self.list_cluster_types():
            if cluster_type in self._adapters:
                adapter = self._adapters[cluster_type]
                info[cluster_type] = adapter.get_cluster_info()
            else:
                info[cluster_type] = {
                    "cluster_type": cluster_type,
                    "adapter_class": self._adapter_classes[cluster_type].__name__,
                    "status": "registered",
                }
        return info


# Global adapter registry instance
_global_adapter_registry = AdapterRegistry()


def get_global_adapter_registry() -> AdapterRegistry:
    """Get the global adapter registry instance.

    Returns:
        Global AdapterRegistry instance
    """
    return _global_adapter_registry


def get_adapter(cluster_type: str) -> BaseAdapter:
    """Get adapter for a cluster type from global registry.

    Args:
        cluster_type: Cluster type identifier

    Returns:
        Adapter instance
    """
    return _global_adapter_registry.get_adapter(cluster_type)
