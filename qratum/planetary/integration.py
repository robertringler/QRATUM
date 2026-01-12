"""
Cross-Domain Integration Module

Models QRATUM adoption across energy, transport, healthcare, finance,
logistics, and communications with interoperability and real-time KPI monitoring.

Version: 1.0.0
Status: Production
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DomainType(Enum):
    """Cross-domain sector types."""

    ENERGY = "energy"
    TRANSPORT = "transport"
    HEALTHCARE = "healthcare"
    FINANCE = "finance"
    LOGISTICS = "logistics"
    COMMUNICATIONS = "communications"
    MANUFACTURING = "manufacturing"
    AGRICULTURE = "agriculture"


class ComplianceLevel(Enum):
    """Compliance certification levels."""

    NONE = "none"
    BASIC = "basic"
    STANDARD = "standard"
    ADVANCED = "advanced"
    CERTIFIED = "certified"


@dataclass
class KPIMetric:
    """Key Performance Indicator metric.

    Attributes:
        metric_id: Unique metric identifier
        name: Metric name
        value: Current value
        target: Target value
        unit: Unit of measurement
        timestamp: Last updated timestamp
        trend: Recent trend (up/down/stable)
    """

    metric_id: str
    name: str
    value: float
    target: float
    unit: str = ""
    timestamp: str = ""
    trend: str = "stable"

    def __post_init__(self) -> None:
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def update(self, new_value: float) -> None:
        """Update metric value and trend.

        Args:
            new_value: New metric value
        """
        if new_value > self.value:
            self.trend = "up"
        elif new_value < self.value:
            self.trend = "down"
        else:
            self.trend = "stable"
        self.value = new_value
        self.timestamp = datetime.now(timezone.utc).isoformat()

    def is_on_target(self) -> bool:
        """Check if metric is meeting target.

        Returns:
            True if value meets or exceeds target
        """
        return self.value >= self.target

    def to_dict(self) -> dict[str, Any]:
        """Serialize metric to dictionary."""
        return {
            "metric_id": self.metric_id,
            "name": self.name,
            "value": self.value,
            "target": self.target,
            "unit": self.unit,
            "timestamp": self.timestamp,
            "trend": self.trend,
            "on_target": self.is_on_target(),
        }


@dataclass
class KPIMonitor:
    """Real-time KPI monitoring system.

    Attributes:
        monitor_id: Unique monitor identifier
        domain: Domain being monitored
        metrics: Dictionary of KPI metrics
        alerts: List of active alerts
        refresh_interval_sec: Refresh interval in seconds
    """

    monitor_id: str
    domain: DomainType
    metrics: dict[str, KPIMetric] = field(default_factory=dict)
    alerts: list[dict[str, Any]] = field(default_factory=list)
    refresh_interval_sec: int = 60

    def add_metric(
        self,
        name: str,
        target: float,
        unit: str = "",
        initial_value: float = 0.0,
    ) -> KPIMetric:
        """Add a KPI metric.

        Args:
            name: Metric name
            target: Target value
            unit: Unit of measurement
            initial_value: Initial value

        Returns:
            Created metric
        """
        metric_id = f"{self.domain.value}_{name.lower().replace(' ', '_')}"
        metric = KPIMetric(
            metric_id=metric_id,
            name=name,
            value=initial_value,
            target=target,
            unit=unit,
        )
        self.metrics[metric_id] = metric
        return metric

    def update_metric(self, metric_id: str, value: float) -> None:
        """Update a metric value.

        Args:
            metric_id: Metric identifier
            value: New value
        """
        if metric_id in self.metrics:
            self.metrics[metric_id].update(value)

            # Check for alerts
            metric = self.metrics[metric_id]
            if not metric.is_on_target():
                self._create_alert(metric)

    def _create_alert(self, metric: KPIMetric) -> None:
        """Create an alert for off-target metric.

        Args:
            metric: The metric triggering the alert
        """
        alert = {
            "alert_id": f"alert_{len(self.alerts):04d}",
            "metric_id": metric.metric_id,
            "metric_name": metric.name,
            "current_value": metric.value,
            "target_value": metric.target,
            "gap": metric.target - metric.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolved": False,
        }
        self.alerts.append(alert)

    def get_dashboard(self) -> dict[str, Any]:
        """Get KPI dashboard summary.

        Returns:
            Dashboard data
        """
        total_metrics = len(self.metrics)
        on_target = sum(1 for m in self.metrics.values() if m.is_on_target())
        active_alerts = sum(1 for a in self.alerts if not a.get("resolved", True))

        return {
            "monitor_id": self.monitor_id,
            "domain": self.domain.value,
            "total_metrics": total_metrics,
            "metrics_on_target": on_target,
            "metrics_off_target": total_metrics - on_target,
            "active_alerts": active_alerts,
            "health_score": (on_target / total_metrics * 100) if total_metrics > 0 else 0,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize monitor to dictionary."""
        return {
            "monitor_id": self.monitor_id,
            "domain": self.domain.value,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "alerts": self.alerts,
            "refresh_interval_sec": self.refresh_interval_sec,
        }


@dataclass
class SectorAdapter:
    """Adapter for integrating with a specific sector.

    Attributes:
        adapter_id: Unique adapter identifier
        domain: Target domain
        api_endpoints: API endpoints for integration
        data_formats: Supported data formats
        compliance_level: Current compliance level
        active_connections: Number of active connections
        throughput_mbps: Current throughput in Mbps
        kpi_monitor: KPI monitoring instance
    """

    adapter_id: str
    domain: DomainType
    api_endpoints: list[str] = field(default_factory=list)
    data_formats: list[str] = field(default_factory=list)
    compliance_level: ComplianceLevel = ComplianceLevel.NONE
    active_connections: int = 0
    throughput_mbps: float = 0.0
    kpi_monitor: KPIMonitor | None = None

    def __post_init__(self) -> None:
        if not self.data_formats:
            self.data_formats = ["json", "cbor", "protobuf"]
        if self.kpi_monitor is None:
            self.kpi_monitor = KPIMonitor(
                monitor_id=f"kpi_{self.adapter_id}",
                domain=self.domain,
            )
            self._initialize_default_kpis()

    def _initialize_default_kpis(self) -> None:
        """Initialize default KPIs for the domain."""
        if self.kpi_monitor is None:
            raise RuntimeError("KPI monitor not initialized")

        # Common KPIs
        self.kpi_monitor.add_metric("Uptime", target=99.9, unit="%", initial_value=99.0)
        self.kpi_monitor.add_metric("Latency", target=100, unit="ms", initial_value=150)
        self.kpi_monitor.add_metric("Error Rate", target=0.1, unit="%", initial_value=0.05)

        # Domain-specific KPIs
        if self.domain == DomainType.ENERGY:
            self.kpi_monitor.add_metric("Grid Efficiency", target=95, unit="%", initial_value=92)
            self.kpi_monitor.add_metric("Carbon Reduction", target=50, unit="%", initial_value=35)
        elif self.domain == DomainType.HEALTHCARE:
            self.kpi_monitor.add_metric("Data Compliance", target=100, unit="%", initial_value=98)
            self.kpi_monitor.add_metric(
                "Processing Time", target=1000, unit="ms", initial_value=800
            )
        elif self.domain == DomainType.FINANCE:
            self.kpi_monitor.add_metric("Transaction Speed", target=50, unit="ms", initial_value=45)
            self.kpi_monitor.add_metric("Fraud Detection", target=99, unit="%", initial_value=97)
        elif self.domain == DomainType.TRANSPORT:
            self.kpi_monitor.add_metric("Route Optimization", target=90, unit="%", initial_value=85)
            self.kpi_monitor.add_metric("Fuel Efficiency", target=20, unit="%", initial_value=15)
        elif self.domain == DomainType.LOGISTICS:
            self.kpi_monitor.add_metric("Delivery Accuracy", target=99, unit="%", initial_value=96)
            self.kpi_monitor.add_metric(
                "Inventory Accuracy", target=99.5, unit="%", initial_value=98
            )
        elif self.domain == DomainType.COMMUNICATIONS:
            self.kpi_monitor.add_metric("Bandwidth Util", target=80, unit="%", initial_value=65)
            self.kpi_monitor.add_metric("Signal Quality", target=95, unit="%", initial_value=92)

    def connect(self, endpoint: str) -> bool:
        """Connect to an endpoint.

        Args:
            endpoint: Endpoint URL

        Returns:
            True if connection successful
        """
        if endpoint not in self.api_endpoints:
            self.api_endpoints.append(endpoint)
        self.active_connections += 1
        return True

    def disconnect(self, endpoint: str) -> bool:
        """Disconnect from an endpoint.

        Args:
            endpoint: Endpoint URL

        Returns:
            True if disconnection successful
        """
        if endpoint in self.api_endpoints:
            self.api_endpoints.remove(endpoint)
            self.active_connections = max(0, self.active_connections - 1)
            return True
        return False

    def transform_data(self, data: dict[str, Any], target_format: str) -> dict[str, Any]:
        """Transform data to target format.

        Args:
            data: Input data
            target_format: Target format

        Returns:
            Transformed data
        """
        if target_format not in self.data_formats:
            return {"error": f"Unsupported format: {target_format}"}

        return {
            "format": target_format,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "adapter_id": self.adapter_id,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize adapter to dictionary."""
        return {
            "adapter_id": self.adapter_id,
            "domain": self.domain.value,
            "api_endpoints": self.api_endpoints,
            "data_formats": self.data_formats,
            "compliance_level": self.compliance_level.value,
            "active_connections": self.active_connections,
            "throughput_mbps": self.throughput_mbps,
            "kpi_monitor": self.kpi_monitor.to_dict() if self.kpi_monitor else None,
        }


@dataclass
class InteroperabilityLayer:
    """Layer for cross-domain interoperability.

    Attributes:
        layer_id: Unique layer identifier
        adapters: Dictionary of sector adapters
        routing_table: Routing rules between domains
        transformations: Data transformation rules
    """

    layer_id: str
    adapters: dict[DomainType, SectorAdapter] = field(default_factory=dict)
    routing_table: dict[str, list[DomainType]] = field(default_factory=dict)
    transformations: dict[str, dict[str, Any]] = field(default_factory=dict)

    def add_adapter(self, adapter: SectorAdapter) -> None:
        """Add a sector adapter.

        Args:
            adapter: Sector adapter to add
        """
        self.adapters[adapter.domain] = adapter

    def route_data(
        self,
        source: DomainType,
        data: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """Route data from source to connected domains.

        Args:
            source: Source domain
            data: Data to route

        Returns:
            List of routed data packets
        """
        results = []
        targets = self.routing_table.get(source.value, [])

        for target in targets:
            if target in self.adapters:
                adapter = self.adapters[target]
                transformed = adapter.transform_data(data, "json")
                results.append(
                    {
                        "source": source.value,
                        "target": target.value,
                        "data": transformed,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                    }
                )

        return results

    def define_route(self, source: DomainType, targets: list[DomainType]) -> None:
        """Define routing rule.

        Args:
            source: Source domain
            targets: Target domains
        """
        self.routing_table[source.value] = targets

    def get_connectivity_matrix(self) -> dict[str, list[str]]:
        """Get connectivity matrix.

        Returns:
            Dictionary showing connections between domains
        """
        matrix = {}
        for source, targets in self.routing_table.items():
            matrix[source] = [t.value for t in targets]
        return matrix

    def to_dict(self) -> dict[str, Any]:
        """Serialize layer to dictionary."""
        return {
            "layer_id": self.layer_id,
            "adapters": {k.value: v.to_dict() for k, v in self.adapters.items()},
            "routing_table": {k: [t.value for t in v] for k, v in self.routing_table.items()},
            "transformations": self.transformations,
        }


class DomainIntegration:
    """Main domain integration manager.

    Manages cross-domain integration across all sectors.

    Attributes:
        integration_id: Unique integration identifier
        interop_layer: Interoperability layer
        compliance_requirements: Compliance requirements by domain
        use_cases: Defined use cases
    """

    def __init__(self, integration_id: str | None = None) -> None:
        """Initialize domain integration.

        Args:
            integration_id: Optional integration ID
        """
        self.integration_id = integration_id or self._generate_id()
        self.interop_layer = InteroperabilityLayer(layer_id=f"interop_{self.integration_id}")
        self.compliance_requirements: dict[DomainType, list[str]] = {}
        self.use_cases: list[dict[str, Any]] = []
        self.created_at = datetime.now(timezone.utc).isoformat()

        self._initialize_domains()

    def _generate_id(self) -> str:
        """Generate unique integration ID."""
        timestamp = datetime.now(timezone.utc).isoformat()
        return f"domain_{hashlib.sha256(timestamp.encode()).hexdigest()[:12]}"

    def _initialize_domains(self) -> None:
        """Initialize all domain adapters."""
        for domain in DomainType:
            adapter = SectorAdapter(
                adapter_id=f"adapter_{domain.value}",
                domain=domain,
            )
            self.interop_layer.add_adapter(adapter)

        # Define default routes
        self._setup_default_routes()
        self._setup_compliance_requirements()

    def _setup_default_routes(self) -> None:
        """Setup default routing between domains."""
        # Energy routes to transport, manufacturing
        self.interop_layer.define_route(
            DomainType.ENERGY,
            [DomainType.TRANSPORT, DomainType.MANUFACTURING],
        )
        # Healthcare routes to finance (billing), logistics (supply chain)
        self.interop_layer.define_route(
            DomainType.HEALTHCARE,
            [DomainType.FINANCE, DomainType.LOGISTICS],
        )
        # Finance routes to all other domains
        self.interop_layer.define_route(
            DomainType.FINANCE,
            [
                DomainType.HEALTHCARE,
                DomainType.ENERGY,
                DomainType.LOGISTICS,
                DomainType.TRANSPORT,
            ],
        )
        # Logistics routes to transport, manufacturing, healthcare
        self.interop_layer.define_route(
            DomainType.LOGISTICS,
            [DomainType.TRANSPORT, DomainType.MANUFACTURING, DomainType.HEALTHCARE],
        )
        # Communications routes to all domains
        self.interop_layer.define_route(
            DomainType.COMMUNICATIONS,
            list(DomainType),
        )

    def _setup_compliance_requirements(self) -> None:
        """Setup compliance requirements by domain."""
        self.compliance_requirements = {
            DomainType.HEALTHCARE: ["HIPAA", "GDPR", "HL7", "FHIR"],
            DomainType.FINANCE: ["PCI-DSS", "SOX", "GDPR", "AML"],
            DomainType.ENERGY: ["NERC-CIP", "IEC-62351", "ISO-50001"],
            DomainType.TRANSPORT: ["ISO-39001", "DOT", "FAA"],
            DomainType.LOGISTICS: ["ISO-28000", "C-TPAT", "AEO"],
            DomainType.COMMUNICATIONS: ["FCC", "GDPR", "CPNI"],
            DomainType.MANUFACTURING: ["ISO-9001", "ISO-14001", "OSHA"],
            DomainType.AGRICULTURE: ["USDA", "FDA", "GAP"],
        }

    def create_use_case(
        self,
        name: str,
        description: str,
        domains: list[DomainType],
        kpis: dict[str, float],
    ) -> dict[str, Any]:
        """Create a cross-domain use case.

        Args:
            name: Use case name
            description: Use case description
            domains: Involved domains
            kpis: Expected KPI improvements

        Returns:
            Use case definition
        """
        use_case = {
            "use_case_id": f"uc_{hashlib.sha256(name.encode()).hexdigest()[:8]}",
            "name": name,
            "description": description,
            "domains": [d.value for d in domains],
            "kpis": kpis,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "defined",
        }
        self.use_cases.append(use_case)
        return use_case

    def simulate_use_case(self, use_case_id: str, duration_days: int = 30) -> dict[str, Any]:
        """Simulate a use case and generate metrics.

        Args:
            use_case_id: Use case identifier
            duration_days: Simulation duration in days

        Returns:
            Simulation results
        """
        use_case = next(
            (uc for uc in self.use_cases if uc["use_case_id"] == use_case_id),
            None,
        )

        if not use_case:
            return {"error": f"Use case not found: {use_case_id}"}

        # Simulate metrics improvement
        results = {
            "use_case_id": use_case_id,
            "simulation_days": duration_days,
            "domains_involved": use_case["domains"],
            "metrics": {},
            "value_created": 0.0,
        }

        for kpi, target in use_case.get("kpis", {}).items():
            # Simulate gradual improvement
            improvement = target * 0.7  # Achieve 70% of target
            results["metrics"][kpi] = {
                "baseline": 0,
                "achieved": improvement,
                "target": target,
                "achievement_rate": 70,
            }
            results["value_created"] += improvement * 1000  # Simplified value calculation

        return results

    def get_compliance_status(self) -> dict[str, Any]:
        """Get compliance status across all domains.

        Returns:
            Compliance status report
        """
        status = {}
        for domain, adapter in self.interop_layer.adapters.items():
            requirements = self.compliance_requirements.get(domain, [])
            status[domain.value] = {
                "compliance_level": adapter.compliance_level.value,
                "requirements": requirements,
                "requirements_count": len(requirements),
            }
        return status

    def get_statistics(self) -> dict[str, Any]:
        """Get integration statistics.

        Returns:
            Integration statistics
        """
        total_connections = sum(a.active_connections for a in self.interop_layer.adapters.values())
        total_throughput = sum(a.throughput_mbps for a in self.interop_layer.adapters.values())

        return {
            "integration_id": self.integration_id,
            "domains_active": len(self.interop_layer.adapters),
            "total_connections": total_connections,
            "total_throughput_mbps": total_throughput,
            "use_cases_defined": len(self.use_cases),
            "routes_defined": len(self.interop_layer.routing_table),
            "created_at": self.created_at,
        }

    def to_dict(self) -> dict[str, Any]:
        """Serialize integration to dictionary."""
        return {
            "integration_id": self.integration_id,
            "interop_layer": self.interop_layer.to_dict(),
            "compliance_requirements": {
                k.value: v for k, v in self.compliance_requirements.items()
            },
            "use_cases": self.use_cases,
            "created_at": self.created_at,
        }
