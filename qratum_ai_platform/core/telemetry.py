"""QRATUM Telemetry and Observability Framework.

Provides comprehensive telemetry collection, metrics emission, and
distributed tracing for quantum simulation operations.

Designed for production observability with support for:
- Prometheus metrics
- OpenTelemetry traces
- Structured logging
- Real-time monitoring dashboards

Classification: UNCLASSIFIED // CUI
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from typing import Any, Callable, Dict, Generator, List, Optional, TypeVar

__all__ = [
    "MetricType",
    "Metric",
    "Counter",
    "Gauge",
    "Histogram",
    "Timer",
    "MetricsRegistry",
    "TelemetryExporter",
    "PrometheusExporter",
    "Span",
    "Tracer",
    "get_metrics_registry",
    "get_tracer",
    "timed",
    "counted",
]


class MetricType(Enum):
    """Types of metrics supported."""

    COUNTER = auto()  # Monotonically increasing counter
    GAUGE = auto()  # Value that can go up or down
    HISTOGRAM = auto()  # Distribution of values
    SUMMARY = auto()  # Similar to histogram with quantiles


@dataclass
class MetricLabel:
    """Label for metric identification."""

    name: str
    value: str


@dataclass
class Metric:
    """Base metric class.

    Attributes:
        name: Metric name (should follow Prometheus naming conventions)
        description: Human-readable description
        metric_type: Type of metric
        labels: Metric labels for dimensional analysis
        unit: Unit of measurement (e.g., 'seconds', 'bytes')
    """

    name: str
    description: str
    metric_type: MetricType
    labels: Dict[str, str] = field(default_factory=dict)
    unit: str = ""

    def with_labels(self, **labels: str) -> Metric:
        """Create metric variant with additional labels."""
        new_labels = {**self.labels, **labels}
        return Metric(
            name=self.name,
            description=self.description,
            metric_type=self.metric_type,
            labels=new_labels,
            unit=self.unit,
        )


class Counter:
    """Monotonically increasing counter metric.

    Thread-safe counter that can only be incremented.
    """

    def __init__(self, name: str, description: str, labels: Optional[Dict[str, str]] = None):
        """Initialize counter.

        Args:
            name: Metric name
            description: Metric description
            labels: Optional labels
        """
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
        self._lock = threading.Lock()

    def inc(self, value: float = 1.0) -> None:
        """Increment counter by value.

        Args:
            value: Amount to increment (must be positive)

        Raises:
            ValueError: If value is negative
        """
        if value < 0:
            raise ValueError("Counter can only be incremented")
        with self._lock:
            self._value += value

    def get(self) -> float:
        """Get current counter value."""
        with self._lock:
            return self._value

    @property
    def value(self) -> float:
        """Current counter value."""
        return self.get()


class Gauge:
    """Gauge metric that can increase or decrease.

    Thread-safe gauge for tracking values that fluctuate.
    """

    def __init__(self, name: str, description: str, labels: Optional[Dict[str, str]] = None):
        """Initialize gauge.

        Args:
            name: Metric name
            description: Metric description
            labels: Optional labels
        """
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._value = 0.0
        self._lock = threading.Lock()

    def set(self, value: float) -> None:
        """Set gauge to specific value."""
        with self._lock:
            self._value = value

    def inc(self, value: float = 1.0) -> None:
        """Increment gauge."""
        with self._lock:
            self._value += value

    def dec(self, value: float = 1.0) -> None:
        """Decrement gauge."""
        with self._lock:
            self._value -= value

    def get(self) -> float:
        """Get current gauge value."""
        with self._lock:
            return self._value

    @property
    def value(self) -> float:
        """Current gauge value."""
        return self.get()

    @contextmanager
    def track_inprogress(self) -> Generator[None, None, None]:
        """Context manager to track in-progress operations."""
        self.inc()
        try:
            yield
        finally:
            self.dec()


class Histogram:
    """Histogram metric for tracking value distributions.

    Tracks observations and computes statistics over buckets.
    """

    DEFAULT_BUCKETS = (
        0.001,
        0.005,
        0.01,
        0.025,
        0.05,
        0.1,
        0.25,
        0.5,
        1.0,
        2.5,
        5.0,
        10.0,
        float("inf"),
    )

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[tuple] = None,
    ):
        """Initialize histogram.

        Args:
            name: Metric name
            description: Metric description
            labels: Optional labels
            buckets: Bucket boundaries (default: exponential)
        """
        self.name = name
        self.description = description
        self.labels = labels or {}
        self._buckets = buckets or self.DEFAULT_BUCKETS
        self._bucket_counts: Dict[float, int] = dict.fromkeys(self._buckets, 0)
        self._sum = 0.0
        self._count = 0
        self._lock = threading.Lock()

    def observe(self, value: float) -> None:
        """Record an observation.

        Args:
            value: Observed value
        """
        with self._lock:
            self._sum += value
            self._count += 1
            for bucket in self._buckets:
                if value <= bucket:
                    self._bucket_counts[bucket] += 1

    def get_sample(self) -> Dict[str, Any]:
        """Get histogram sample data."""
        with self._lock:
            return {
                "sum": self._sum,
                "count": self._count,
                "buckets": dict(self._bucket_counts),
            }

    @property
    def sum(self) -> float:
        """Sum of all observations."""
        with self._lock:
            return self._sum

    @property
    def count(self) -> int:
        """Count of observations."""
        with self._lock:
            return self._count

    def mean(self) -> float:
        """Mean of observations."""
        with self._lock:
            return self._sum / self._count if self._count > 0 else 0.0


class Timer:
    """Timer for measuring operation duration.

    Convenience wrapper around Histogram for timing operations.
    """

    def __init__(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None,
    ):
        """Initialize timer.

        Args:
            name: Metric name
            description: Metric description
            labels: Optional labels
        """
        self._histogram = Histogram(
            name=name,
            description=description,
            labels=labels,
            buckets=(
                0.001,
                0.005,
                0.01,
                0.025,
                0.05,
                0.1,
                0.25,
                0.5,
                1.0,
                2.5,
                5.0,
                10.0,
                30.0,
                60.0,
                float("inf"),
            ),
        )
        self._start_time: Optional[int] = None

    def start(self) -> Timer:
        """Start timer."""
        self._start_time = time.perf_counter_ns()
        return self

    def stop(self) -> float:
        """Stop timer and record duration.

        Returns:
            Duration in seconds
        """
        if self._start_time is None:
            raise RuntimeError("Timer not started")
        duration_ns = time.perf_counter_ns() - self._start_time
        duration_s = duration_ns / 1_000_000_000
        self._histogram.observe(duration_s)
        self._start_time = None
        return duration_s

    @contextmanager
    def time(self) -> Generator[None, None, None]:
        """Context manager for timing operations."""
        self.start()
        try:
            yield
        finally:
            self.stop()

    @property
    def histogram(self) -> Histogram:
        """Underlying histogram."""
        return self._histogram


class MetricsRegistry:
    """Central registry for all metrics.

    Thread-safe registry for managing metrics across the application.
    """

    _instance: Optional[MetricsRegistry] = None
    _lock = threading.Lock()

    def __init__(self):
        """Initialize metrics registry."""
        self._counters: Dict[str, Counter] = {}
        self._gauges: Dict[str, Gauge] = {}
        self._histograms: Dict[str, Histogram] = {}
        self._timers: Dict[str, Timer] = {}
        self._registry_lock = threading.Lock()
        self._exporters: List[TelemetryExporter] = []

        # Register default metrics
        self._register_default_metrics()

    @classmethod
    def get_instance(cls) -> MetricsRegistry:
        """Get singleton registry instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def _register_default_metrics(self) -> None:
        """Register default QRATUM metrics."""
        # Simulation metrics
        self.counter(
            "qratum_simulations_total",
            "Total number of quantum simulations executed",
        )
        self.counter(
            "qratum_simulation_errors_total",
            "Total number of simulation errors",
        )
        self.histogram(
            "qratum_simulation_duration_seconds",
            "Duration of quantum simulations",
        )
        self.gauge(
            "qratum_active_simulations",
            "Number of currently active simulations",
        )

        # Qubit metrics
        self.histogram(
            "qratum_circuit_qubits",
            "Number of qubits in simulated circuits",
        )
        self.histogram(
            "qratum_circuit_depth",
            "Depth of simulated circuits",
        )
        self.histogram(
            "qratum_circuit_gates",
            "Number of gates in simulated circuits",
        )

        # Backend metrics
        self.gauge(
            "qratum_backend_memory_bytes",
            "Memory usage by backend",
        )
        self.gauge(
            "qratum_backend_gpu_utilization",
            "GPU utilization percentage",
        )

        # Validation metrics
        self.counter(
            "qratum_validation_checks_total",
            "Total validation checks performed",
        )
        self.counter(
            "qratum_validation_failures_total",
            "Total validation failures",
        )

    def counter(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Counter:
        """Get or create a counter metric."""
        with self._registry_lock:
            key = self._make_key(name, labels)
            if key not in self._counters:
                self._counters[key] = Counter(name, description, labels)
            return self._counters[key]

    def gauge(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Gauge:
        """Get or create a gauge metric."""
        with self._registry_lock:
            key = self._make_key(name, labels)
            if key not in self._gauges:
                self._gauges[key] = Gauge(name, description, labels)
            return self._gauges[key]

    def histogram(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None,
        buckets: Optional[tuple] = None,
    ) -> Histogram:
        """Get or create a histogram metric."""
        with self._registry_lock:
            key = self._make_key(name, labels)
            if key not in self._histograms:
                self._histograms[key] = Histogram(name, description, labels, buckets)
            return self._histograms[key]

    def timer(
        self,
        name: str,
        description: str,
        labels: Optional[Dict[str, str]] = None,
    ) -> Timer:
        """Get or create a timer metric."""
        with self._registry_lock:
            key = self._make_key(name, labels)
            if key not in self._timers:
                self._timers[key] = Timer(name, description, labels)
            return self._timers[key]

    def _make_key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Create unique key for metric."""
        if labels:
            label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def add_exporter(self, exporter: TelemetryExporter) -> None:
        """Add telemetry exporter."""
        self._exporters.append(exporter)

    def export(self) -> Dict[str, Any]:
        """Export all metrics to dictionary."""
        result: Dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "counters": {},
            "gauges": {},
            "histograms": {},
        }

        with self._registry_lock:
            for key, counter in self._counters.items():
                result["counters"][key] = {
                    "name": counter.name,
                    "description": counter.description,
                    "labels": counter.labels,
                    "value": counter.get(),
                }

            for key, gauge in self._gauges.items():
                result["gauges"][key] = {
                    "name": gauge.name,
                    "description": gauge.description,
                    "labels": gauge.labels,
                    "value": gauge.get(),
                }

            for key, histogram in self._histograms.items():
                sample = histogram.get_sample()
                result["histograms"][key] = {
                    "name": histogram.name,
                    "description": histogram.description,
                    "labels": histogram.labels,
                    **sample,
                }

        return result


class TelemetryExporter(ABC):
    """Abstract base class for telemetry exporters."""

    @abstractmethod
    def export_metrics(self, metrics: Dict[str, Any]) -> None:
        """Export metrics to destination."""
        pass

    @abstractmethod
    def export_span(self, span: Span) -> None:
        """Export span to destination."""
        pass


class PrometheusExporter(TelemetryExporter):
    """Prometheus-compatible metrics exporter.

    Formats metrics in Prometheus exposition format.
    """

    def __init__(self, prefix: str = "qratum"):
        """Initialize Prometheus exporter.

        Args:
            prefix: Metric name prefix
        """
        self._prefix = prefix

    def export_metrics(self, metrics: Dict[str, Any]) -> None:
        """Export metrics (writes to prometheus format)."""
        # In production, this would push to Prometheus Pushgateway
        # or expose via HTTP endpoint
        pass

    def export_span(self, span: Span) -> None:
        """Export span (no-op for Prometheus)."""
        pass

    def format(self, registry: MetricsRegistry) -> str:
        """Format metrics in Prometheus exposition format.

        Args:
            registry: Metrics registry to format

        Returns:
            Prometheus exposition format string
        """
        lines: List[str] = []
        data = registry.export()

        # Counters
        for key, counter_data in data["counters"].items():
            name = counter_data["name"]
            desc = counter_data["description"]
            value = counter_data["value"]
            labels = counter_data["labels"]

            lines.append(f"# HELP {name} {desc}")
            lines.append(f"# TYPE {name} counter")
            label_str = self._format_labels(labels)
            lines.append(f"{name}{label_str} {value}")

        # Gauges
        for key, gauge_data in data["gauges"].items():
            name = gauge_data["name"]
            desc = gauge_data["description"]
            value = gauge_data["value"]
            labels = gauge_data["labels"]

            lines.append(f"# HELP {name} {desc}")
            lines.append(f"# TYPE {name} gauge")
            label_str = self._format_labels(labels)
            lines.append(f"{name}{label_str} {value}")

        # Histograms
        for key, hist_data in data["histograms"].items():
            name = hist_data["name"]
            desc = hist_data["description"]
            labels = hist_data["labels"]
            buckets = hist_data["buckets"]
            total_sum = hist_data["sum"]
            count = hist_data["count"]

            lines.append(f"# HELP {name} {desc}")
            lines.append(f"# TYPE {name} histogram")

            label_str = self._format_labels(labels)
            for le, bucket_count in sorted(buckets.items()):
                le_label = f'le="{le}"' if le != float("inf") else 'le="+Inf"'
                if label_str:
                    full_labels = f"{{{label_str[1:-1]},{le_label}}}"
                else:
                    full_labels = f"{{{le_label}}}"
                lines.append(f"{name}_bucket{full_labels} {bucket_count}")

            lines.append(f"{name}_sum{label_str} {total_sum}")
            lines.append(f"{name}_count{label_str} {count}")

        return "\n".join(lines)

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus."""
        if not labels:
            return ""
        pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(pairs) + "}"


@dataclass
class Span:
    """Distributed tracing span.

    Represents a unit of work in a distributed trace.
    """

    trace_id: str
    span_id: str
    name: str
    parent_span_id: Optional[str] = None
    start_time_ns: int = 0
    end_time_ns: int = 0
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "OK"

    @property
    def duration_ns(self) -> int:
        """Span duration in nanoseconds."""
        return self.end_time_ns - self.start_time_ns

    @property
    def duration_ms(self) -> float:
        """Span duration in milliseconds."""
        return self.duration_ns / 1_000_000

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """Add event to span."""
        self.events.append(
            {
                "name": name,
                "timestamp": time.perf_counter_ns(),
                "attributes": attributes or {},
            }
        )

    def set_attribute(self, key: str, value: Any) -> None:
        """Set span attribute."""
        self.attributes[key] = value

    def set_status(self, status: str, description: str = "") -> None:
        """Set span status."""
        self.status = status
        if description:
            self.attributes["status.description"] = description


class Tracer:
    """Distributed tracing tracer.

    Creates and manages spans for distributed tracing.
    """

    _instance: Optional[Tracer] = None
    _lock = threading.Lock()

    def __init__(self, service_name: str = "qratum"):
        """Initialize tracer.

        Args:
            service_name: Name of the service
        """
        self._service_name = service_name
        self._active_spans: Dict[int, List[Span]] = {}
        self._spans_lock = threading.Lock()
        self._exporters: List[TelemetryExporter] = []

    @classmethod
    def get_instance(cls, service_name: str = "qratum") -> Tracer:
        """Get singleton tracer instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls(service_name)
        return cls._instance

    @contextmanager
    def start_span(
        self,
        name: str,
        parent: Optional[Span] = None,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> Generator[Span, None, None]:
        """Start a new span.

        Args:
            name: Span name
            parent: Optional parent span
            attributes: Span attributes

        Yields:
            New span
        """
        import uuid

        # Generate IDs
        if parent:
            trace_id = parent.trace_id
            parent_span_id = parent.span_id
        else:
            trace_id = str(uuid.uuid4())
            parent_span_id = None

        span = Span(
            trace_id=trace_id,
            span_id=str(uuid.uuid4())[:16],
            name=name,
            parent_span_id=parent_span_id,
            start_time_ns=time.perf_counter_ns(),
            attributes=attributes or {},
        )

        # Push to thread-local stack
        tid = threading.get_ident()
        with self._spans_lock:
            if tid not in self._active_spans:
                self._active_spans[tid] = []
            self._active_spans[tid].append(span)

        try:
            yield span
            span.status = "OK"
        except Exception as e:
            span.status = "ERROR"
            span.set_attribute("error.type", type(e).__name__)
            span.set_attribute("error.message", str(e))
            raise
        finally:
            span.end_time_ns = time.perf_counter_ns()

            # Pop from stack
            with self._spans_lock:
                if tid in self._active_spans:
                    self._active_spans[tid].pop()
                    if not self._active_spans[tid]:
                        del self._active_spans[tid]

            # Export span
            for exporter in self._exporters:
                exporter.export_span(span)

    def current_span(self) -> Optional[Span]:
        """Get current active span."""
        tid = threading.get_ident()
        with self._spans_lock:
            if tid in self._active_spans and self._active_spans[tid]:
                return self._active_spans[tid][-1]
        return None

    def add_exporter(self, exporter: TelemetryExporter) -> None:
        """Add span exporter."""
        self._exporters.append(exporter)


def get_metrics_registry() -> MetricsRegistry:
    """Get global metrics registry."""
    return MetricsRegistry.get_instance()


def get_tracer(service_name: str = "qratum") -> Tracer:
    """Get global tracer."""
    return Tracer.get_instance(service_name)


T = TypeVar("T")


def timed(
    name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to time function execution.

    Args:
        name: Timer name (defaults to function name)
        labels: Timer labels

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        timer_name = name or f"qratum_{func.__name__}_duration_seconds"
        timer = get_metrics_registry().timer(
            timer_name,
            f"Duration of {func.__name__}",
            labels,
        )

        def wrapper(*args: Any, **kwargs: Any) -> T:
            with timer.time():
                return func(*args, **kwargs)

        return wrapper

    return decorator


def counted(
    name: Optional[str] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Decorator to count function invocations.

    Args:
        name: Counter name (defaults to function name)
        labels: Counter labels

    Returns:
        Decorated function
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        counter_name = name or f"qratum_{func.__name__}_total"
        counter = get_metrics_registry().counter(
            counter_name,
            f"Number of calls to {func.__name__}",
            labels,
        )

        def wrapper(*args: Any, **kwargs: Any) -> T:
            counter.inc()
            return func(*args, **kwargs)

        return wrapper

    return decorator
