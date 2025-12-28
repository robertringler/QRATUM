"""OpenTelemetry Instrumentation for QRATUM.

Provides comprehensive observability across all QRATUM components:
- Distributed tracing for cross-vertical queries
- Metrics for performance monitoring
- Logs correlated with Merkle chain events
- OTLP exporters to prevent memory leaks through proper batching and flushing
"""

import json
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

# Optional OpenTelemetry imports for production OTLP export
try:
    from opentelemetry import metrics, trace
    from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    print("Warning: OpenTelemetry not available, using in-memory fallback")


class OTelInstrumentation:
    """OpenTelemetry instrumentation for QRATUM.

    Integrates production OpenTelemetry SDK with OTLP exporters for proper
    observability without memory leaks. Uses batching and periodic flushing.

    If OpenTelemetry is not available, falls back to in-memory storage.
    """

    def __init__(
        self,
        service_name: str,
        otlp_endpoint: Optional[str] = None,
        export_interval_ms: int = 5000,
        max_queue_size: int = 2048,
        max_export_batch_size: int = 512,
    ):
        """Initialize instrumentation.

        Args:
            service_name: Name of the service (e.g., "qratum-platform", "aethernet")
            otlp_endpoint: OTLP collector endpoint (e.g., "http://localhost:4317")
            export_interval_ms: Metrics export interval in milliseconds
            max_queue_size: Maximum queue size for batching
            max_export_batch_size: Maximum batch size for export
        """
        self.service_name = service_name
        self.enabled = True
        self.otlp_endpoint = otlp_endpoint or "http://localhost:4317"
        self.export_interval_ms = export_interval_ms
        self.max_queue_size = max_queue_size
        self.max_export_batch_size = max_export_batch_size

        # In-memory fallback storage
        self.traces = []
        self.metrics = []
        self.logs = []
        self._lock = threading.Lock()  # Thread safety for in-memory storage

        # Initialize OpenTelemetry if available
        if OTEL_AVAILABLE and otlp_endpoint:
            self._init_otel_exporters()
        else:
            self.tracer = None
            self.meter = None
            print(f"Using in-memory observability for {service_name}")

    def _init_otel_exporters(self):
        """Initialize OpenTelemetry with OTLP exporters.

        Sets up proper batching and periodic export to prevent memory leaks.
        """
        try:
            # Create resource for service identification
            resource = Resource.create(
                {
                    "service.name": self.service_name,
                    "service.version": "1.0.0",
                }
            )

            # Initialize trace provider with OTLP exporter
            trace_exporter = OTLPSpanExporter(
                endpoint=self.otlp_endpoint,
                insecure=True,  # Use TLS in production
            )

            # Use BatchSpanProcessor to batch and periodically export spans
            # This prevents memory leaks by limiting queue size
            span_processor = BatchSpanProcessor(
                trace_exporter,
                max_queue_size=self.max_queue_size,
                max_export_batch_size=self.max_export_batch_size,
                schedule_delay_millis=self.export_interval_ms,
            )

            tracer_provider = TracerProvider(resource=resource)
            tracer_provider.add_span_processor(span_processor)
            trace.set_tracer_provider(tracer_provider)

            self.tracer = trace.get_tracer(__name__)

            # Initialize metrics provider with OTLP exporter
            metric_exporter = OTLPMetricExporter(
                endpoint=self.otlp_endpoint,
                insecure=True,  # Use TLS in production
            )

            # Use PeriodicExportingMetricReader to periodically export metrics
            metric_reader = PeriodicExportingMetricReader(
                metric_exporter,
                export_interval_millis=self.export_interval_ms,
            )

            meter_provider = MeterProvider(
                resource=resource,
                metric_readers=[metric_reader],
            )
            metrics.set_meter_provider(meter_provider)

            self.meter = metrics.get_meter(__name__)

            print(f"OTLP exporters initialized for {self.service_name} -> {self.otlp_endpoint}")

        except Exception as e:
            print(f"Failed to initialize OTLP exporters: {e}")
            self.tracer = None
            self.meter = None

    @contextmanager
    def trace_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Create a trace span.

        Args:
            name: Span name
            attributes: Span attributes

        Example:
            with otel.trace_span("execute_contract", {"contract_id": "123"}):
                # ... work ...
                pass
        """
        if attributes is None:
            attributes = {}

        # Use OpenTelemetry tracer if available
        if self.tracer is not None:
            with self.tracer.start_as_current_span(name) as span:
                for key, value in attributes.items():
                    span.set_attribute(key, value)
                yield span
        else:
            # Fallback to in-memory tracking
            start_time = time.time()
            span_id = f"span_{len(self.traces)}"

            try:
                yield span_id
            finally:
                end_time = time.time()
                duration_ms = (end_time - start_time) * 1000

                span = {
                    "span_id": span_id,
                    "name": name,
                    "service": self.service_name,
                    "start_time": start_time,
                    "duration_ms": duration_ms,
                    "attributes": attributes,
                }

                if self.enabled:
                    # Limit in-memory storage to prevent leaks (thread-safe)
                    with self._lock:
                        if len(self.traces) >= self.max_queue_size:
                            self.traces = self.traces[-self.max_queue_size // 2 :]
                        self.traces.append(span)

    def record_metric(
        self, name: str, value: float, unit: str = "", labels: Optional[Dict[str, str]] = None
    ):
        """Record a metric.

        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            labels: Metric labels/tags
        """
        if labels is None:
            labels = {}

        # Use OpenTelemetry meter if available
        if self.meter is not None:
            try:
                counter = self.meter.create_counter(
                    name=name,
                    unit=unit,
                    description=f"Metric: {name}",
                )
                counter.add(value, labels)
            except Exception as e:
                print(f"Failed to record metric: {e}")
        else:
            # Fallback to in-memory tracking
            metric = {
                "name": name,
                "value": value,
                "unit": unit,
                "labels": labels,
                "service": self.service_name,
                "timestamp": time.time(),
            }

            if self.enabled:
                # Limit in-memory storage to prevent leaks
                if len(self.metrics) >= self.max_queue_size:
                    self.metrics = self.metrics[-self.max_queue_size // 2 :]
                self.metrics.append(metric)

    def log(self, level: str, message: str, attributes: Optional[Dict[str, Any]] = None):
        """Log a message.

        Args:
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Log message
            attributes: Additional attributes
        """
        if attributes is None:
            attributes = {}

        log_entry = {
            "level": level,
            "message": message,
            "service": self.service_name,
            "timestamp": time.time(),
            "attributes": attributes,
        }

        if self.enabled:
            # Limit in-memory storage to prevent leaks (thread-safe)
            with self._lock:
                if len(self.logs) >= self.max_queue_size:
                    self.logs = self.logs[-self.max_queue_size // 2 :]
                self.logs.append(log_entry)

    def flush(self):
        """Manually flush all pending telemetry data.

        Forces immediate export of batched data. Call before shutdown.
        """
        if OTEL_AVAILABLE:
            try:
                # Flush traces
                if hasattr(trace.get_tracer_provider(), "force_flush"):
                    trace.get_tracer_provider().force_flush()

                # Flush metrics
                if hasattr(metrics.get_meter_provider(), "force_flush"):
                    metrics.get_meter_provider().force_flush()

                print(f"Flushed telemetry data for {self.service_name}")
            except Exception as e:
                print(f"Failed to flush telemetry: {e}")

    def shutdown(self):
        """Shutdown telemetry and release resources.

        Call when application is shutting down to ensure all data is exported.
        """
        self.flush()

        if OTEL_AVAILABLE:
            try:
                if hasattr(trace.get_tracer_provider(), "shutdown"):
                    trace.get_tracer_provider().shutdown()

                if hasattr(metrics.get_meter_provider(), "shutdown"):
                    metrics.get_meter_provider().shutdown()

                print(f"Shutdown telemetry for {self.service_name}")
            except Exception as e:
                print(f"Failed to shutdown telemetry: {e}")

    def export_traces(self) -> str:
        """Export traces in OTLP JSON format (in-memory fallback only)."""
        with self._lock:
            return json.dumps(self.traces[-100:], indent=2)  # Last 100 traces

    def export_metrics(self) -> str:
        """Export metrics in OTLP JSON format (in-memory fallback only)."""
        with self._lock:
            return json.dumps(self.metrics[-100:], indent=2)  # Last 100 metrics

    def export_logs(self) -> str:
        """Export logs in OTLP JSON format (in-memory fallback only)."""
        with self._lock:
            return json.dumps(self.logs[-100:], indent=2)  # Last 100 logs


# Singleton instance
_default_otel: Optional[OTelInstrumentation] = None


def get_otel(service_name: str = "qratum") -> OTelInstrumentation:
    """Get or create the default OTEL instrumentation."""
    global _default_otel
    if _default_otel is None:
        _default_otel = OTelInstrumentation(service_name)
    return _default_otel


# Decorators for automatic instrumentation
def trace(name: Optional[str] = None):
    """Decorator to automatically trace a function.

    Example:
        @trace("my_function")
        def my_function(x, y):
            return x + y
    """

    def decorator(func):
        span_name = name or func.__name__

        def wrapper(*args, **kwargs):
            otel = get_otel()
            with otel.trace_span(span_name):
                return func(*args, **kwargs)

        return wrapper

    return decorator


def record_duration(metric_name: str):
    """Decorator to record function duration as a metric.

    Example:
        @record_duration("contract_execution_duration_ms")
        def execute_contract(contract):
            # ... work ...
            pass
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            otel = get_otel()
            start = time.time()
            try:
                return func(*args, **kwargs)
            finally:
                duration = (time.time() - start) * 1000
                otel.record_metric(metric_name, duration, "ms")

        return wrapper

    return decorator


# Example usage
if __name__ == "__main__":
    # Initialize instrumentation
    otel = OTelInstrumentation("qratum-demo")

    # Trace a span
    with otel.trace_span("demo_operation", {"user": "alice", "task": "simulate"}):
        time.sleep(0.1)  # Simulate work

        # Record metrics
        otel.record_metric("contracts_executed", 1, "count", {"vertical": "QUASIM"})
        otel.record_metric("execution_duration", 100.5, "ms", {"vertical": "QUASIM"})

        # Log messages
        otel.log("INFO", "Contract execution started", {"contract_id": "123"})
        otel.log(
            "INFO", "Contract execution completed", {"contract_id": "123", "status": "success"}
        )

    # Export observability data
    print("=== TRACES ===")
    print(otel.export_traces())
    print("\n=== METRICS ===")
    print(otel.export_metrics())
    print("\n=== LOGS ===")
    print(otel.export_logs())
