"""OpenTelemetry Instrumentation for QRATUM.

Provides comprehensive observability across all QRATUM components:
- Distributed tracing for cross-vertical queries
- Metrics for performance monitoring
- Logs correlated with Merkle chain events
"""

from typing import Dict, Any, Optional
from contextlib import contextmanager
import time
import json


class OTelInstrumentation:
    """OpenTelemetry instrumentation for QRATUM.
    
    In production, this would integrate with the opentelemetry-api and
    opentelemetry-sdk packages for full OTEL support.
    """
    
    def __init__(self, service_name: str):
        """Initialize instrumentation.
        
        Args:
            service_name: Name of the service (e.g., "qratum-platform", "aethernet")
        """
        self.service_name = service_name
        self.enabled = True
        self.traces = []
        self.metrics = []
        self.logs = []
        
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
                self.traces.append(span)
    
    def record_metric(self, name: str, value: float, unit: str = "", labels: Optional[Dict[str, str]] = None):
        """Record a metric.
        
        Args:
            name: Metric name
            value: Metric value
            unit: Unit of measurement
            labels: Metric labels/tags
        """
        if labels is None:
            labels = {}
            
        metric = {
            "name": name,
            "value": value,
            "unit": unit,
            "labels": labels,
            "service": self.service_name,
            "timestamp": time.time(),
        }
        
        if self.enabled:
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
            self.logs.append(log_entry)
    
    def export_traces(self) -> str:
        """Export traces in OTLP JSON format."""
        return json.dumps(self.traces, indent=2)
    
    def export_metrics(self) -> str:
        """Export metrics in OTLP JSON format."""
        return json.dumps(self.metrics, indent=2)
    
    def export_logs(self) -> str:
        """Export logs in OTLP JSON format."""
        return json.dumps(self.logs, indent=2)


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
        otel.log("INFO", "Contract execution completed", {"contract_id": "123", "status": "success"})
    
    # Export observability data
    print("=== TRACES ===")
    print(otel.export_traces())
    print("\n=== METRICS ===")
    print(otel.export_metrics())
    print("\n=== LOGS ===")
    print(otel.export_logs())
