"""Tests for OTLP observability exporters.

Validates that OTLP exporters prevent memory leaks through proper batching
and periodic flushing.
"""

import pytest
import time


def test_otel_instrumentation_initialization():
    """Test OTLP instrumentation initializes correctly."""
    from observability.otel.instrumentation import OTelInstrumentation
    
    otel = OTelInstrumentation(
        service_name="test-service",
        otlp_endpoint=None,  # Use in-memory fallback
        export_interval_ms=1000,
        max_queue_size=100,
    )
    
    assert otel.service_name == "test-service"
    assert otel.enabled is True
    assert otel.max_queue_size == 100


def test_trace_span_creation():
    """Test that trace spans can be created."""
    from observability.otel.instrumentation import OTelInstrumentation
    
    otel = OTelInstrumentation("test-service", otlp_endpoint=None)
    
    with otel.trace_span("test_operation", {"user": "test"}):
        time.sleep(0.01)  # Simulate work
    
    # Verify span was recorded (in-memory fallback)
    assert len(otel.traces) > 0
    assert otel.traces[0]["name"] == "test_operation"
    assert otel.traces[0]["attributes"]["user"] == "test"
    assert otel.traces[0]["duration_ms"] > 0


def test_metric_recording():
    """Test that metrics can be recorded."""
    from observability.otel.instrumentation import OTelInstrumentation
    
    otel = OTelInstrumentation("test-service", otlp_endpoint=None)
    
    otel.record_metric("test_counter", 42.0, "count", {"label": "value"})
    otel.record_metric("test_gauge", 100.5, "bytes", {"type": "memory"})
    
    # Verify metrics were recorded (in-memory fallback)
    assert len(otel.metrics) == 2
    assert otel.metrics[0]["name"] == "test_counter"
    assert otel.metrics[0]["value"] == 42.0
    assert otel.metrics[1]["name"] == "test_gauge"
    assert otel.metrics[1]["value"] == 100.5


def test_log_recording():
    """Test that logs can be recorded."""
    from observability.otel.instrumentation import OTelInstrumentation
    
    otel = OTelInstrumentation("test-service", otlp_endpoint=None)
    
    otel.log("INFO", "Test log message", {"key": "value"})
    otel.log("ERROR", "Test error", {"error_code": 500})
    
    # Verify logs were recorded
    assert len(otel.logs) == 2
    assert otel.logs[0]["level"] == "INFO"
    assert otel.logs[0]["message"] == "Test log message"
    assert otel.logs[1]["level"] == "ERROR"


def test_memory_leak_prevention():
    """Test that memory is bounded by max_queue_size."""
    from observability.otel.instrumentation import OTelInstrumentation
    
    max_queue = 50
    otel = OTelInstrumentation(
        "test-service",
        otlp_endpoint=None,
        max_queue_size=max_queue,
    )
    
    # Generate more traces than max_queue_size
    for i in range(max_queue * 2):
        with otel.trace_span(f"operation_{i}"):
            pass
    
    # Verify that traces are limited to prevent memory leak
    assert len(otel.traces) <= max_queue
    
    # Generate more metrics than max_queue_size
    for i in range(max_queue * 2):
        otel.record_metric(f"metric_{i}", float(i))
    
    # Verify that metrics are limited
    assert len(otel.metrics) <= max_queue
    
    # Generate more logs than max_queue_size
    for i in range(max_queue * 2):
        otel.log("INFO", f"Log message {i}")
    
    # Verify that logs are limited
    assert len(otel.logs) <= max_queue


def test_flush_functionality():
    """Test that flush can be called without errors."""
    from observability.otel.instrumentation import OTelInstrumentation
    
    otel = OTelInstrumentation("test-service", otlp_endpoint=None)
    
    otel.record_metric("test_metric", 1.0)
    with otel.trace_span("test"):
        pass
    
    # Flush should not raise errors
    otel.flush()


def test_shutdown_functionality():
    """Test that shutdown can be called without errors."""
    from observability.otel.instrumentation import OTelInstrumentation
    
    otel = OTelInstrumentation("test-service", otlp_endpoint=None)
    
    otel.record_metric("test_metric", 1.0)
    
    # Shutdown should not raise errors
    otel.shutdown()


def test_export_functions():
    """Test that telemetry can be exported."""
    from observability.otel.instrumentation import OTelInstrumentation
    
    otel = OTelInstrumentation("test-service", otlp_endpoint=None)
    
    with otel.trace_span("test"):
        pass
    otel.record_metric("test", 1.0)
    otel.log("INFO", "test")
    
    # Export should return JSON strings
    traces_json = otel.export_traces()
    metrics_json = otel.export_metrics()
    logs_json = otel.export_logs()
    
    assert isinstance(traces_json, str)
    assert isinstance(metrics_json, str)
    assert isinstance(logs_json, str)
    assert "test" in traces_json
    assert "test" in metrics_json
    assert "test" in logs_json


def test_otlp_exporter_configuration():
    """Test OTLP exporter configuration parameters."""
    from observability.otel.instrumentation import OTelInstrumentation
    
    otel = OTelInstrumentation(
        service_name="test-service",
        otlp_endpoint="http://localhost:4317",
        export_interval_ms=5000,
        max_queue_size=2048,
        max_export_batch_size=512,
    )
    
    assert otel.otlp_endpoint == "http://localhost:4317"
    assert otel.export_interval_ms == 5000
    assert otel.max_queue_size == 2048
    assert otel.max_export_batch_size == 512


def test_nested_spans():
    """Test that nested spans work correctly."""
    from observability.otel.instrumentation import OTelInstrumentation
    
    otel = OTelInstrumentation("test-service", otlp_endpoint=None)
    
    with otel.trace_span("parent"):
        time.sleep(0.01)
        with otel.trace_span("child"):
            time.sleep(0.01)
    
    # Both spans should be recorded
    assert len(otel.traces) >= 2


def test_concurrent_operations():
    """Test that concurrent operations are handled safely."""
    from observability.otel.instrumentation import OTelInstrumentation
    import threading
    
    otel = OTelInstrumentation("test-service", otlp_endpoint=None)
    
    def record_data():
        for i in range(10):
            with otel.trace_span(f"op_{i}"):
                otel.record_metric(f"metric_{i}", float(i))
                otel.log("INFO", f"Log {i}")
    
    # Run multiple threads
    threads = [threading.Thread(target=record_data) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    
    # Verify some data was recorded
    assert len(otel.traces) > 0
    assert len(otel.metrics) > 0
    assert len(otel.logs) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
