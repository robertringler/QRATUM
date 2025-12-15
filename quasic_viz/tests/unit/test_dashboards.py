"""Unit tests for dashboards and analytics."""

import time

import numpy as np


class TestWebGPUDashboard:
    """Tests for WebGPU dashboard functions."""

    def test_get_latest_frame_json(self):
        """Test getting latest frame as JSON."""
        from quasic_viz.dashboards.webgpu_dashboard import \
            get_latest_frame_json

        json_str = get_latest_frame_json()
        assert isinstance(json_str, str)
        assert "mesh" in json_str
        assert "fields" in json_str

    def test_update_frame_data(self):
        """Test updating frame data."""
        from quasic_viz.dashboards.webgpu_dashboard import (
            get_latest_frame_json, update_frame_data)

        update_frame_data([1, 2, 3], {"temp": [100, 200]})
        json_str = get_latest_frame_json()
        assert "1" in json_str or "[1" in json_str

    def test_create_dashboard_app(self):
        """Test dashboard app creation."""
        from quasic_viz.dashboards.webgpu_dashboard import create_dashboard_app

        app = create_dashboard_app()
        # May be None if FastAPI not installed
        if app is not None:
            assert hasattr(app, "routes")


class TestTimeSeriesAnalytics:
    """Tests for TimeSeriesAnalytics class."""

    def test_init(self):
        """Test analytics initialization."""
        from quasic_viz.dashboards.analytics import TimeSeriesAnalytics

        analytics = TimeSeriesAnalytics(max_history=100)
        assert analytics.frame_count == 0

    def test_record_frame(self):
        """Test recording frames."""
        from quasic_viz.dashboards.analytics import TimeSeriesAnalytics

        analytics = TimeSeriesAnalytics()
        analytics.record_frame([1, 2, 3], {"temp": np.array([100, 200])})
        assert analytics.frame_count == 1

    def test_get_frame_at(self):
        """Test getting frame at timestamp."""
        from quasic_viz.dashboards.analytics import TimeSeriesAnalytics

        analytics = TimeSeriesAnalytics()
        start_time = time.time()
        analytics.record_frame([1, 2, 3], {"temp": [100]})

        frame = analytics.get_frame_at(start_time)
        assert frame is not None
        assert "mesh" in frame

    def test_get_latest_frames(self):
        """Test getting latest frames."""
        from quasic_viz.dashboards.analytics import TimeSeriesAnalytics

        analytics = TimeSeriesAnalytics()
        for i in range(5):
            analytics.record_frame([i], {"val": [i * 10]})

        frames = analytics.get_latest_frames(3)
        assert len(frames) == 3

    def test_compute_field_statistics(self):
        """Test computing field statistics."""
        from quasic_viz.dashboards.analytics import TimeSeriesAnalytics

        analytics = TimeSeriesAnalytics()
        analytics.record_frame([], {"temp": np.array([10, 20, 30])})
        analytics.record_frame([], {"temp": np.array([40, 50, 60])})

        stats = analytics.compute_field_statistics("temp")
        assert "min" in stats
        assert "max" in stats
        assert "mean" in stats
        assert "std" in stats

    def test_clear(self):
        """Test clearing history."""
        from quasic_viz.dashboards.analytics import TimeSeriesAnalytics

        analytics = TimeSeriesAnalytics()
        analytics.record_frame([1], {})
        analytics.clear()
        assert analytics.frame_count == 0
