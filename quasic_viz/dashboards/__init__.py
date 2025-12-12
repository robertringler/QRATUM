"""WebGPU dashboards and analytics for QuASIC visualization."""

from .analytics import TimeSeriesAnalytics
from .webgpu_dashboard import create_dashboard_app, get_latest_frame_json

__all__ = ["create_dashboard_app", "get_latest_frame_json", "TimeSeriesAnalytics"]
