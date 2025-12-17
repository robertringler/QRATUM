"""Unit tests for services."""

import time


class TestFrameCache:
    """Tests for FrameCache class."""

    def test_init(self):
        """Test cache initialization."""

        from quasic_viz.services.qubic_render.frame_cache import FrameCache

        cache = FrameCache(max_frames=100)
        assert len(cache) == 0

    def test_cache_frame(self):
        """Test caching frames."""

        from quasic_viz.services.qubic_render.frame_cache import FrameCache

        cache = FrameCache()
        cache.cache_frame([1, 2, 3], {"temp": [100]})
        assert len(cache) == 1

    def test_get_latest(self):
        """Test getting latest frame."""

        from quasic_viz.services.qubic_render.frame_cache import FrameCache

        cache = FrameCache()
        cache.cache_frame([1, 2, 3], {"temp": [100]})
        frame = cache.get_latest()

        assert frame is not None
        assert "mesh" in frame
        assert "fields" in frame

    def test_get_frame_at(self):
        """Test getting frame at timestamp."""

        from quasic_viz.services.qubic_render.frame_cache import FrameCache

        cache = FrameCache()
        start_time = time.time()
        cache.cache_frame([1, 2, 3], {"temp": [100]})

        frame = cache.get_frame_at(start_time)
        assert frame is not None

    def test_get_range(self):
        """Test getting frames in time range."""

        from quasic_viz.services.qubic_render.frame_cache import FrameCache

        cache = FrameCache()
        start_time = time.time()
        for i in range(5):
            cache.cache_frame([i], {"val": [i * 10]})
            time.sleep(0.01)
        end_time = time.time()

        frames = cache.get_range(start_time, end_time)
        assert len(frames) == 5

    def test_clear(self):
        """Test clearing cache."""

        from quasic_viz.services.qubic_render.frame_cache import FrameCache

        cache = FrameCache()
        cache.cache_frame([1], {})
        cache.clear()
        assert len(cache) == 0

    def test_timestamps(self):
        """Test timestamp properties."""

        from quasic_viz.services.qubic_render.frame_cache import FrameCache

        cache = FrameCache()
        assert cache.oldest_timestamp is None
        assert cache.newest_timestamp is None

        cache.cache_frame([1], {})
        assert cache.oldest_timestamp is not None
        assert cache.newest_timestamp is not None


class TestGlobalFrameCache:
    """Tests for global frame cache functions."""

    def test_cache_frame_global(self):
        """Test global cache_frame function."""

        from quasic_viz.services.qubic_render.frame_cache import (
            FRAME_HISTORY,
            cache_frame,
        )

        initial_len = len(FRAME_HISTORY)
        cache_frame([1, 2, 3], {"temp": [100]})
        assert len(FRAME_HISTORY) > initial_len

    def test_get_frame_at_global(self):
        """Test global get_frame_at function."""

        from quasic_viz.services.qubic_render.frame_cache import (
            cache_frame,
            get_frame_at,
        )

        start_time = time.time()
        cache_frame([1, 2, 3], {"temp": [100]})

        frame = get_frame_at(start_time)
        assert frame is not None


class TestWebSocketServer:
    """Tests for WebSocketServer class."""

    def test_init(self):
        """Test server initialization."""

        from quasic_viz.services.qubic_render.ws_server import WebSocketServer

        server = WebSocketServer()
        assert server.client_count == 0

    def test_create_ws_app(self):
        """Test creating WebSocket app."""

        from quasic_viz.services.qubic_render.ws_server import create_ws_app

        app = create_ws_app()
        # May be None if FastAPI not installed
        if app is not None:
            assert hasattr(app, "routes")
