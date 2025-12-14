"""FastAPI server for GPU rendering service."""

from __future__ import annotations

from typing import Dict

try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import JSONResponse

    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False
    FastAPI = None
    WebSocket = None
    JSONResponse = None


class RenderServer:
    """GPU render server using FastAPI.

    Args:
        gpu_available: Controls whether GPU acceleration is enabled for the render server
    """

    def __init__(self, gpu_available: bool = True) -> None:
        """Initialize render server."""
        if not FASTAPI_AVAILABLE:
            raise ImportError("FastAPI not installed. Install with: pip install fastapi uvicorn")

        self.gpu_available = gpu_available
        self.jobs: Dict[str, Dict] = {}
        self.app = self._create_app()

    def _create_app(self) -> FastAPI:
        """Create FastAPI application.

        Returns:
            FastAPI app instance
        """
        app = FastAPI(
            title="QUBIC Render Service",
            description="Distributed GPU rendering for QuASIM visualization",
            version="0.1.0",
        )

        @app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {
                "status": "healthy",
                "gpu_available": self.gpu_available,
                "active_jobs": len(self.jobs),
            }

        @app.get("/gpu-status")
        async def gpu_status():
            """GPU status endpoint."""
            gpu_info = self._get_gpu_info()
            return {"gpu_available": self.gpu_available, "info": gpu_info}

        return app

    def _get_gpu_info(self) -> Dict:
        """Get GPU information.

        Returns:
            GPU info dictionary
        """
        if not self.gpu_available:
            return {"message": "No GPU available"}

        try:
            import torch

            if torch.cuda.is_available():
                return {
                    "device_count": torch.cuda.device_count(),
                    "current_device": torch.cuda.current_device(),
                    "device_name": torch.cuda.get_device_name(0),
                    "memory_allocated": torch.cuda.memory_allocated(),
                    "memory_reserved": torch.cuda.memory_reserved(),
                }
        except ImportError:
            pass

        return {"message": "GPU detection not available"}


def create_app() -> FastAPI:
    """Create FastAPI application.

    Returns:
        FastAPI app instance
    """
    server = RenderServer()
    return server.app


if __name__ == "__main__":
    try:
        import uvicorn

        app = create_app()
        uvicorn.run(app, host="0.0.0.0", port=8000)
    except ImportError:
        print("uvicorn not installed. Install with: pip install uvicorn")
