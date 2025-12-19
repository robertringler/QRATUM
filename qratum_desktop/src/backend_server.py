#!/usr/bin/env python3
"""
QRATUM Desktop Edition - Backend Server

Lightweight FastAPI server optimized for desktop mode:
- Local SQLite database (no PostgreSQL)
- In-memory session management (no Redis)
- Simple local authentication (no Vault/OIDC)
- Thread-based workers (no Kubernetes/Ray)
- Automatic GPU detection and fallback

Usage:
    python backend_server.py --port 8000 --desktop-mode
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
repo_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(repo_root))


def check_gpu_availability():
    """Check if GPU is available for computation."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
            return True
        else:
            logger.info("No GPU detected, using CPU")
            return False
    except ImportError:
        logger.info("PyTorch not available, using CPU")
        return False


def setup_desktop_mode():
    """Configure environment for desktop mode."""
    # Set desktop mode environment variable
    os.environ['QRATUM_DESKTOP_MODE'] = '1'
    
    # Create desktop data directory
    if sys.platform == 'win32':
        data_dir = Path(os.environ['APPDATA']) / 'QRATUM'
    elif sys.platform == 'darwin':
        data_dir = Path.home() / 'Library' / 'Application Support' / 'QRATUM'
    else:  # Linux
        data_dir = Path.home() / '.local' / 'share' / 'qratum'
    
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ['QRATUM_DATA_DIR'] = str(data_dir)
    
    # Setup cache directory
    if sys.platform == 'win32':
        cache_dir = data_dir / 'cache'
    elif sys.platform == 'darwin':
        cache_dir = Path.home() / 'Library' / 'Caches' / 'QRATUM'
    else:  # Linux
        cache_dir = Path.home() / '.cache' / 'qratum'
    
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ['QRATUM_CACHE_DIR'] = str(cache_dir)
    
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Cache directory: {cache_dir}")
    
    return data_dir


def create_desktop_app(port: int = 8000):
    """Create FastAPI application optimized for desktop mode."""
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import JSONResponse
        from datetime import datetime
    except ImportError:
        logger.error("FastAPI not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)
    
    app = FastAPI(
        title="QRATUM Desktop Backend",
        description="Local backend server for QRATUM Desktop Edition",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[f"http://localhost:{port}", "file://"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
    )
    
    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return JSONResponse({
            "status": "healthy",
            "mode": "desktop",
            "version": "2.0.0",
            "timestamp": datetime.utcnow().isoformat() + "Z"
        })
    
    @app.get("/api/system/info")
    async def system_info():
        """Get system information."""
        import platform
        
        return JSONResponse({
            "platform": platform.system(),
            "platform_release": platform.release(),
            "platform_version": platform.version(),
            "architecture": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "gpu_available": check_gpu_availability()
        })
    
    @app.get("/api/desktop/status")
    async def desktop_status():
        """Get desktop mode status."""
        data_dir = os.environ.get('QRATUM_DATA_DIR', 'N/A')
        
        return JSONResponse({
            "desktop_mode": True,
            "data_dir": data_dir,
            "backend_port": port,
            "gpu_available": check_gpu_availability()
        })
    
    logger.info("Desktop backend application created")
    return app


def main():
    """Main entry point for desktop backend server."""
    parser = argparse.ArgumentParser(description='QRATUM Desktop Backend Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run on')
    parser.add_argument('--desktop-mode', action='store_true', help='Enable desktop mode')
    parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("QRATUM Desktop Edition - Backend Server")
    logger.info("=" * 60)
    logger.info(f"Version: 2.0.0")
    logger.info(f"Port: {args.port}")
    logger.info(f"Desktop Mode: {args.desktop_mode}")
    
    # Setup desktop environment
    if args.desktop_mode:
        data_dir = setup_desktop_mode()
        check_gpu_availability()
    
    # Create FastAPI app
    app = create_desktop_app(port=args.port)
    
    # Import existing API routes if available
    try:
        from api.v1.main import create_app
        logger.info("Importing existing QRATUM API routes")
        existing_app = create_app()
        
        # Mount existing app routes
        if existing_app:
            # Copy routes from existing app
            for route in existing_app.routes:
                if hasattr(route, 'path') and not route.path.startswith('/health'):
                    app.routes.append(route)
            logger.info("Existing API routes imported successfully")
    except ImportError as e:
        logger.warning(f"Could not import existing API: {e}")
        logger.info("Running with basic endpoints only")
    
    # Start server
    try:
        import uvicorn
        logger.info(f"Starting server on http://127.0.0.1:{args.port}")
        logger.info("Backend ready")  # Signal to Electron that we're ready
        
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=args.port,
            log_level="info",
            reload=args.reload
        )
    except ImportError:
        logger.error("uvicorn not installed. Install with: pip install uvicorn")
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
