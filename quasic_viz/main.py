"""QuASIC Visualization main entry point.

Example runner demonstrating the visualization platform features.
"""

from __future__ import annotations


def main() -> None:
    """Run the QuASIC visualization server."""
    try:
        import uvicorn

        from quasic_viz.dashboards.webgpu_dashboard import create_dashboard_app

        app = create_dashboard_app()
        if app is None:
            print("FastAPI not available. Install with: pip install fastapi uvicorn")
            return

        print("Starting QuASIC Visualization Server...")
        print("Dashboard available at: http://0.0.0.0:8000")
        print("WebSocket endpoint: ws://0.0.0.0:8000/ws/dashboard")

        uvicorn.run(app, host="0.0.0.0", port=8000)

    except ImportError as e:
        print(f"Missing dependencies: {e}")
        print("Install with: pip install fastapi uvicorn")


if __name__ == "__main__":
    main()
