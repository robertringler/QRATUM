#!/bin/bash
# Run full stack locally without Docker
# This script starts both backend and frontend services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
BACKEND_DIR="$ROOT_DIR/autonomous_systems_platform/services/backend"
FRONTEND_DIR="$ROOT_DIR/autonomous_systems_platform/services/frontend"

echo "=== QuASIM Full Stack Local Runner ==="
echo ""

# Check prerequisites
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 is required but not installed."
    exit 1
fi

echo "Starting backend service..."
cd "$BACKEND_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
echo "Installing backend dependencies..."
pip install -q -r requirements.txt

# Set environment variables
export JAX_PLATFORM_NAME=cpu
export PORT=8000

# Start backend in background
echo "Starting backend on http://localhost:8000"
python3 app.py &
BACKEND_PID=$!

# Give backend time to start
sleep 3

# Start frontend
echo "Starting frontend on http://localhost:8080"
cd "$FRONTEND_DIR"
python3 -m http.server 8080 &
FRONTEND_PID=$!

echo ""
echo "=== Services Started ==="
echo "Backend:  http://localhost:8000"
echo "Frontend: http://localhost:8080"
echo ""
echo "Press Ctrl+C to stop all services"
echo ""

# Handle cleanup on exit
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo "Done."
    exit 0
}

trap cleanup EXIT INT TERM

# Wait for user interrupt
wait
