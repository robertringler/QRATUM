#!/bin/bash
# QRATUM Sandbox Launch Script
# Launches the full QRATUM production-ready platform with a single command

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

echo "================================================================="
echo -e "${CYAN}🚀 QRATUM Sandbox Environment Launch${NC}"
echo "================================================================="
echo ""

# Function to print status messages
status() {
    echo -e "${CYAN}[*]${NC} $1"
}

success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

error() {
    echo -e "${RED}[✗]${NC} $1"
}

warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check prerequisites
status "Checking prerequisites..."

# Check Python version
if ! command -v python3 &> /dev/null; then
    error "Python 3 is required but not installed"
    exit 1
fi

PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)"; then
    error "Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi
success "Python version: $PYTHON_VERSION"

# Navigate to root directory
cd "$ROOT_DIR"

# Create virtual environment
VENV_DIR="$ROOT_DIR/venv-sandbox"
if [ ! -d "$VENV_DIR" ]; then
    status "Creating Python virtual environment..."
    python3 -m venv "$VENV_DIR"
    success "Virtual environment created at $VENV_DIR"
else
    status "Using existing virtual environment at $VENV_DIR"
fi

# Activate virtual environment
status "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
success "Virtual environment activated"

# Upgrade pip
status "Upgrading pip..."
pip install --upgrade pip setuptools wheel -q
success "pip upgraded"

# Install dependencies
status "Installing dependencies from requirements.txt..."
pip install -r requirements.txt -q
success "Core dependencies installed"

if [ -f "requirements-prod.txt" ]; then
    status "Installing production dependencies from requirements-prod.txt..."
    # Try to install, but continue if some packages fail due to version constraints
    if ! pip install -r requirements-prod.txt -q 2>/tmp/pip_error.log; then
        warning "Some production dependencies could not be installed (may be due to Python version constraints)"
        if [ -s /tmp/pip_error.log ]; then
            echo "  See /tmp/pip_error.log for details"
        fi
    fi
    success "Production dependencies installation attempted"
fi

# Install Flask and Flask-CORS if not present (for platform server)
status "Installing platform dependencies..."
pip install flask flask-cors -q
success "Platform dependencies installed"

# Initialize QRADLE Merkle chain genesis block
status "Initializing QRADLE Merkle chain genesis block..."
python3 << 'PYTHON_SCRIPT'
import sys
sys.path.insert(0, '.')

from qradle.core.merkle import MerkleChain
from qradle.core.engine import DeterministicEngine
import time

# Create genesis block
chain = MerkleChain(genesis_data={
    "type": "genesis",
    "version": "1.0.0",
    "timestamp": time.time(),
    "network": "sandbox",
    "description": "QRATUM Sandbox Genesis Block"
})

# Initialize engine
engine = DeterministicEngine()

# Verify initialization
if len(chain.nodes) >= 1:
    print(f"✓ QRADLE genesis block created")
    print(f"  Genesis hash: {chain.get_root_hash()[:16]}...")
    print(f"  Deterministic engine initialized")
else:
    print("✗ Failed to initialize QRADLE")
    sys.exit(1)
PYTHON_SCRIPT

if [ $? -eq 0 ]; then
    success "QRADLE genesis block initialized"
else
    error "Failed to initialize QRADLE"
    exit 1
fi

# Create a simple QRADLE server wrapper
status "Creating QRADLE service wrapper..."
cat > /tmp/qradle_server.py << 'EOF'
#!/usr/bin/env python3
"""
QRADLE Service - Merkle Chain and Deterministic Execution Engine
Runs on port 8001
"""
import sys
sys.path.insert(0, '.')

from flask import Flask, jsonify
from flask_cors import CORS
import time
from qradle.core.merkle import MerkleChain
from qradle.core.engine import DeterministicEngine

app = Flask(__name__)
CORS(app)

# Initialize QRADLE components
chain = MerkleChain(genesis_data={
    "type": "genesis",
    "version": "1.0.0",
    "timestamp": time.time(),
    "network": "sandbox"
})
engine = DeterministicEngine()
start_time = time.time()

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "QRADLE",
        "version": "1.0.0",
        "uptime": time.time() - start_time
    })

@app.route('/api/chain/status')
def chain_status():
    """Get Merkle chain status"""
    return jsonify({
        "chain_length": len(chain.nodes),
        "root_hash": chain.get_root_hash(),
        "genesis_hash": chain.nodes[0].node_hash if chain.nodes else None
    })

@app.route('/api/engine/status')
def engine_status():
    """Get deterministic engine status"""
    return jsonify({
        "initialized": True,
        "seed": engine.seed if hasattr(engine, 'seed') else None,
        "operations_count": len(engine.execution_log) if hasattr(engine, 'execution_log') else 0
    })

@app.route('/')
def index():
    """Landing page"""
    return jsonify({
        "service": "QRADLE - Quantum-Resilient Auditable Deterministic Ledger Engine",
        "endpoints": {
            "health": "/health",
            "chain_status": "/api/chain/status",
            "engine_status": "/api/engine/status"
        }
    })

if __name__ == '__main__':
    print("=" * 70)
    print("🛡️  QRADLE Service Starting")
    print("=" * 70)
    print()
    print("Merkle Chain Genesis Block Initialized")
    print(f"Root Hash: {chain.get_root_hash()[:32]}...")
    print()
    print("Starting QRADLE on http://0.0.0.0:8001")
    print("=" * 70)
    app.run(host='0.0.0.0', port=8001, debug=False)
EOF

success "QRADLE service wrapper created"

# Start services in background
status "Starting QRADLE service on port 8001..."
python3 /tmp/qradle_server.py > /tmp/qradle.log 2>&1 &
QRADLE_PID=$!
sleep 2

# Check if QRADLE started successfully
if ps -p $QRADLE_PID > /dev/null; then
    success "QRADLE service started (PID: $QRADLE_PID)"
else
    error "Failed to start QRADLE service"
    cat /tmp/qradle.log
    exit 1
fi

status "Starting QRATUM Platform on port 8002..."
PORT=8002 python3 qratum_platform.py > /tmp/platform.log 2>&1 &
PLATFORM_PID=$!
sleep 2

# Check if Platform started successfully
if ps -p $PLATFORM_PID > /dev/null; then
    success "QRATUM Platform started (PID: $PLATFORM_PID)"
else
    error "Failed to start QRATUM Platform"
    cat /tmp/platform.log
    exit 1
fi

# Health check verification
status "Performing health checks..."
sleep 3

# Check QRADLE health
if curl -s http://localhost:8001/health > /dev/null 2>&1; then
    success "QRADLE health check passed"
else
    warning "QRADLE health check failed (service may still be starting)"
fi

# Check Platform health
if curl -s http://localhost:8002/ > /dev/null 2>&1; then
    success "QRATUM Platform health check passed"
else
    warning "QRATUM Platform health check failed (service may still be starting)"
fi

echo ""
echo "================================================================="
echo -e "${GREEN}✓ QRATUM Sandbox Environment Running${NC}"
echo "================================================================="
echo ""
echo "Services:"
echo "  🛡️  QRADLE:          http://localhost:8001"
echo "  🚀 QRATUM Platform: http://localhost:8002"
echo ""
echo "Endpoints:"
echo "  - QRADLE Health:        http://localhost:8001/health"
echo "  - QRADLE Chain Status:  http://localhost:8001/api/chain/status"
echo "  - QRADLE Engine Status: http://localhost:8001/api/engine/status"
echo "  - Platform Status:      http://localhost:8002/api/status"
echo ""
echo "Logs:"
echo "  - QRADLE:  /tmp/qradle.log"
echo "  - Platform: /tmp/platform.log"
echo ""
echo "To stop services:"
echo "  kill $QRADLE_PID $PLATFORM_PID"
echo ""
echo "To run tests:"
echo "  python3 sandbox/test_sandbox.py"
echo ""
echo "================================================================="

# Cleanup function
cleanup() {
    echo ""
    echo "Stopping services..."
    kill $QRADLE_PID $PLATFORM_PID 2>/dev/null || true
    echo "Services stopped"
    exit 0
}

trap cleanup EXIT INT TERM

# Keep script running
echo "Press Ctrl+C to stop all services"
wait
