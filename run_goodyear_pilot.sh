#!/bin/bash
# Goodyear Quantum Tire Pilot - Quick Launch Script
# 
# This script runs the full Goodyear Quantum Tire Pilot simulation campaign,
# generating 10,000+ unique tire simulation scenarios with quantum-enhanced
# optimization.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                    GOODYEAR QUANTUM TIRE PILOT                                ║"
echo "║                 QuASIM Quantum-Enhanced Tire Simulation                       ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "This will generate 10,000+ tire simulation scenarios using:"
echo "  • 1,000+ Goodyear materials (8 material families)"
echo "  • 8 tire types (passenger, truck, off-road, racing, EV, winter, all-season, performance)"
echo "  • 12 surface types and 8 weather conditions"
echo "  • Temperature range: -40°C to +80°C"
echo "  • Quantum-enhanced optimization (QAOA, VQE, hybrid algorithms)"
echo ""
echo "Expected runtime: ~2-3 seconds"
echo "Output directory: goodyear_quantum_pilot_full/"
echo ""
echo "Press Ctrl+C to cancel, or wait 3 seconds to continue..."
sleep 3
echo ""

# Check for Python 3
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found. Please install Python 3.10 or later."
    exit 1
fi

# Check for required modules
echo "Checking dependencies..."
python3 -c "import numpy, yaml, click" 2>/dev/null || {
    echo "Installing required dependencies..."
    pip install -q numpy pyyaml click
    echo "✓ Dependencies installed"
}
echo "✓ Dependencies OK"
echo ""

# Run the pilot
echo "Launching Goodyear Quantum Tire Pilot..."
echo ""
python3 run_goodyear_quantum_pilot.py

echo ""
echo "╔═══════════════════════════════════════════════════════════════════════════════╗"
echo "║                         EXECUTION COMPLETE                                    ║"
echo "╚═══════════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Output files:"
echo "  📊 Results: goodyear_quantum_pilot_full/goodyear_simulation_results.json"
echo "  🧪 Materials: goodyear_quantum_pilot_full/goodyear_materials_database.json"
echo "  📖 Documentation: goodyear_quantum_pilot_full/README.md"
echo ""
echo "Next steps:"
echo "  1. Review comprehensive usage guide: GOODYEAR_PILOT_USAGE.md"
echo "  2. Explore simulation results"
echo "  3. Integrate with CAD/FEA/AI workflows"
echo ""
