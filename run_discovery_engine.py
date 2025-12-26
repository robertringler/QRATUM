#!/usr/bin/env python3
"""Convenience script to run QRATUM Discovery Engine.

This script provides access to QRATUM's vulnerability discovery capabilities:

1. Vulnerability Discovery (default): Non-exploitative scientific analysis
   of latent structural vulnerabilities across the QRATUM stack.
   
2. Q-FORGE Discovery: Cross-domain hypothesis generation and synthesis.

Usage:
    python3 run_discovery_engine.py [mode]
    
    mode: 'vulnerability' (default) or 'qforge'
    
Examples:
    python3 run_discovery_engine.py
    python3 run_discovery_engine.py vulnerability
    python3 run_discovery_engine.py qforge
"""

import sys
from pathlib import Path

# Add QRATUM root directory to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Run the appropriate discovery engine based on command-line argument."""
    mode = sys.argv[1] if len(sys.argv) > 1 else "vulnerability"
    
    if mode == "vulnerability":
        from qratum_asi.examples.vulnerability_discovery_demo import main as vuln_main
        vuln_main()
    elif mode == "qforge":
        from qratum_asi.examples.discovery_demo import main as qforge_main
        qforge_main()
    else:
        print(f"Unknown mode: {mode}")
        print("Valid modes: 'vulnerability', 'qforge'")
        sys.exit(1)


if __name__ == "__main__":
    main()
