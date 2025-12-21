#!/usr/bin/env python3
"""Convenience script to run QRATUM-ASI Q-FORGE Discovery Engine.

This script provides a simple interface to execute the Q-FORGE Discovery
demonstration showcasing cross-domain hypothesis generation, discovery
validation, and novel synthesis capabilities.

The Discovery Engine demonstrates:
- Cross-domain knowledge integration (QUASIM, XENON, QUBIC verticals)
- Hypothesis generation with confidence and novelty scoring
- Discovery creation and validation
- Multi-discovery synthesis for novel insights
- Merkle chain integrity verification

Usage:
    python3 run_discovery_engine.py
"""

import sys
from pathlib import Path

# Add QRATUM root directory to path
sys.path.insert(0, str(Path(__file__).parent))

from qratum_asi.examples.discovery_demo import main

if __name__ == "__main__":
    main()
