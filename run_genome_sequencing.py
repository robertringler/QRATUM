#!/usr/bin/env python3
"""Convenience script to run XENON v5 full genome sequencing pipeline.

This script provides a simple interface to execute the full genome sequencing
pipeline with XENON Quantum Bioinformatics v5 engines.
"""

import sys
from pathlib import Path

# Add xenon to path
sys.path.insert(0, str(Path(__file__).parent))

from xenon.bioinformatics.full_genome_sequencing import main

if __name__ == "__main__":
    main()
