#!/usr/bin/env python3
"""
CIIR Monograph — Master Figure Generation Script
==================================================
Generates all 8 scientific illustrations for the CIIR monograph.
Run from the repository root:

    python manuscripts/ciir_monograph/figures/generate_all.py

Requirements: matplotlib, numpy, scipy
Output: PNG (300 DPI) and PDF for each figure in this directory.
"""

import importlib.util
import os
import sys
import time

FIGURE_DIR = os.path.dirname(os.path.abspath(__file__))

FIGURES = [
    ("fig01_ciir_architecture.py",   "CIIR Architecture Overview"),
    ("fig02_primitive_triple.py",    "Primitive Triple (S, C, D)"),
    ("fig03_recursive_loop.py",      "CRS–CIIR–Observer Recursive Loop"),
    ("fig04_emergent_geometry.py",   "Emergent Metric Geometry Pipeline"),
    ("fig05_probability_emergence.py", "Probability from Constraint Minimization"),
    ("fig06_emergent_gravity.py",    "Emergent Spacetime & Gravity"),
    ("fig07_crs_architecture.py",    "CRS 8-Layer Architecture"),
    ("fig08_functorial_structure.py", "Functorial Structure CogSys → Hilb"),
    ("fig09_informational_filtering.py", "Topological Manifold of Informational Filtering"),
    ("fig10_complexity_vs_icons.py", "Objective Complexity vs Interface Icons"),
    ("fig11_evolutionary_feedback.py", "Neuro-Evolutionary Feedback Loop"),
]


def run_figure(filename, description):
    """Import and run a figure generation script."""
    filepath = os.path.join(FIGURE_DIR, filename)
    module_name = filename.replace(".py", "")
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
        mod.main()
        return True
    except Exception as e:
        print(f"✗ {description}: {e}")
        return False


def main():
    # Change to repo root for correct relative paths
    repo_root = os.path.abspath(os.path.join(FIGURE_DIR, "..", "..", ".."))
    os.chdir(repo_root)

    print("=" * 60)
    print("  CIIR Monograph — Scientific Illustration Generation")
    print("=" * 60)
    print(f"  Output directory: {FIGURE_DIR}")
    print()

    t0 = time.time()
    success = 0
    total = len(FIGURES)

    for filename, description in FIGURES:
        print(f"  Generating: {description}...")
        if run_figure(filename, description):
            success += 1

    elapsed = time.time() - t0
    print()
    print("=" * 60)
    print(f"  {success}/{total} figures generated in {elapsed:.1f}s")
    print("=" * 60)

    # List generated files
    print("\n  Generated files:")
    for f in sorted(os.listdir(FIGURE_DIR)):
        if f.endswith((".png", ".pdf")):
            size = os.path.getsize(os.path.join(FIGURE_DIR, f))
            print(f"    {f:50s}  {size/1024:7.1f} KB")

    return 0 if success == total else 1


if __name__ == "__main__":
    sys.exit(main())
