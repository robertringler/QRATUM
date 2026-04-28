#!/usr/bin/env python3
"""

Export QuASIM valuation dashboard as static PNG images.

This script imports the dashboard figures and exports them as high-resolution PNG files
for inclusion in reports and documentation.

Usage:
    python docs/valuation/export_static_images.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import dashboard module
sys.path.insert(0, str(Path(__file__).parent))

import plotly.io as pio
from quasim_valuation_dashboard import (
    create_dcf_waterfall_chart,
    create_monte_carlo_waterfall,
    create_sensitivity_heatmap,
)

# Output directory
ASSETS_DIR = Path(__file__).parent.parent / "assets"
ASSETS_DIR.mkdir(exist_ok=True)


def main():
    """Export dashboard figures as static PNG images."""

    print("QuASIM Valuation Dashboard — Static Image Export")
    print("=" * 50)

    # Create figures
    print("\n[1/4] Generating dashboard figures...")
    fig_dcf = create_dcf_waterfall_chart()
    fig_heatmap = create_sensitivity_heatmap()
    fig_waterfall = create_monte_carlo_waterfall()
    print("  ✓ All figures generated")

    # Export DCF chart
    print("\n[2/4] Exporting DCF valuation chart...")
    dcf_path = ASSETS_DIR / "valuation_dcf.png"
    pio.write_image(fig_dcf, dcf_path, scale=2, width=1200, height=600)
    print(f"  ✓ Saved to {dcf_path}")

    # Export sensitivity heatmap
    print("\n[3/4] Exporting sensitivity heatmap...")
    heatmap_path = ASSETS_DIR / "valuation_heatmap.png"
    pio.write_image(fig_heatmap, heatmap_path, scale=2, width=1200, height=600)
    print(f"  ✓ Saved to {heatmap_path}")

    # Export Monte Carlo waterfall
    print("\n[4/4] Exporting Monte Carlo waterfall...")
    waterfall_path = ASSETS_DIR / "valuation_waterfall.png"
    pio.write_image(fig_waterfall, waterfall_path, scale=2, width=1200, height=600)
    print(f"  ✓ Saved to {waterfall_path}")

    print("\n" + "=" * 50)
    print("Static image export complete!")
    print(f"\nImages saved to: {ASSETS_DIR}")
    print("  - valuation_dcf.png")
    print("  - valuation_heatmap.png")
    print("  - valuation_waterfall.png")


if __name__ == "__main__":
    main()
