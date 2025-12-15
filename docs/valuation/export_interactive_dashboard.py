#!/usr/bin/env python3
"""
Export QuASIM valuation dashboard as interactive HTML.

This script generates a standalone HTML page with all dashboard visualizations
for interactive exploration in web browsers.

Usage:
    python docs/valuation/export_interactive_dashboard.py
"""

import sys
from pathlib import Path

# Add parent directory to path to import dashboard module
sys.path.insert(0, str(Path(__file__).parent))

import plotly.io as pio
from quasim_valuation_dashboard import (create_dcf_waterfall_chart,
                                        create_monte_carlo_waterfall,
                                        create_sensitivity_heatmap)

# Output path
OUTPUT_PATH = Path(__file__).parent.parent / "valuation_dashboard.html"


def create_html_dashboard():
    """Create standalone HTML dashboard with all figures."""
    # Create figures
    fig_dcf = create_dcf_waterfall_chart()
    fig_heatmap = create_sensitivity_heatmap()
    fig_waterfall = create_monte_carlo_waterfall()

    # Generate HTML for each figure
    html_dcf = pio.to_html(fig_dcf, full_html=False, include_plotlyjs=False)
    html_heatmap = pio.to_html(fig_heatmap, full_html=False, include_plotlyjs=False)
    html_waterfall = pio.to_html(fig_waterfall, full_html=False, include_plotlyjs=False)

    # Create full HTML page
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QuASIM Valuation Dashboard — Q1 2026</title>
    <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f9fafb;
            color: #1f2937;
        }}
        .header {{
            background: linear-gradient(135deg, #2E86AB 0%, #1e5a7a 100%);
            color: white;
            padding: 2rem;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .header h1 {{
            margin: 0;
            font-size: 2.5rem;
            font-weight: 700;
        }}
        .header p {{
            margin: 0.5rem 0 0 0;
            font-size: 1.125rem;
            opacity: 0.95;
        }}
        .container {{
            max-width: 1400px;
            margin: 0 auto;
            padding: 2rem;
        }}
        .dashboard-section {{
            background: white;
            border-radius: 8px;
            padding: 2rem;
            margin-bottom: 2rem;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .dashboard-section h2 {{
            margin-top: 0;
            color: #1f2937;
            font-size: 1.5rem;
            border-bottom: 2px solid #2E86AB;
            padding-bottom: 0.5rem;
        }}
        .figure-container {{
            margin-top: 1rem;
        }}
        .footer {{
            text-align: center;
            padding: 2rem;
            color: #6b7280;
            font-size: 0.875rem;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1rem;
            margin-bottom: 2rem;
        }}
        .metric-card {{
            background: linear-gradient(135deg, #f3f4f6 0%, #e5e7eb 100%);
            padding: 1.5rem;
            border-radius: 8px;
            border-left: 4px solid #2E86AB;
        }}
        .metric-card h3 {{
            margin: 0 0 0.5rem 0;
            font-size: 0.875rem;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .metric-card p {{
            margin: 0;
            font-size: 1.75rem;
            font-weight: 700;
            color: #1f2937;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>QuASIM Valuation Dashboard</h1>
        <p>Q1 2026 Financial Analysis — Pre-Revenue Valuation Update</p>
    </div>

    <div class="container">
        <div class="dashboard-section">
            <h2>Key Metrics</h2>
            <div class="metrics">
                <div class="metric-card">
                    <h3>P50 Enterprise Value</h3>
                    <p>$13.9M</p>
                </div>
                <div class="metric-card">
                    <h3>Base Case DCF</h3>
                    <p>$14.7M</p>
                </div>
                <div class="metric-card">
                    <h3>P90 Upside</h3>
                    <p>$28.1M</p>
                </div>
                <div class="metric-card">
                    <h3>Real Options Uplift</h3>
                    <p>$20.2M</p>
                </div>
            </div>
        </div>

        <div class="dashboard-section">
            <h2>1. DCF Valuation Analysis</h2>
            <p>Discounted cash flow analysis showing present value components for the base case scenario.</p>
            <div class="figure-container">
                {html_dcf}
            </div>
        </div>

        <div class="dashboard-section">
            <h2>2. Sensitivity Analysis</h2>
            <p>Enterprise value sensitivity to weighted average cost of capital (WACC) and terminal growth rate assumptions.</p>
            <div class="figure-container">
                {html_heatmap}
            </div>
        </div>

        <div class="dashboard-section">
            <h2>3. Monte Carlo Distribution</h2>
            <p>Probabilistic valuation outcomes based on 20,000 Monte Carlo simulation trials.</p>
            <div class="figure-container">
                {html_waterfall}
            </div>
        </div>
    </div>

    <div class="footer">
        <p><strong>QuASIM</strong> — Quantum-Inspired Autonomous Simulation Platform</p>
        <p>Generated: {Path(__file__).stat().st_mtime} | Methodology: DCF + Monte Carlo + Real Options</p>
    </div>
</body>
</html>
"""

    return html_template


def main():
    """Generate and save interactive HTML dashboard."""
    print("QuASIM Valuation Dashboard — Interactive HTML Export")
    print("=" * 50)

    print("\n[1/2] Generating interactive dashboard...")
    html_content = create_html_dashboard()
    print("  ✓ Dashboard HTML generated")

    print("\n[2/2] Saving to file...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"  ✓ Saved to {OUTPUT_PATH}")

    print("\n" + "=" * 50)
    print("Interactive dashboard export complete!")
    print(f"\nOpen in browser: file://{OUTPUT_PATH.absolute()}")


if __name__ == "__main__":
    main()
