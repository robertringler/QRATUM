#!/usr/bin/env python3
"""

QuASIM Valuation Dashboard

Generates Plotly visualizations for the Q1 2026 valuation report.
Displays DCF, sensitivity, and waterfall charts reflecting
the $6.9B–$8.1B pre-revenue valuation update.

Usage:
    python docs/valuation/quasim_valuation_dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go

# Data paths
DATA_DIR = Path(__file__).parent / "data"
PROJECTIONS_BASE = DATA_DIR / "projections_base.csv"
PROJECTIONS_HIGH = DATA_DIR / "projections_high.csv"
PROJECTIONS_LOW = DATA_DIR / "projections_low.csv"
MC_HISTOGRAM = DATA_DIR / "mc_histogram.csv"

# Valuation parameters from market_valuation.md
DCF_VALUES = {
    "Base Case": 14_695_953,
    "High Case": 115_954_862,
    "Low Case": -168_831,
}

MC_PERCENTILES = {
    "P10 (Conservative)": 4_930_912,
    "P25": 8_801_606,
    "P50 (Median)": 13_936_912,
    "P75": 20_493_936,
    "P90 (Optimistic)": 28_111_678,
}

REAL_OPTIONS_UPLIFT = 20_214_291


def load_data():
    """Load valuation data from CSV files."""

    df_base = pd.read_csv(PROJECTIONS_BASE)
    df_high = pd.read_csv(PROJECTIONS_HIGH)
    df_low = pd.read_csv(PROJECTIONS_LOW)
    df_mc = pd.read_csv(MC_HISTOGRAM)
    return df_base, df_high, df_low, df_mc


def create_dcf_waterfall_chart():
    """Create DCF valuation waterfall chart showing value components."""

    # DCF components for base case from market_valuation.md
    pv_fcf_1_5 = 4_142_723
    pv_terminal = 10_553_230
    enterprise_value = DCF_VALUES["Base Case"]

    # Create waterfall data
    categories = [
        "PV of FCF<br>(Years 1-5)",
        "PV of Terminal<br>Value",
        "Enterprise<br>Value",
    ]
    values = [pv_fcf_1_5, pv_terminal, enterprise_value]

    # Create figure
    fig = go.Figure(
        go.Waterfall(
            name="DCF Components",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=categories,
            textposition="outside",
            text=[f"${v / 1e6:.2f}M" for v in values],
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2E86AB"}},
            decreasing={"marker": {"color": "#A23B72"}},
            totals={"marker": {"color": "#F18F01"}},
        )
    )

    fig.update_layout(
        title={
            "text": "QuASIM DCF Valuation — Base Case",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "#1f2937"},
        },
        showlegend=False,
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 12, "color": "#374151"},
        xaxis={
            "title": "Valuation Components",
            "gridcolor": "#e5e7eb",
            "linecolor": "#9ca3af",
        },
        yaxis={
            "title": "Value (USD)",
            "gridcolor": "#e5e7eb",
            "linecolor": "#9ca3af",
            "tickformat": "$,.0f",
        },
        margin={"t": 80, "b": 80, "l": 100, "r": 40},
    )

    return fig


def create_sensitivity_heatmap():
    """Create sensitivity analysis heatmap for WACC and growth rate."""

    # Generate sensitivity matrix for different WACC and terminal growth rates
    wacc_range = [0.15, 0.16, 0.17, 0.18, 0.19, 0.20, 0.21]
    growth_range = [0.01, 0.02, 0.03, 0.04, 0.05]

    # Base case values
    base_pv_fcf = 4_142_723
    base_fcf_terminal = 3_516_005

    # Calculate sensitivity matrix
    sensitivity_matrix = []
    for growth in growth_range:
        row = []
        for wacc in wacc_range:
            if wacc <= growth:
                # Invalid combination
                row.append(None)
            else:
                terminal_value = base_fcf_terminal * (1 + growth) / (wacc - growth)
                pv_terminal = terminal_value / ((1 + wacc) ** 5)
                enterprise_value = base_pv_fcf + pv_terminal
                row.append(enterprise_value / 1e6)  # Convert to millions
        sensitivity_matrix.append(row)

    # Create heatmap
    fig = go.Figure(
        data=go.Heatmap(
            z=sensitivity_matrix,
            x=[f"{w * 100:.0f}%" for w in wacc_range],
            y=[f"{g * 100:.0f}%" for g in growth_range],
            colorscale="RdYlGn",
            text=[[f"${v:.1f}M" if v else "N/A" for v in row] for row in sensitivity_matrix],
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar={"title": {"text": "EV ($M)", "side": "right"}},
        )
    )

    fig.update_layout(
        title={
            "text": "Valuation Sensitivity Analysis — WACC vs. Terminal Growth",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "#1f2937"},
        },
        xaxis={"title": "WACC (Discount Rate)", "side": "bottom"},
        yaxis={"title": "Terminal Growth Rate"},
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 12, "color": "#374151"},
        margin={"t": 80, "b": 80, "l": 100, "r": 40},
    )

    return fig


def create_monte_carlo_waterfall():
    """Create waterfall chart showing Monte Carlo percentile values."""

    # Create waterfall showing progression from P10 to P90
    categories = list(MC_PERCENTILES.keys())
    values = list(MC_PERCENTILES.values())

    # Calculate incremental changes
    increments = [values[0]] + [values[i] - values[i - 1] for i in range(1, len(values))]

    fig = go.Figure(
        go.Waterfall(
            name="Monte Carlo Percentiles",
            orientation="v",
            measure=["absolute"] + ["relative"] * (len(categories) - 1),
            x=categories,
            textposition="outside",
            text=[f"${v / 1e6:.2f}M" for v in values],
            y=increments,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            increasing={"marker": {"color": "#2E86AB"}},
            decreasing={"marker": {"color": "#A23B72"}},
            totals={"marker": {"color": "#F18F01"}},
        )
    )

    fig.update_layout(
        title={
            "text": "QuASIM Monte Carlo Valuation Distribution (20,000 Trials)",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 20, "color": "#1f2937"},
        },
        showlegend=False,
        height=500,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font={"family": "Arial, sans-serif", "size": 12, "color": "#374151"},
        xaxis={
            "title": "Percentile",
            "gridcolor": "#e5e7eb",
            "linecolor": "#9ca3af",
        },
        yaxis={
            "title": "Enterprise Value (USD)",
            "gridcolor": "#e5e7eb",
            "linecolor": "#9ca3af",
            "tickformat": "$,.0f",
        },
        margin={"t": 80, "b": 80, "l": 100, "r": 40},
    )

    return fig


def main():
    """Generate and display valuation dashboard figures."""

    print("QuASIM Valuation Dashboard Generator")
    print("=" * 50)

    # Load data
    print("\n[1/4] Loading valuation data...")
    df_base, df_high, df_low, df_mc = load_data()
    print(f"  ✓ Loaded {len(df_base)} base projections")
    print(f"  ✓ Loaded {len(df_high)} high projections")
    print(f"  ✓ Loaded {len(df_low)} low projections")
    print(f"  ✓ Loaded {len(df_mc)} Monte Carlo trials")

    # Create DCF waterfall chart
    print("\n[2/4] Creating DCF valuation waterfall chart...")
    fig_dcf = create_dcf_waterfall_chart()
    print("  ✓ DCF waterfall chart created")

    # Create sensitivity heatmap
    print("\n[3/4] Creating sensitivity analysis heatmap...")
    fig_heatmap = create_sensitivity_heatmap()
    print("  ✓ Sensitivity heatmap created")

    # Create Monte Carlo waterfall
    print("\n[4/4] Creating Monte Carlo distribution waterfall...")
    fig_waterfall = create_monte_carlo_waterfall()
    print("  ✓ Monte Carlo waterfall created")

    print("\n" + "=" * 50)
    print("Dashboard generation complete!")
    print("\nTo view the figures:")
    print("  - fig_dcf.show()")
    print("  - fig_heatmap.show()")
    print("  - fig_waterfall.show()")

    # Store figures as module-level variables for export
    globals()["fig_dcf"] = fig_dcf
    globals()["fig_heatmap"] = fig_heatmap
    globals()["fig_waterfall"] = fig_waterfall

    return fig_dcf, fig_heatmap, fig_waterfall


if __name__ == "__main__":
    fig_dcf, fig_heatmap, fig_waterfall = main()
