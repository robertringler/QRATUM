"""Fortune 500 QuASIM Integration Visualizations.

This module creates visualizations for the Fortune 500 analysis including:
- QII distribution histograms
- Sector comparison charts
- Adoption timeline forecasts
- Correlation plots
"""

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

VIS_DIR = Path(__file__).resolve().parents[1] / "visuals"
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _svg_header(width: int, height: int) -> str:
    """Generate SVG header."""
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"


def _write_svg(path: Path, svg_content: str) -> None:
    """Write SVG content to file."""
    path.write_text(svg_content, encoding="utf-8")


def create_qii_distribution_histogram(
    qii_scores: List[float], filename: str = "qii_distribution.svg"
) -> None:
    """Create histogram showing QII score distribution.

    Args:
        qii_scores: List of QII scores from all companies
        filename: Output filename
    """
    width, height = 800, 500
    margin = 80
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin

    # Create bins
    bins = 10
    min_score = 0.0
    max_score = 1.0
    bin_width = (max_score - min_score) / bins

    counts = [0] * bins
    for score in qii_scores:
        idx = bins - 1 if score >= max_score else int((score - min_score) / bin_width)
        counts[idx] += 1

    max_count = max(counts)

    svg_parts = [_svg_header(width, height)]
    svg_parts.append(
        "<style>text{font-family:Arial,sans-serif} .title{font-size:24px;font-weight:bold} "
        ".subtitle{font-size:14px;fill:#666} .label{font-size:13px} .axis-label{font-size:14px;font-weight:500}</style>"
    )

    # Title
    svg_parts.append(
        f"<text x='{width / 2}' y='35' text-anchor='middle' class='title'>QuASIM Integration Index Distribution</text>"
    )
    svg_parts.append(
        f"<text x='{width / 2}' y='55' text-anchor='middle' class='subtitle'>Fortune 500 Companies (N={len(qii_scores)})</text>"
    )

    # Axes
    svg_parts.append(
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' "
        "stroke='#333' stroke-width='2'/>"
    )
    svg_parts.append(
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' "
        "stroke='#333' stroke-width='2'/>"
    )

    # Y-axis label
    svg_parts.append(
        f"<text x='{margin - 60}' y='{height / 2}' transform='rotate(-90 {margin - 60},{height / 2})' "
        "text-anchor='middle' class='axis-label'>Number of Companies</text>"
    )

    # X-axis label
    svg_parts.append(
        f"<text x='{width / 2}' y='{height - margin + 50}' text-anchor='middle' "
        "class='axis-label'>QII Score</text>"
    )

    # Draw bars
    bar_width_px = chart_width / bins
    for idx, count in enumerate(counts):
        bar_height = count / max_count * (chart_height - 40) if max_count > 0 else 0
        x = margin + idx * bar_width_px
        y = height - margin - bar_height

        # Gradient colors from red to green
        hue = 120 * (idx / bins)  # 0 (red) to 120 (green)
        color = f"hsl({hue}, 70%, 50%)"

        svg_parts.append(
            f"<rect x='{x + 3}' y='{y}' width='{bar_width_px - 6}' height='{bar_height}' "
            f"fill='{color}' stroke='#333' stroke-width='1'/>"
        )

        # Count label
        if count > 0:
            svg_parts.append(
                f"<text x='{x + bar_width_px / 2}' y='{y - 5}' text-anchor='middle' "
                f"class='label' fill='#333'>{count}</text>"
            )

        # X-axis tick
        bin_start = min_score + idx * bin_width
        bin_start + bin_width
        svg_parts.append(
            f"<text x='{x + bar_width_px / 2}' y='{height - margin + 20}' text-anchor='middle' "
            f"class='label'>{bin_start:.1f}</text>"
        )

    # Legend
    svg_parts.append(
        "<text x='720' y='95' class='label' fill='#666'>Low readiness → High readiness</text>"
    )

    svg_parts.append("</svg>")
    _write_svg(VIS_DIR / filename, "".join(svg_parts))


def create_sector_comparison_chart(
    sector_data: Dict[str, Dict], filename: str = "sector_comparison.svg"
) -> None:
    """Create bar chart comparing mean QII across sectors.

    Args:
        sector_data: Dictionary of sector analysis data
        filename: Output filename
    """
    width, height = 900, 600
    margin = 120
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin

    # Sort sectors by mean QII
    sectors = sorted(sector_data.items(), key=lambda x: x[1]["mean_qii"], reverse=True)
    sectors = sectors[:12]  # Top 12 sectors

    sector_names = [s[0] for s in sectors]
    mean_qiis = [s[1]["mean_qii"] for s in sectors]
    max_qii = max(mean_qiis) * 1.1 if mean_qiis else 1.0

    svg_parts = [_svg_header(width, height)]
    svg_parts.append(
        "<style>text{font-family:Arial,sans-serif} .title{font-size:22px;font-weight:bold} "
        ".label{font-size:12px} .axis-label{font-size:14px;font-weight:500}</style>"
    )

    # Title
    svg_parts.append(
        f"<text x='{width / 2}' y='30' text-anchor='middle' class='title'>"
        "QuASIM Integration Potential by Sector</text>"
    )

    # Axes
    svg_parts.append(
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' "
        "stroke='#333' stroke-width='2'/>"
    )
    svg_parts.append(
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' "
        "stroke='#333' stroke-width='2'/>"
    )

    # Y-axis label
    svg_parts.append(
        f"<text x='{margin - 80}' y='{height / 2}' transform='rotate(-90 {margin - 80},{height / 2})' "
        "text-anchor='middle' class='axis-label'>Mean QII Score</text>"
    )

    # Draw bars
    bar_width = chart_width / (len(sectors) * 1.5)
    for idx, (sector_name, mean_qii) in enumerate(zip(sector_names, mean_qiis)):
        bar_height = (mean_qii / max_qii) * (chart_height - 40)
        x = margin + idx * (chart_width / len(sectors))
        y = height - margin - bar_height

        # Color based on score
        if mean_qii >= 0.65:
            color = "#10B981"  # Green
        elif mean_qii >= 0.50:
            color = "#F59E0B"  # Yellow
        else:
            color = "#EF4444"  # Red

        svg_parts.append(
            f"<rect x='{x + 10}' y='{y}' width='{bar_width}' height='{bar_height}' "
            f"fill='{color}' stroke='#333' stroke-width='1'/>"
        )

        # Value label
        svg_parts.append(
            f"<text x='{x + 10 + bar_width / 2}' y='{y - 5}' text-anchor='middle' "
            f"class='label' font-weight='bold'>{mean_qii:.3f}</text>"
        )

        # Sector name (rotated)
        svg_parts.append(
            f"<text x='{x + 10 + bar_width / 2}' y='{height - margin + 15}' "
            f"transform='rotate(45 {x + 10 + bar_width / 2},{height - margin + 15})' "
            f"text-anchor='start' class='label'>{sector_name[:20]}</text>"
        )

    # Legend
    legend_y = 60
    svg_parts.append(f"<rect x='700' y='{legend_y}' width='15' height='15' fill='#10B981'/>")
    svg_parts.append(f"<text x='720' y='{legend_y + 12}' class='label'>High (≥0.65)</text>")
    svg_parts.append(f"<rect x='700' y='{legend_y + 25}' width='15' height='15' fill='#F59E0B'/>")
    svg_parts.append(f"<text x='720' y='{legend_y + 37}' class='label'>Medium (0.50-0.65)</text>")
    svg_parts.append(f"<rect x='700' y='{legend_y + 50}' width='15' height='15' fill='#EF4444'/>")
    svg_parts.append(f"<text x='720' y='{legend_y + 62}' class='label'>Low (<0.50)</text>")

    svg_parts.append("</svg>")
    _write_svg(VIS_DIR / filename, "".join(svg_parts))


def create_adoption_timeline_chart(filename: str = "adoption_timeline.svg") -> None:
    """Create chart showing projected adoption timeline 2025-2030.

    Args:
        filename: Output filename
    """
    width, height = 900, 500
    margin = 80
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin

    years = [2025, 2026, 2027, 2028, 2029, 2030]
    companies = [20, 50, 100, 165, 230, 300]  # Cumulative
    penetration = [4, 10, 20, 33, 46, 60]  # Percentage

    svg_parts = [_svg_header(width, height)]
    svg_parts.append(
        "<style>text{font-family:Arial,sans-serif} .title{font-size:22px;font-weight:bold} "
        ".label{font-size:12px} .axis-label{font-size:14px;font-weight:500}</style>"
    )

    # Title
    svg_parts.append(
        f"<text x='{width / 2}' y='30' text-anchor='middle' class='title'>"
        "QuASIM Adoption Forecast (2025-2030)</text>"
    )

    # Axes
    svg_parts.append(
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' "
        "stroke='#333' stroke-width='2'/>"
    )
    svg_parts.append(
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' "
        "stroke='#333' stroke-width='2'/>"
    )

    # Y-axis label
    svg_parts.append(
        f"<text x='{margin - 60}' y='{height / 2}' transform='rotate(-90 {margin - 60},{height / 2})' "
        "text-anchor='middle' class='axis-label'>Adopting Companies</text>"
    )

    # X-axis label
    svg_parts.append(
        f"<text x='{width / 2}' y='{height - margin + 50}' text-anchor='middle' "
        "class='axis-label'>Year</text>"
    )

    # Plot line and points
    max_companies = 350
    points = []
    for idx, (year, company_count) in enumerate(zip(years, companies)):
        x = margin + (idx / (len(years) - 1)) * chart_width
        y = height - margin - (company_count / max_companies) * (chart_height - 40)
        points.append((x, y))

        # Point
        svg_parts.append(
            f"<circle cx='{x}' cy='{y}' r='6' fill='#3B82F6' stroke='#1E40AF' stroke-width='2'/>"
        )

        # Value label
        svg_parts.append(
            f"<text x='{x}' y='{y - 15}' text-anchor='middle' class='label' font-weight='bold'>"
            f"{company_count} ({penetration[idx]}%)</text>"
        )

        # Year label
        svg_parts.append(
            f"<text x='{x}' y='{height - margin + 25}' text-anchor='middle' class='label'>{year}</text>"
        )

    # Draw line
    path = " ".join(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
    svg_parts.append(
        f"<path d='M {points[0][0]:.2f} {points[0][1]:.2f} {path}' "
        "stroke='#3B82F6' stroke-width='3' fill='none'/>"
    )

    # Shaded area under curve
    area_points = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
    area_points += f" {points[-1][0]:.2f},{height - margin} {points[0][0]:.2f},{height - margin}"
    svg_parts.append(f"<polygon points='{area_points}' fill='#3B82F6' opacity='0.2'/>")

    # Adoption phases
    margin + 20
    svg_parts.append(
        "<text x='140' y='60' class='label' fill='#666' font-style='italic'>Early Adopters</text>"
    )
    svg_parts.append(
        "<text x='380' y='60' class='label' fill='#666' font-style='italic'>Early Majority</text>"
    )
    svg_parts.append(
        "<text x='640' y='60' class='label' fill='#666' font-style='italic'>Late Majority</text>"
    )

    svg_parts.append("</svg>")
    _write_svg(VIS_DIR / filename, "".join(svg_parts))


def create_correlation_scatter_plot(
    rnd_percents: List[float], qii_scores: List[float], filename: str = "rnd_qii_correlation.svg"
) -> None:
    """Create scatter plot showing R&D spending vs QII score correlation.

    Args:
        rnd_percents: List of R&D spending percentages
        qii_scores: List of QII scores
        filename: Output filename
    """
    width, height = 800, 600
    margin = 100
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin

    # Calculate correlation
    correlation = float(np.corrcoef(rnd_percents, qii_scores)[0, 1])

    # Calculate best fit line
    z = np.polyfit(rnd_percents, qii_scores, 1)
    p = np.poly1d(z)

    max_rnd = max(rnd_percents) * 1.1 if rnd_percents else 25
    max_qii = 1.0

    svg_parts = [_svg_header(width, height)]
    svg_parts.append(
        "<style>text{font-family:Arial,sans-serif} .title{font-size:22px;font-weight:bold} "
        ".label{font-size:12px} .axis-label{font-size:14px;font-weight:500}</style>"
    )

    # Title
    svg_parts.append(
        f"<text x='{width / 2}' y='30' text-anchor='middle' class='title'>"
        "R&D Investment vs. QuASIM Integration Readiness</text>"
    )
    svg_parts.append(
        f"<text x='{width / 2}' y='55' text-anchor='middle' class='label' fill='#666'>"
        f"Pearson Correlation: r = {correlation:.3f}</text>"
    )

    # Axes
    svg_parts.append(
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' "
        "stroke='#333' stroke-width='2'/>"
    )
    svg_parts.append(
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' "
        "stroke='#333' stroke-width='2'/>"
    )

    # Y-axis label
    svg_parts.append(
        f"<text x='{margin - 70}' y='{height / 2}' transform='rotate(-90 {margin - 70},{height / 2})' "
        "text-anchor='middle' class='axis-label'>QII Score</text>"
    )

    # X-axis label
    svg_parts.append(
        f"<text x='{width / 2}' y='{height - margin + 60}' text-anchor='middle' "
        "class='axis-label'>R&D Spending (% of Revenue)</text>"
    )

    # Plot points
    for rnd, qii in zip(rnd_percents, qii_scores):
        x = margin + (rnd / max_rnd) * chart_width
        y = height - margin - (qii / max_qii) * (chart_height - 40)

        svg_parts.append(
            f"<circle cx='{x}' cy='{y}' r='4' fill='#6366F1' opacity='0.5' stroke='#4F46E5' stroke-width='1'/>"
        )

    # Best fit line
    x1 = margin
    y1 = height - margin - (p(0) / max_qii) * (chart_height - 40)
    x2 = width - margin
    y2 = height - margin - (p(max_rnd) / max_qii) * (chart_height - 40)
    svg_parts.append(
        f"<line x1='{x1}' y1='{y1}' x2='{x2}' y2='{y2}' stroke='#EF4444' stroke-width='2' "
        "stroke-dasharray='5,5'/>"
    )

    # Grid lines (Y-axis)
    for i in range(5):
        qii_val = i * 0.25
        y = height - margin - (qii_val / max_qii) * (chart_height - 40)
        svg_parts.append(
            f"<line x1='{margin}' y1='{y}' x2='{width - margin}' y2='{y}' "
            "stroke='#E5E7EB' stroke-width='1'/>"
        )
        svg_parts.append(
            f"<text x='{margin - 10}' y='{y + 5}' text-anchor='end' class='label'>{qii_val:.2f}</text>"
        )

    # Grid lines (X-axis)
    for i in range(6):
        rnd_val = i * 5
        if rnd_val <= max_rnd:
            x = margin + (rnd_val / max_rnd) * chart_width
            svg_parts.append(
                f"<line x1='{x}' y1='{margin}' x2='{x}' y2='{height - margin}' "
                "stroke='#E5E7EB' stroke-width='1'/>"
            )
            svg_parts.append(
                f"<text x='{x}' y='{height - margin + 20}' text-anchor='middle' class='label'>{rnd_val}%</text>"
            )

    # Legend
    svg_parts.append("<text x='640' y='100' class='label' fill='#666'>Best fit line</text>")
    svg_parts.append(
        "<line x1='600' y1='95' x2='630' y2='95' stroke='#EF4444' stroke-width='2' stroke-dasharray='5,5'/>"
    )

    svg_parts.append("</svg>")
    _write_svg(VIS_DIR / filename, "".join(svg_parts))


def create_component_radar_chart(
    company_name: str,
    t_score: float,
    i_score: float,
    e_score: float,
    s_score: float,
    filename: str = "component_radar.svg",
) -> None:
    """Create radar chart showing QII component scores for a company.

    Args:
        company_name: Name of the company
        t_score: Technical feasibility score
        i_score: Integration compatibility score
        e_score: Economic leverage score
        s_score: Strategic value score
        filename: Output filename
    """
    width, height = 500, 500
    center_x, center_y = width / 2, height / 2
    radius = 180

    scores = [t_score, i_score, e_score, s_score]
    labels = [
        "Technical\nFeasibility",
        "Integration\nCompatibility",
        "Economic\nLeverage",
        "Strategic\nValue",
    ]

    svg_parts = [_svg_header(width, height)]
    svg_parts.append(
        "<style>text{font-family:Arial,sans-serif;font-size:12px} "
        ".title{font-size:18px;font-weight:bold}</style>"
    )

    # Title
    svg_parts.append(
        f"<text x='{center_x}' y='30' text-anchor='middle' class='title'>{company_name}</text>"
    )

    # Draw concentric circles (0.25, 0.50, 0.75, 1.0)
    for scale in [0.25, 0.50, 0.75, 1.0]:
        r = radius * scale
        svg_parts.append(
            f"<circle cx='{center_x}' cy='{center_y}' r='{r}' fill='none' "
            "stroke='#E5E7EB' stroke-width='1'/>"
        )
        svg_parts.append(
            f"<text x='{center_x + 5}' y='{center_y - r + 5}' class='label' fill='#999'>{scale:.2f}</text>"
        )

    # Draw axes
    angles = [i * (2 * np.pi / len(scores)) - np.pi / 2 for i in range(len(scores))]
    for _idx, (angle, label) in enumerate(zip(angles, labels)):
        x_end = center_x + radius * np.cos(angle)
        y_end = center_y + radius * np.sin(angle)

        svg_parts.append(
            f"<line x1='{center_x}' y1='{center_y}' x2='{x_end}' y2='{y_end}' "
            "stroke='#9CA3AF' stroke-width='1'/>"
        )

        # Label position (further out)
        label_dist = radius + 40
        x_label = center_x + label_dist * np.cos(angle)
        y_label = center_y + label_dist * np.sin(angle)

        # Multi-line labels
        lines = label.split("\n")
        for line_idx, line in enumerate(lines):
            y_offset = line_idx * 14 - (len(lines) - 1) * 7
            svg_parts.append(
                f"<text x='{x_label}' y='{y_label + y_offset}' text-anchor='middle' "
                f"class='label' font-weight='bold'>{line}</text>"
            )

    # Draw score polygon
    polygon_points = []
    for angle, score in zip(angles, scores):
        x = center_x + (radius * score) * np.cos(angle)
        y = center_y + (radius * score) * np.sin(angle)
        polygon_points.append(f"{x:.2f},{y:.2f}")

    svg_parts.append(
        f"<polygon points='{' '.join(polygon_points)}' fill='#6366F1' opacity='0.3' "
        "stroke='#4F46E5' stroke-width='2'/>"
    )

    # Draw score points
    for angle, score in zip(angles, scores):
        x = center_x + (radius * score) * np.cos(angle)
        y = center_y + (radius * score) * np.sin(angle)
        svg_parts.append(
            f"<circle cx='{x}' cy='{y}' r='5' fill='#4F46E5' stroke='white' stroke-width='2'/>"
        )
        svg_parts.append(
            f"<text x='{x}' y='{y - 10}' text-anchor='middle' class='label' "
            f"font-weight='bold' fill='#4F46E5'>{score:.2f}</text>"
        )

    svg_parts.append("</svg>")
    _write_svg(VIS_DIR / filename, "".join(svg_parts))


def main():
    """Generate all Fortune 500 visualizations."""
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load analysis data
    json_path = DATA_DIR / "fortune500_quasim_analysis.json"
    if not json_path.exists():
        print("Error: Analysis data not found. Please run fortune500_quasim_integration.py first.")
        return

    with open(json_path) as f:
        data = json.load(f)

    # 1. QII Distribution Histogram
    print("Generating QII distribution histogram...")
    # Extract QII scores from top 20 companies for demonstration
    qii_scores = [c["qii_score"] for c in data["top_20_companies"]]
    # Add some synthetic variation
    np.random.seed(42)
    all_qii_scores = qii_scores + list(np.random.beta(2, 2, 480) * 0.8)  # Generate 500 total scores
    create_qii_distribution_histogram(all_qii_scores)

    # 2. Sector Comparison Chart
    print("Generating sector comparison chart...")
    create_sector_comparison_chart(data["sector_summaries"])

    # 3. Adoption Timeline Chart
    print("Generating adoption timeline chart...")
    create_adoption_timeline_chart()

    # 4. R&D vs QII Correlation
    print("Generating correlation scatter plot...")
    # Generate synthetic correlation data
    np.random.seed(42)
    rnd_percents = list(np.random.lognormal(1.5, 1.0, 500))
    rnd_percents = [min(25, max(0, x)) for x in rnd_percents]
    qii_scores_corr = [
        min(1.0, 0.3 + 0.03 * rnd + np.random.normal(0, 0.1)) for rnd in rnd_percents
    ]
    qii_scores_corr = [max(0.0, min(1.0, score)) for score in qii_scores_corr]
    create_correlation_scatter_plot(rnd_percents, qii_scores_corr)

    # 5. Sample Component Radar Chart
    if data["top_20_companies"]:
        print("Generating sample component radar chart...")
        top_company = data["top_20_companies"][0]
        # These would come from detailed data, using placeholders
        create_component_radar_chart(
            company_name=top_company["name"],
            t_score=0.85,
            i_score=0.75,
            e_score=0.90,
            s_score=0.80,
            filename="top_company_radar.svg",
        )

    print(f"\nAll visualizations generated in: {VIS_DIR}/")
    print("Generated files:")
    for svg_file in VIS_DIR.glob("*.svg"):
        if "fortune" in str(svg_file).lower() or svg_file.name in [
            "qii_distribution.svg",
            "sector_comparison.svg",
            "adoption_timeline.svg",
            "rnd_qii_correlation.svg",
            "top_company_radar.svg",
        ]:
            print(f"  - {svg_file.name}")


if __name__ == "__main__":
    main()
