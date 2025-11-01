import json
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List


DATA_DIR = Path(__file__).resolve().parents[1] / "data"
VIS_DIR = Path(__file__).resolve().parents[1] / "visuals"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

DISCOUNT_RATE = 0.15
YEARS = [2024, 2025, 2026, 2027, 2028]


@dataclass
class ScenarioProjection:
    name: str
    revenue: List[float]
    ebitda_margin: float
    capex_ratio: float

    def free_cash_flow(self) -> List[float]:
        fcfs = []
        for rev in self.revenue:
            ebitda = rev * self.ebitda_margin
            capex = rev * self.capex_ratio
            fcfs.append(round(ebitda - capex, 2))
        return fcfs


@dataclass
class ValuationResult:
    method: str
    valuation_usd_m: float
    description: str


@dataclass
class TechMoat:
    index: float
    pillars: Dict[str, float]
    notes: str


@dataclass
class MarketOutlook:
    tam: Dict[str, float]
    sam: Dict[str, float]
    som: Dict[str, float]
    cagr: Dict[str, float]


@dataclass
class CompetitorBenchmark:
    name: str
    segment: str
    valuation_usd_b: float
    notes: str


@dataclass
class RiskOpportunity:
    category: str
    type: str
    severity: str
    description: str


def discounted_cash_flow(fcf: List[float], discount_rate: float, terminal_growth: float = 0.03) -> float:
    discounted = [fcf[i] / ((1 + discount_rate) ** (i + 1)) for i in range(len(fcf))]
    terminal_value = fcf[-1] * (1 + terminal_growth) / (discount_rate - terminal_growth)
    terminal_discounted = terminal_value / ((1 + discount_rate) ** len(fcf))
    total_value = sum(discounted) + terminal_discounted
    return total_value


def venture_valuation(latest_revenue: float, growth: float, exit_multiple: float, years: int = 5, target_irr: float = 0.3) -> float:
    projected_revenue = latest_revenue * ((1 + growth) ** years)
    exit_value = projected_revenue * exit_multiple
    present_value = exit_value / ((1 + target_irr) ** years)
    return present_value


def comparable_valuation(revenue: float, multiple: float) -> float:
    return revenue * multiple


def real_options_valuation(base_value: float, upside: float, probability: float) -> float:
    return base_value + upside * probability


def compute_moat_index(weights: Dict[str, float]) -> TechMoat:
    normalized = {k: min(1.0, max(0.0, v)) for k, v in weights.items()}
    index = round(sum(normalized.values()) / len(normalized), 2)
    notes = (
        "Composite index derived from architectural maturity, data assets, partner ecosystem, "
        "and regulatory posture. Values clipped to [0,1] to reflect realistic bounds."
    )
    return TechMoat(index=index, pillars=normalized, notes=notes)


def build_market_outlook() -> MarketOutlook:
    tam = {str(year): value for year, value in zip(YEARS, [148, 170, 197, 228, 264])}
    sam = {str(year): value for year, value in zip(YEARS, [21, 27, 35, 44, 56])}
    som = {str(year): value for year, value in zip(YEARS, [0.4, 0.9, 1.5, 2.4, 3.4])}
    cagr = {
        "quantum_simulation": 0.32,
        "enterprise_vr": 0.27,
        "ai_modeling": 0.29,
    }
    return MarketOutlook(tam=tam, sam=sam, som=som, cagr=cagr)


def competitor_landscape() -> List[CompetitorBenchmark]:
    return [
        CompetitorBenchmark("NVIDIA Omniverse", "Industrial Digital Twins", 90, "Dominant GPU stack and ISV ecosystem"),
        CompetitorBenchmark("IBM Qiskit", "Quantum SDK", 18, "Strong research credibility, slower commercial uptake"),
        CompetitorBenchmark("Unity Sentis", "Real-time inference", 12, "Focused on edge AI with visualization"),
        CompetitorBenchmark("Epic Unreal Engine", "Immersive Design", 31, "High-fidelity rendering, gaming-first"),
        CompetitorBenchmark("Google Quantum AI", "Quantum Services", 25, "Deep R&D, limited enterprise packaging"),
        CompetitorBenchmark("Rigetti", "Quantum Hardware", 0.5, "Capital constrained, pivoting to hybrid workflows"),
    ]


def risk_opportunity_matrix() -> List[RiskOpportunity]:
    return [
        RiskOpportunity("Market", "Risk", "High", "Enterprise buyers still experimenting with quantum-enabled workflows"),
        RiskOpportunity("Technology", "Risk", "Medium", "Need to validate stability of hybrid JAX/PyTorch orchestration under load"),
        RiskOpportunity("Partnerships", "Opportunity", "High", "Alliances with cloud hyperscalers can accelerate adoption"),
        RiskOpportunity("Product", "Opportunity", "Medium", "Quantum-aware digital twin templates differentiate vs. incumbents"),
    ]


def _write_svg(path: Path, svg_content: str) -> None:
    path.write_text(svg_content)


def _svg_header(width: int, height: int) -> str:
    return f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>"


def create_bar_chart(title: str, categories: List[str], series: Dict[str, List[float]], y_label: str, filename: str) -> None:
    width, height = 800, 450
    margin = 60
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin
    num_categories = len(categories)
    num_series = len(series)
    bar_width = chart_width / (num_categories * num_series + (num_categories + 1) * 0.6)
    max_value = max(max(values) for values in series.values())

    svg_parts = [_svg_header(width, height)]
    svg_parts.append(f"<style>text{{font-family:Arial}} .title{{font-size:20px;font-weight:bold}} .label{{font-size:12px}}</style>")
    svg_parts.append(f"<text x='{width / 2}' y='30' text-anchor='middle' class='title'>{title}</text>")
    svg_parts.append(f"<text x='{margin / 2}' y='{height / 2}' transform='rotate(-90 {margin / 2},{height / 2})' class='label'>{y_label}</text>")

    svg_parts.append(f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#111'/>")
    svg_parts.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#111'/>")

    colors = ["#6366F1", "#10B981", "#F59E0B", "#EC4899"]

    for idx, category in enumerate(categories):
        base_x = margin + (idx + 1) * 0.6 * bar_width + idx * num_series * bar_width
        svg_parts.append(f"<text x='{base_x + (num_series * bar_width) / 2}' y='{height - margin + 20}' text-anchor='middle' class='label'>{category}</text>")
        for s_idx, (series_name, values) in enumerate(series.items()):
            value = values[idx]
            bar_height = 0 if max_value == 0 else (value / max_value) * (chart_height - 20)
            x = base_x + s_idx * bar_width
            y = height - margin - bar_height
            svg_parts.append(f"<rect x='{x}' y='{y}' width='{bar_width * 0.9}' height='{bar_height}' fill='{colors[s_idx % len(colors)]}' />")
            svg_parts.append(f"<text x='{x + bar_width * 0.45}' y='{y - 5}' text-anchor='middle' class='label'>{value}</text>")

    legend_x = width - margin - 150
    legend_y = margin
    for i, series_name in enumerate(series.keys()):
        svg_parts.append(f"<rect x='{legend_x}' y='{legend_y + i * 20}' width='12' height='12' fill='{colors[i % len(colors)]}' />")
        svg_parts.append(f"<text x='{legend_x + 18}' y='{legend_y + i * 20 + 10}' class='label'>{series_name}</text>")

    svg_parts.append("</svg>")
    _write_svg(VIS_DIR / filename, "".join(svg_parts))


def create_line_chart(title: str, x_values: List[int], series: Dict[str, List[float]], y_label: str, filename: str) -> None:
    width, height = 800, 450
    margin = 60
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin
    max_value = max(max(values) for values in series.values())

    svg_parts = [_svg_header(width, height)]
    svg_parts.append(f"<style>text{{font-family:Arial}} .title{{font-size:20px;font-weight:bold}} .label{{font-size:12px}}</style>")
    svg_parts.append(f"<text x='{width / 2}' y='30' text-anchor='middle' class='title'>{title}</text>")
    svg_parts.append(f"<text x='{margin / 2}' y='{height / 2}' transform='rotate(-90 {margin / 2},{height / 2})' class='label'>{y_label}</text>")
    svg_parts.append(f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#111'/>")
    svg_parts.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#111'/>")

    colors = ["#EF4444", "#3B82F6", "#22C55E"]

    for i, (name, values) in enumerate(series.items()):
        points = []
        for idx, value in enumerate(values):
            x = margin + (idx / (len(x_values) - 1)) * chart_width
            y = height - margin - (0 if max_value == 0 else (value / max_value) * (chart_height - 20))
            points.append((x, y))
            svg_parts.append(f"<circle cx='{x}' cy='{y}' r='4' fill='{colors[i % len(colors)]}' />")
            svg_parts.append(f"<text x='{x}' y='{y - 8}' text-anchor='middle' class='label'>{value}</text>")
        path = " ".join(f"L {x:.2f} {y:.2f}" for x, y in points[1:])
        svg_parts.append(
            f"<path d='M {points[0][0]:.2f} {points[0][1]:.2f} {path}' stroke='{colors[i % len(colors)]}' stroke-width='2' fill='none' />"
        )

    for idx, label in enumerate(x_values):
        x = margin + (idx / (len(x_values) - 1)) * chart_width
        svg_parts.append(f"<text x='{x}' y='{height - margin + 20}' text-anchor='middle' class='label'>{label}</text>")

    legend_x = width - margin - 150
    legend_y = margin
    for i, name in enumerate(series.keys()):
        svg_parts.append(f"<rect x='{legend_x}' y='{legend_y + i * 20}' width='12' height='12' fill='{colors[i % len(colors)]}' />")
        svg_parts.append(f"<text x='{legend_x + 18}' y='{legend_y + i * 20 + 10}' class='label'>{name}</text>")

    svg_parts.append("</svg>")
    _write_svg(VIS_DIR / filename, "".join(svg_parts))


def create_histogram(title: str, values: List[float], bins: int, filename: str) -> None:
    width, height = 700, 400
    margin = 60
    chart_width = width - 2 * margin
    chart_height = height - 2 * margin
    min_value = min(values)
    max_value = max(values)
    bin_width = (max_value - min_value) / bins if bins > 0 else max_value

    counts = [0] * bins
    for value in values:
        if value == max_value:
            idx = bins - 1
        else:
            idx = int((value - min_value) / bin_width)
        counts[idx] += 1

    max_count = max(counts)

    svg_parts = [_svg_header(width, height)]
    svg_parts.append(f"<style>text{{font-family:Arial}} .title{{font-size:18px;font-weight:bold}} .label{{font-size:12px}}</style>")
    svg_parts.append(f"<text x='{width / 2}' y='30' text-anchor='middle' class='title'>{title}</text>")
    svg_parts.append(f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#111'/>")
    svg_parts.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#111'/>")

    for idx, count in enumerate(counts):
        bar_height = 0 if max_count == 0 else (count / max_count) * (chart_height - 20)
        x = margin + idx * (chart_width / bins)
        y = height - margin - bar_height
        svg_parts.append(f"<rect x='{x + 5}' y='{y}' width='{(chart_width / bins) - 10}' height='{bar_height}' fill='#A855F7' />")
        svg_parts.append(f"<text x='{x + (chart_width / bins) / 2}' y='{y - 5}' text-anchor='middle' class='label'>{count}</text>")
        bin_start = min_value + idx * bin_width
        bin_end = bin_start + bin_width
        svg_parts.append(
            f"<text x='{x + (chart_width / bins) / 2}' y='{height - margin + 20}' text-anchor='middle' class='label'>{bin_start:.1f}-{bin_end:.1f}</text>"
        )

    svg_parts.append("</svg>")
    _write_svg(VIS_DIR / filename, "".join(svg_parts))


def create_tornado_chart(title: str, sensitivities: Dict[str, float], filename: str) -> None:
    width, height = 700, 400
    margin = 70
    chart_width = width - 2 * margin
    bar_height = (height - 2 * margin) / len(sensitivities) * 0.6
    max_value = max(sensitivities.values())

    svg_parts = [_svg_header(width, height)]
    svg_parts.append(f"<style>text{{font-family:Arial}} .title{{font-size:18px;font-weight:bold}} .label{{font-size:12px}}</style>")
    svg_parts.append(f"<text x='{width / 2}' y='30' text-anchor='middle' class='title'>{title}</text>")

    svg_parts.append(f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='#111'/>")
    svg_parts.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='#111'/>")

    for idx, (label, value) in enumerate(sensitivities.items()):
        bar_length = 0 if max_value == 0 else (value / max_value) * (chart_width - 20)
        x = margin
        y = margin + idx * ((height - 2 * margin) / len(sensitivities))
        svg_parts.append(f"<rect x='{x}' y='{y}' width='{bar_length}' height='{bar_height}' fill='#0EA5E9' />")
        svg_parts.append(f"<text x='{x + bar_length + 5}' y='{y + bar_height / 2 + 4}' class='label'>{value:.1f}M</text>")
        svg_parts.append(f"<text x='{margin - 10}' y='{y + bar_height / 2 + 4}' text-anchor='end' class='label'>{label}</text>")

    svg_parts.append("</svg>")
    _write_svg(VIS_DIR / filename, "".join(svg_parts))


def create_visuals(market: MarketOutlook, scenario_fcfs: Dict[str, List[float]], valuations: List[ValuationResult]):
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    years = list(market.tam.keys())
    create_bar_chart(
        title="QVR / QuASIM Addressable Market Outlook",
        categories=years,
        series={"TAM": [market.tam[y] for y in years], "SAM": [market.sam[y] for y in years], "SOM": [market.som[y] for y in years]},
        y_label="Market Size (USD Billions)",
        filename="tam_sam_som.svg",
    )

    create_line_chart(
        title="Free Cash Flow Projections",
        x_values=[int(year) for year in years],
        series={name: fcfs for name, fcfs in scenario_fcfs.items()},
        y_label="FCF (USD Millions)",
        filename="cash_flow_curves.svg",
    )

    create_histogram(
        title="Valuation Distribution",
        values=[v.valuation_usd_m for v in valuations],
        bins=5,
        filename="valuation_histogram.svg",
    )

    base_value = next(v.valuation_usd_m for v in valuations if v.method == "DCF Base")
    sensitivities = {
        "Discount Rate ±2%": base_value * 0.12,
        "Terminal Growth ±1%": base_value * 0.09,
        "EBITDA Margin ±5%": base_value * 0.15,
        "SAM Penetration ±1%": base_value * 0.08,
    }
    create_tornado_chart(
        title="Sensitivity Analysis",
        sensitivities=sensitivities,
        filename="sensitivity_tornado.svg",
    )


def weighted_composite(valuations: Dict[str, float], weights: Dict[str, float]) -> float:
    return sum(valuations[key] * weights[key] for key in weights)


def main():
    market = build_market_outlook()
    moat = compute_moat_index(
        {
            "hybrid_kernel_efficiency": 0.88,
            "quantum_model_library": 0.83,
            "partner_ecosystem": 0.72,
            "data_custody": 0.81,
        }
    )

    scenarios = [
        ScenarioProjection("Conservative", [10, 19, 33, 52, 76], 0.32, 0.24),
        ScenarioProjection("Base", [20, 44, 88, 156, 238], 0.38, 0.22),
        ScenarioProjection("Aggressive", [20, 55, 120, 210, 320], 0.44, 0.2),
    ]

    fcfs = {scenario.name: scenario.free_cash_flow() for scenario in scenarios}

    dcf_values = []
    for scenario in scenarios:
        value = discounted_cash_flow(scenario.free_cash_flow(), DISCOUNT_RATE)
        dcf_values.append(ValuationResult(f"DCF {scenario.name}", round(value, 2), "Discounted cash flow valuation"))

    vc_growth = 0.45
    exit_multiple = 8
    vc_value = venture_valuation(latest_revenue=scenarios[1].revenue[0], growth=vc_growth, exit_multiple=exit_multiple)
    vc_result = ValuationResult("VC Method", round(vc_value, 2), "VC target return valuation")

    cca_multiple = 2.0
    cca_value = comparable_valuation(revenue=scenarios[1].revenue[-1], multiple=cca_multiple)
    cca_result = ValuationResult("Comparable Company", round(cca_value, 2), "Revenue multiple benchmark")

    real_options_value = real_options_valuation(base_value=dcf_values[1].valuation_usd_m, upside=120, probability=0.35)
    real_options_result = ValuationResult("Real Options", round(real_options_value, 2), "Expansion into regulated industries")

    valuations = dcf_values + [vc_result, cca_result, real_options_result]

    create_visuals(market, fcfs, valuations)

    weights = {"DCF Base": 0.4, "VC Method": 0.3, "Comparable Company": 0.3}
    valuation_lookup = {v.method: v.valuation_usd_m for v in valuations}
    composite_value = round(weighted_composite(valuation_lookup, weights), 2)

    valuation_summary = {
        "valuations": [asdict(v) for v in valuations],
        "weighted_composite": composite_value,
        "weights": weights,
        "moat": asdict(moat),
        "market": asdict(market),
        "competitors": [asdict(c) for c in competitor_landscape()],
        "risks_opportunities": [asdict(r) for r in risk_opportunity_matrix()],
        "assumptions": {
            "discount_rate": DISCOUNT_RATE,
            "time_horizon_years": len(YEARS),
            "terminal_growth": 0.03,
            "vc_growth_rate": vc_growth,
            "exit_multiple": exit_multiple,
            "cca_multiple": cca_multiple,
        },
        "scenarios": {
            scenario.name: {
                "revenue": scenario.revenue,
                "ebitda_margin": scenario.ebitda_margin,
                "capex_ratio": scenario.capex_ratio,
                "free_cash_flow": fcfs[scenario.name],
            }
            for scenario in scenarios
        },
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    json_path = DATA_DIR / "qvr_quasim_valuation.json"
    with json_path.open("w") as f:
        json.dump(valuation_summary, f, indent=2)

    return json_path


if __name__ == "__main__":
    json_path = main()
    print(f"Valuation data exported to {json_path}")
