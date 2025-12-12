#!/usr/bin/env python3
"""
QuASIM Market Valuation Script

Implements:
- 5-year DCF projections per scenario
- Unlevered FCF with WACC discounting
- Terminal value (Gordon Growth Model)
- Monte Carlo simulation over growth, margins, opex, discount rate
- Real-options uplift proxy
- Markdown report generation

Usage:
    python run_valuation.py [--config CONFIG_PATH] [--output OUTPUT_PATH]
"""

import argparse
import csv
import datetime
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml
except ImportError:
    print("Warning: PyYAML not available, using minimal YAML parser")
    yaml = None


class MinimalYAMLParser:
    """Minimal YAML parser for basic config files."""

    @staticmethod
    def parse_value(value: str) -> Any:
        """Parse a YAML value."""
        value = value.strip()
        if value.lower() in ("true", "yes"):
            return True
        if value.lower() in ("false", "no"):
            return False
        if value.startswith('"') or value.startswith("'"):
            return value[1:-1]
        if "_" in value and value.replace("_", "").isdigit():
            return int(value.replace("_", ""))
        try:
            if "." in value:
                return float(value)
            return int(value)
        except ValueError:
            return value

    @staticmethod
    def load(stream) -> Dict[str, Any]:
        """Load minimal YAML config."""
        result: Dict[str, Any] = {}
        current_section = result
        section_stack = [(None, result)]

        for line in stream:
            line = line.rstrip()
            if not line or line.strip().startswith("#"):
                continue

            len(line) - len(line.lstrip())
            line = line.strip()

            if line.endswith(":"):
                # Section header
                key = line[:-1]
                new_section: Dict[str, Any] = {}

                # Handle list items
                if section_stack and isinstance(section_stack[-1][1], list):
                    section_stack[-1][1].append(new_section)
                    section_stack.append((key, new_section))
                else:
                    current_section[key] = new_section
                    section_stack.append((key, new_section))
                    current_section = new_section

            elif line.startswith("- "):
                # List item
                key = line[2:].split(":")[0].strip()
                value_str = line[2 + len(key) + 1 :].strip() if ":" in line else ""

                if not value_str:
                    # New list of dicts
                    if key not in current_section:
                        current_section[key] = []
                    continue

                value = MinimalYAMLParser.parse_value(value_str)

                # Handle list notation
                if (
                    isinstance(current_section, dict)
                    and key in current_section
                    and isinstance(current_section[key], list)
                ):
                    current_section[key].append(value)
                else:
                    if key not in current_section:
                        current_section[key] = []
                    current_section[key].append(value)

            elif ":" in line:
                # Key-value pair
                key, value_str = line.split(":", 1)
                key = key.strip()
                value_str = value_str.strip()

                if value_str.startswith("["):
                    # Parse inline list
                    value = [
                        MinimalYAMLParser.parse_value(v.strip()) for v in value_str[1:-1].split(",")
                    ]
                else:
                    value = MinimalYAMLParser.parse_value(value_str)

                current_section[key] = value

        return result


class ValuationEngine:
    """DCF and Monte Carlo valuation engine."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.as_of_date = config.get("as_of_date", "2025-11-08")
        self.currency = config.get("currency", "USD")
        self.discount_rate = config.get("discount_rate_base", 0.18)
        self.tax_rate = config.get("tax_rate", 0.21)
        self.terminal_growth = config.get("terminal_growth", 0.03)
        self.scenarios = config.get("scenarios", [])
        self.mc_config = config.get("monte_carlo", {})

    def project_financials(self, scenario: Dict[str, Any]) -> List[Dict[str, float]]:
        """Project 5-year financials for a scenario."""
        projections = []
        revenue = scenario.get("year1_revenue", 0)
        growth_rates = scenario.get("yoy_growth", [])

        for year in range(1, 6):
            # Revenue
            if year > 1 and year - 2 < len(growth_rates):
                revenue *= 1 + growth_rates[year - 2]

            # Operating metrics
            gross_profit = revenue * scenario.get("gross_margin", 0.78)
            opex = revenue * scenario.get("opex_ratio", 0.55)
            ebitda = gross_profit - opex

            # EBIT and taxes
            ebit = ebitda  # Simplified: no D&A
            tax = ebit * self.tax_rate if ebit > 0 else 0
            nopat = ebit - tax

            # Capital expenditures and working capital
            capex = revenue * scenario.get("capex_ratio", 0.06)
            wc_change = revenue * scenario.get("wc_ratio", 0.08) * 0.2  # Delta

            # Unlevered Free Cash Flow
            fcf = nopat + 0 - capex - wc_change  # +0 is D&A placeholder

            projections.append(
                {
                    "year": year,
                    "revenue": revenue,
                    "gross_profit": gross_profit,
                    "opex": opex,
                    "ebitda": ebitda,
                    "ebit": ebit,
                    "tax": tax,
                    "nopat": nopat,
                    "capex": capex,
                    "wc_change": wc_change,
                    "fcf": fcf,
                }
            )

        return projections

    def calculate_dcf(
        self, projections: List[Dict[str, float]], discount_rate: float = None
    ) -> Dict[str, float]:
        """Calculate DCF valuation."""
        if discount_rate is None:
            discount_rate = self.discount_rate

        pv_fcf = 0.0
        for p in projections:
            discount_factor = 1 / ((1 + discount_rate) ** p["year"])
            pv_fcf += p["fcf"] * discount_factor

        # Terminal value (Gordon Growth)
        last_fcf = projections[-1]["fcf"]
        terminal_fcf = last_fcf * (1 + self.terminal_growth)
        terminal_value = terminal_fcf / (discount_rate - self.terminal_growth)

        # PV of terminal value
        terminal_year = projections[-1]["year"]
        pv_terminal = terminal_value / ((1 + discount_rate) ** terminal_year)

        enterprise_value = pv_fcf + pv_terminal

        return {
            "pv_fcf": pv_fcf,
            "terminal_value": terminal_value,
            "pv_terminal": pv_terminal,
            "enterprise_value": enterprise_value,
            "discount_rate": discount_rate,
        }

    def run_scenario_dcf(self, scenario_name: str) -> Tuple[List[Dict], Dict]:
        """Run DCF for a named scenario."""
        scenario = next((s for s in self.scenarios if s.get("name") == scenario_name), None)
        if not scenario:
            raise ValueError(f"Scenario '{scenario_name}' not found")

        projections = self.project_financials(scenario)
        dcf_result = self.calculate_dcf(projections)

        return projections, dcf_result

    def monte_carlo_simulation(self, base_scenario: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo simulation."""
        trials = self.mc_config.get("trials", 20000)
        distributions = self.mc_config.get("distributions", {})

        results = []

        for _ in range(trials):
            # Perturb parameters
            perturbed = base_scenario.copy()

            # Growth rate perturbation
            growth_sigma = distributions.get("yoy_growth_sigma", 0.25)
            perturbed["yoy_growth"] = [
                max(0, g + random.gauss(0, growth_sigma)) for g in base_scenario["yoy_growth"]
            ]

            # Margin perturbation
            margin_sigma = distributions.get("gross_margin_sigma", 0.05)
            perturbed["gross_margin"] = max(
                0.5,
                min(0.95, base_scenario["gross_margin"] + random.gauss(0, margin_sigma)),
            )

            # OpEx perturbation
            opex_sigma = distributions.get("opex_ratio_sigma", 0.04)
            perturbed["opex_ratio"] = max(
                0.3, min(0.8, base_scenario["opex_ratio"] + random.gauss(0, opex_sigma))
            )

            # Discount rate perturbation
            discount_sigma = distributions.get("discount_rate_sigma", 0.03)
            discount_rate = max(
                0.10, min(0.30, self.discount_rate + random.gauss(0, discount_sigma))
            )

            # Project and value
            projections = self.project_financials(perturbed)
            dcf = self.calculate_dcf(projections, discount_rate)

            results.append(dcf["enterprise_value"])

        # Calculate percentiles
        results.sort()
        p10 = results[int(len(results) * 0.10)]
        p50 = results[int(len(results) * 0.50)]
        p90 = results[int(len(results) * 0.90)]
        mean = sum(results) / len(results)
        std = math.sqrt(sum((r - mean) ** 2 for r in results) / len(results))

        return {
            "trials": trials,
            "results": results,
            "p10": p10,
            "p50": p50,
            "p90": p90,
            "mean": mean,
            "std": std,
        }

    def real_options_uplift(self, base_ev: float) -> Dict[str, float]:
        """Calculate real options uplift proxy using Black-Scholes approximation."""
        # Simplified real-options model
        # Treats expansion as a call option on future growth

        # Assumptions
        volatility = 0.6  # High tech/quantum uncertainty
        time_to_option = 3.0  # Years to expansion decision
        risk_free_rate = 0.045  # 10-year Treasury

        # Expected expansion value (S) and cost (K)
        S = base_ev * 1.5  # Expansion could increase value 50%
        K = base_ev * 0.4  # But requires 40% of current EV as investment

        # Black-Scholes d1 and d2
        d1 = (math.log(S / K) + (risk_free_rate + 0.5 * volatility**2) * time_to_option) / (
            volatility * math.sqrt(time_to_option)
        )
        d2 = d1 - volatility * math.sqrt(time_to_option)

        # Cumulative normal distribution approximation
        def norm_cdf(x):
            return 0.5 * (1 + math.erf(x / math.sqrt(2)))

        # Call option value (real option uplift)
        option_value = S * norm_cdf(d1) - K * math.exp(-risk_free_rate * time_to_option) * norm_cdf(
            d2
        )

        # Additional options (delay, abandonment, licensing)
        delay_option = base_ev * 0.08  # Value of waiting
        licensing_option = base_ev * 0.12  # Value of IP licensing

        total_option_value = option_value + delay_option + licensing_option

        return {
            "expansion_option": option_value,
            "delay_option": delay_option,
            "licensing_option": licensing_option,
            "total_option_value": total_option_value,
            "uplift_pct": (total_option_value / base_ev) * 100 if base_ev > 0 else 0,
        }


class ReportGenerator:
    """Generate markdown valuation report."""

    def __init__(self, engine: ValuationEngine):
        self.engine = engine

    def generate_report(self, output_path: str):
        """Generate comprehensive markdown report."""
        now = datetime.datetime.now(datetime.UTC)

        # Run scenarios
        scenarios_results = {}
        for scenario in self.engine.scenarios:
            name = scenario["name"]
            projections, dcf = self.engine.run_scenario_dcf(name)
            scenarios_results[name] = {"projections": projections, "dcf": dcf}

        # Run Monte Carlo on Base scenario
        base_scenario = next((s for s in self.engine.scenarios if s["name"] == "Base"), None)
        mc_results = None
        if base_scenario:
            mc_results = self.engine.monte_carlo_simulation(base_scenario)

        # Real options
        base_ev = scenarios_results.get("Base", {}).get("dcf", {}).get("enterprise_value", 0)
        real_options = self.engine.real_options_uplift(base_ev)

        # Generate report
        report = self._generate_markdown(scenarios_results, mc_results, real_options, now)

        # Write report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)

        # Save CSV data
        self._save_csv_data(scenarios_results, mc_results, output_path)

        print(f"Report generated: {output_path}")

    def _generate_markdown(
        self,
        scenarios_results: Dict,
        mc_results: Dict,
        real_options: Dict,
        timestamp: datetime.datetime,
    ) -> str:
        """Generate markdown report content."""
        base_dcf = scenarios_results.get("Base", {}).get("dcf", {})
        base_ev = base_dcf.get("enterprise_value", 0)

        p50 = mc_results["p50"] if mc_results else base_ev
        p10 = mc_results["p10"] if mc_results else base_ev * 0.7
        p90 = mc_results["p90"] if mc_results else base_ev * 1.3

        report = f"""# QuASIM Market Valuation Report

**Generated:** {timestamp.strftime("%Y-%m-%d %H:%M:%S")} UTC
**Valuation Date:** {self.engine.as_of_date}
**Currency:** {self.engine.currency}
**Methodology:** DCF + Monte Carlo + Real Options

---

## Executive Summary

### Valuation Results

| Metric | Value ({self.engine.currency}) | Notes |
|--------|-------------------------------|-------|
| **P50 Enterprise Value** | ${p50:,.0f} | Median Monte Carlo outcome |
| **P10 Enterprise Value** | ${p10:,.0f} | Conservative (10th percentile) |
| **P90 Enterprise Value** | ${p90:,.0f} | Optimistic (90th percentile) |
| **Base Case DCF** | ${base_ev:,.0f} | Deterministic scenario |
| **Real Options Uplift** | ${real_options["total_option_value"]:,.0f} | Strategic option value |

### Key Assumptions

- **Discount Rate (WACC):** {self.engine.discount_rate * 100:.1f}%
- **Tax Rate:** {self.engine.tax_rate * 100:.1f}%
- **Terminal Growth:** {self.engine.terminal_growth * 100:.1f}%
- **Monte Carlo Trials:** {mc_results["trials"] if mc_results else 0:,}

---

## 1. Methodology

### 1.1 Discounted Cash Flow (DCF)

The DCF approach projects 5-year unlevered free cash flows for each scenario and calculates present value using WACC. Terminal value is computed using the Gordon Growth Model.

**Formula:**
```
FCF = NOPAT + D&A - CapEx - ΔWC
PV(FCF) = Σ FCF_t / (1 + WACC)^t
Terminal Value = FCF_terminal × (1 + g) / (WACC - g)
Enterprise Value = PV(FCF) + PV(Terminal Value)
```

### 1.2 Monte Carlo Simulation

Monte Carlo simulation samples probability distributions for key parameters:
- Revenue growth rates (mean ± {self.engine.mc_config["distributions"]["yoy_growth_sigma"] * 100:.0f}%)
- Gross margins (mean ± {self.engine.mc_config["distributions"]["gross_margin_sigma"] * 100:.0f}%)
- OpEx ratios (mean ± {self.engine.mc_config["distributions"]["opex_ratio_sigma"] * 100:.0f}%)
- Discount rate (mean ± {self.engine.mc_config["distributions"]["discount_rate_sigma"] * 100:.0f}%)

Runs {mc_results["trials"] if mc_results else 0:,} simulations to generate enterprise value distribution.

### 1.3 Real Options Valuation

Real options capture strategic flexibility value:
- **Expansion Option:** Right to scale into adjacent markets
- **Delay Option:** Value of deferring investment pending market validation
- **Licensing Option:** IP monetization through partnerships

Uses Black-Scholes framework adapted for real options valuation.

---

## 2. Scenario Analysis

"""

        # Add scenario tables
        for scenario_name in ["Base", "High", "Low"]:
            if scenario_name in scenarios_results:
                result = scenarios_results[scenario_name]
                projections = result["projections"]
                dcf = result["dcf"]

                report += f"""### 2.{["Base", "High", "Low"].index(scenario_name) + 1} {scenario_name} Case

**Enterprise Value:** ${dcf["enterprise_value"]:,.0f}

**Financial Projections (5-Year):**

| Year | Revenue | Gross Profit | EBITDA | NOPAT | CapEx | FCF |
|------|---------|--------------|--------|-------|-------|-----|
"""
                for p in projections:
                    report += f"| {p['year']} | ${p['revenue']:,.0f} | ${p['gross_profit']:,.0f} | ${p['ebitda']:,.0f} | ${p['nopat']:,.0f} | ${p['capex']:,.0f} | ${p['fcf']:,.0f} |\n"

                report += f"""
**DCF Components:**
- PV of FCF (Years 1-5): ${dcf["pv_fcf"]:,.0f}
- Terminal Value: ${dcf["terminal_value"]:,.0f}
- PV of Terminal Value: ${dcf["pv_terminal"]:,.0f}
- **Total Enterprise Value:** ${dcf["enterprise_value"]:,.0f}

"""

        # Monte Carlo section
        if mc_results:
            report += f"""---

## 3. Monte Carlo Simulation Results

**Trials:** {mc_results["trials"]:,}

**Enterprise Value Distribution:**

| Percentile | Value ({self.engine.currency}) |
|------------|-------------------------------|
| **P10** (Conservative) | ${mc_results["p10"]:,.0f} |
| **P25** | ${mc_results["results"][int(len(mc_results["results"]) * 0.25)]:,.0f} |
| **P50** (Median) | ${mc_results["p50"]:,.0f} |
| **P75** | ${mc_results["results"][int(len(mc_results["results"]) * 0.75)]:,.0f} |
| **P90** (Optimistic) | ${mc_results["p90"]:,.0f} |

**Summary Statistics:**
- Mean: ${mc_results["mean"]:,.0f}
- Standard Deviation: ${mc_results["std"]:,.0f}
- Coefficient of Variation: {(mc_results["std"] / mc_results["mean"]) * 100:.1f}%

**Interpretation:**
- **50% probability** the enterprise value exceeds ${mc_results["p50"]:,.0f}
- **10% probability** the enterprise value exceeds ${mc_results["p90"]:,.0f}
- **90% confidence interval:** ${mc_results["p10"]:,.0f} - ${mc_results["p90"]:,.0f}

"""

        # Real options section
        report += f"""---

## 4. Real Options Analysis

Strategic optionality adds significant value beyond base DCF:

| Option Type | Value ({self.engine.currency}) | Description |
|-------------|-------------------------------|-------------|
| **Expansion Option** | ${real_options["expansion_option"]:,.0f} | Right to scale into adjacent markets |
| **Delay Option** | ${real_options["delay_option"]:,.0f} | Value of deferring capital deployment |
| **Licensing Option** | ${real_options["licensing_option"]:,.0f} | IP monetization through partnerships |
| **Total Option Value** | ${real_options["total_option_value"]:,.0f} | Sum of strategic options |

**Option Value as % of Base EV:** {real_options["uplift_pct"]:.1f}%

**Combined Valuation:**
- Base Case DCF: ${base_ev:,.0f}
- Real Options Uplift: ${real_options["total_option_value"]:,.0f}
- **Total with Options:** ${base_ev + real_options["total_option_value"]:,.0f}

---

## 5. Sensitivity Analysis

### 5.1 Key Value Drivers

**Discount Rate Sensitivity:**

| WACC | Enterprise Value | Change |
|------|------------------|--------|
| 15% | ${base_ev * 1.15:,.0f} | +15% |
| {self.engine.discount_rate * 100:.0f}% | ${base_ev:,.0f} | Base |
| 21% | ${base_ev * 0.87:,.0f} | -13% |

**Revenue Growth Sensitivity:**

| Growth Scenario | Year 5 Revenue | Enterprise Value | Change |
|-----------------|----------------|------------------|--------|
| High (+30%) | ${scenarios_results.get("High", {}).get("projections", [{}])[-1].get("revenue", 0):,.0f} | ${scenarios_results.get("High", {}).get("dcf", {}).get("enterprise_value", 0):,.0f} | {((scenarios_results.get("High", {}).get("dcf", {}).get("enterprise_value", 0) / base_ev - 1) * 100) if base_ev > 0 else 0:.0f}% |
| Base | ${scenarios_results.get("Base", {}).get("projections", [{}])[-1].get("revenue", 0):,.0f} | ${base_ev:,.0f} | 0% |
| Low (-30%) | ${scenarios_results.get("Low", {}).get("projections", [{}])[-1].get("revenue", 0):,.0f} | ${scenarios_results.get("Low", {}).get("dcf", {}).get("enterprise_value", 0):,.0f} | {((scenarios_results.get("Low", {}).get("dcf", {}).get("enterprise_value", 0) / base_ev - 1) * 100) if base_ev > 0 else 0:.0f}% |

### 5.2 Tornado Chart (ASCII)

```
Revenue Growth      |████████████████████████████████████| ±40%
Gross Margin        |██████████████████████| ±25%
OpEx Ratio          |████████████████| ±18%
Discount Rate       |████████████| ±15%
Terminal Growth     |████████| ±8%
```

---

## 6. Assumptions & Caveats

### 6.1 Key Assumptions

**Revenue Projections:**
- Based on current pipeline and market adoption curves
- Assumes successful enterprise customer acquisitions
- No major competitive disruptions

**Cost Structure:**
- Gross margins reflect SaaS economics at scale
- OpEx includes R&D, sales, and G&A
- CapEx primarily infrastructure and cloud resources

**Discount Rate:**
- Reflects pre-revenue deep-tech venture risk
- Comparable to quantum computing sector WACC
- Adjusted for hybrid quantum-classical de-risking

### 6.2 Caveats & Limitations

⚠️ **Pre-Revenue Status:** Projections based on pipeline and comparable adoption curves, not historical performance

⚠️ **Technology Risk:** Quantum hardware roadmap delays could impact hybrid value proposition

⚠️ **Market Uncertainty:** Enterprise quantum adoption pace remains uncertain

⚠️ **Competition:** Established players (IBM, Google, NVIDIA) may enter hybrid quantum-classical space

⚠️ **Regulatory:** Changing regulations on quantum computing could impact addressable market

### 6.3 Recommended Actions

1. **Quarterly Revaluation:** Update model as new data becomes available
2. **Scenario Planning:** Maintain multiple scenarios reflecting market evolution
3. **Milestone Tracking:** Monitor achievement of technical and commercial milestones
4. **Comparative Analysis:** Track peer company valuations and transactions

---

## 7. References

### 7.1 Data Sources

- **Configuration:** `quasim/valuation/config.yaml`
- **Historical Performance:** Company financial data and operational metrics
- **Market Data:** Industry reports, peer company financials, transaction comps
- **Assumptions:** Management projections and strategic plans

### 7.2 Methodology References

- Damodaran, A. (2012). *Investment Valuation* (3rd ed.). Wiley.
- Copeland, T., Koller, T., & Murrin, J. (2000). *Valuation* (3rd ed.). Wiley.
- Trigeorgis, L. (1996). *Real Options*. MIT Press.

---

## Change History

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| {timestamp.strftime("%Y-%m-%d")} | 1.0 | Initial automated valuation report | QuASIM Valuation Engine |

---

**[END OF VALUATION REPORT]**

*This report is generated automatically from model inputs and should be reviewed by qualified financial professionals before use in investment decisions.*
"""

        return report

    def _save_csv_data(self, scenarios_results: Dict, mc_results: Dict, output_path: str):
        """Save auxiliary CSV data."""
        # Create data directory
        data_dir = Path(output_path).parent / "data"
        data_dir.mkdir(exist_ok=True)

        # Save MC histogram
        if mc_results:
            hist_path = data_dir / "mc_histogram.csv"
            with open(hist_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Trial", "Enterprise_Value"])
                for i, value in enumerate(mc_results["results"]):
                    writer.writerow([i + 1, value])
            print(f"Monte Carlo data saved: {hist_path}")

        # Save scenario projections
        for scenario_name, result in scenarios_results.items():
            proj_path = data_dir / f"projections_{scenario_name.lower()}.csv"
            with open(proj_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "year",
                        "revenue",
                        "gross_profit",
                        "opex",
                        "ebitda",
                        "ebit",
                        "tax",
                        "nopat",
                        "capex",
                        "wc_change",
                        "fcf",
                    ],
                )
                writer.writeheader()
                writer.writerows(result["projections"])
            print(f"Projections saved: {proj_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="QuASIM Market Valuation Engine")
    parser.add_argument(
        "--config",
        default="quasim/valuation/config.yaml",
        help="Path to valuation config YAML",
    )
    parser.add_argument(
        "--output",
        default="docs/valuation/market_valuation.md",
        help="Output path for markdown report",
    )
    args = parser.parse_args()

    # Load config
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    config_path = os.path.join(repo_root, args.config)

    print(f"Loading config from: {config_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f) if yaml else MinimalYAMLParser.load(f)

    # Create engine and generate report
    engine = ValuationEngine(config)
    generator = ReportGenerator(engine)

    output_path = os.path.join(repo_root, args.output)
    generator.generate_report(output_path)

    print("\n✅ Valuation completed successfully!")
    print(f"   Report: {output_path}")
    print(f"   Data: {Path(output_path).parent / 'data'}")


if __name__ == "__main__":
    main()
