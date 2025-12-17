"""Fortune 500 QuASIM Integration Analysis.

This module implements a comprehensive analysis framework to evaluate QuASIM
(Quantum-Accelerated Simulation Infrastructure & Management) integration
opportunities across all Fortune 500 companies.
"""

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Directory paths
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"
VIS_DIR = Path(__file__).resolve().parents[1] / "visuals"


@dataclass
class CompanyProfile:
    """Profile data for a Fortune 500 company."""

    rank: int
    name: str
    sector: str
    industry: str
    revenue: float  # in millions USD
    profit: float  # in millions USD
    headquarters: str
    cloud_provider: Optional[str] = None
    has_hpc: bool = False
    has_ai_ml: bool = False
    has_quantum: bool = False
    has_digital_twin: bool = False
    has_predictive_analytics: bool = False
    rnd_spending: float = 0.0  # in millions USD
    rnd_percent_revenue: float = 0.0


@dataclass
class QIIComponents:
    """Components of the QuASIM Integration Index."""

    technical_feasibility: float  # T: 0-1
    integration_compatibility: float  # I: 0-1
    economic_leverage: float  # E: 0-1
    strategic_value: float  # S: 0-1


@dataclass
class CompanyAnalysis:
    """Complete QuASIM integration analysis for a company."""

    profile: CompanyProfile
    qii_components: QIIComponents
    qii_score: float
    integration_pathways: List[str] = field(default_factory=list)
    adoption_timeline: str = "2026-2028"
    notes: str = ""


@dataclass
class SectorAnalysis:
    """Aggregated analysis for a specific sector."""

    sector_name: str
    company_count: int
    mean_qii: float
    std_qii: float
    median_qii: float
    top_companies: List[str]
    integration_potential: str
    key_challenges: List[str]
    recommended_approach: str


def load_fortune500_data(filepath: Path) -> List[CompanyProfile]:
    """Load Fortune 500 data from CSV file.

    Args:
        filepath: Path to the CSV file containing Fortune 500 data

    Returns:
        List of CompanyProfile objects
    """

    companies = []

    with open(filepath, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            company = CompanyProfile(
                rank=int(row.get("Rank", 0)),
                name=row.get("Company", "Unknown"),
                sector=row.get("Sector", "Unknown"),
                industry=row.get("Industry", "Unknown"),
                revenue=float(row.get("Revenue", 0)),
                profit=float(row.get("Profit", 0)),
                headquarters=row.get("Headquarters", "Unknown"),
            )
            companies.append(company)

    return companies


def enrich_company_data(company: CompanyProfile) -> CompanyProfile:
    """Enrich company data with technological infrastructure information.

    This function identifies cloud providers, HPC, AI/ML, quantum initiatives
    based on industry patterns and known data points.

    Args:
        company: CompanyProfile to enrich

    Returns:
        Enriched CompanyProfile
    """

    # Technology-heavy sectors
    tech_sectors = [
        "Technology",
        "Telecommunications",
        "Electronics",
        "Semiconductors",
        "Internet Services",
    ]
    research_sectors = [
        "Pharmaceuticals",
        "Biotechnology",
        "Aerospace & Defense",
        "Chemicals",
    ]

    # Cloud provider assignment based on industry patterns
    if company.sector in tech_sectors:
        if "Amazon" in company.name or "AWS" in company.name:
            company.cloud_provider = "AWS"
        elif "Microsoft" in company.name or "Azure" in company.name:
            company.cloud_provider = "Azure"
        elif "Google" in company.name or "Alphabet" in company.name:
            company.cloud_provider = "GCP"
        else:
            company.cloud_provider = "Multi-cloud"
    elif company.sector in ["Financial Services", "Insurance", "Banking"]:
        company.cloud_provider = "Private/Hybrid"
    else:
        company.cloud_provider = "AWS"  # Default for most enterprises

    # HPC presence
    company.has_hpc = company.sector in tech_sectors + research_sectors

    # AI/ML initiatives
    company.has_ai_ml = (
        company.sector in tech_sectors
        or company.sector in research_sectors
        or company.sector in ["Automotive", "Manufacturing", "Energy"]
    )

    # Quantum initiatives (conservative estimate)
    quantum_companies = [
        "IBM",
        "Google",
        "Microsoft",
        "Amazon",
        "Intel",
        "Honeywell",
        "Lockheed Martin",
    ]
    company.has_quantum = any(name in company.name for name in quantum_companies)

    # Digital twin presence
    company.has_digital_twin = company.sector in [
        "Automotive",
        "Aerospace & Defense",
        "Manufacturing",
        "Energy",
    ]

    # Predictive analytics
    company.has_predictive_analytics = (
        company.sector in ["Financial Services", "Insurance", "Healthcare"] or company.has_ai_ml
    )

    # R&D spending estimation (based on industry averages)
    rnd_ratios = {
        "Pharmaceuticals": 0.18,
        "Biotechnology": 0.20,
        "Technology": 0.15,
        "Semiconductors": 0.16,
        "Aerospace & Defense": 0.04,
        "Automotive": 0.05,
        "Chemicals": 0.03,
        "default": 0.02,
    }

    ratio = rnd_ratios.get(company.sector, rnd_ratios["default"])
    company.rnd_spending = company.revenue * ratio
    company.rnd_percent_revenue = ratio * 100

    return company


def calculate_technical_feasibility(company: CompanyProfile) -> float:
    """Calculate technical feasibility score (T component of QII).

    Args:
        company: CompanyProfile with enriched data

    Returns:
        Score from 0.0 to 1.0
    """

    score = 0.0

    # HPC infrastructure presence
    if company.has_hpc:
        score += 0.30

    # AI/ML capabilities
    if company.has_ai_ml:
        score += 0.25

    # Cloud infrastructure maturity
    cloud_scores = {
        "AWS": 0.25,
        "Azure": 0.25,
        "GCP": 0.25,
        "Multi-cloud": 0.20,
        "Private/Hybrid": 0.15,
    }
    score += cloud_scores.get(company.cloud_provider or "Private/Hybrid", 0.10)

    # Existing quantum initiatives
    if company.has_quantum:
        score += 0.20

    return min(1.0, score)


def calculate_integration_compatibility(company: CompanyProfile) -> float:
    """Calculate integration compatibility score (I component of QII).

    Args:
        company: CompanyProfile with enriched data

    Returns:
        Score from 0.0 to 1.0
    """

    score = 0.0

    # Digital twin infrastructure
    if company.has_digital_twin:
        score += 0.35

    # Predictive analytics maturity
    if company.has_predictive_analytics:
        score += 0.30

    # Technology stack compatibility
    if company.cloud_provider in ["AWS", "Azure", "GCP"]:
        score += 0.25

    # Industry standardization
    standardized_sectors = [
        "Technology",
        "Automotive",
        "Aerospace & Defense",
        "Manufacturing",
    ]
    if company.sector in standardized_sectors:
        score += 0.10

    return min(1.0, score)


def calculate_economic_leverage(company: CompanyProfile) -> float:
    """Calculate economic leverage score (E component of QII).

    Args:
        company: CompanyProfile with enriched data

    Returns:
        Score from 0.0 to 1.0
    """

    score = 0.0

    # Revenue scale (larger companies have more resources)
    if company.revenue > 100000:  # > $100B
        score += 0.30
    elif company.revenue > 50000:  # > $50B
        score += 0.25
    elif company.revenue > 20000:  # > $20B
        score += 0.20
    else:
        score += 0.10

    # R&D investment
    if company.rnd_percent_revenue > 10:
        score += 0.30
    elif company.rnd_percent_revenue > 5:
        score += 0.25
    elif company.rnd_percent_revenue > 2:
        score += 0.15
    else:
        score += 0.05

    # Profitability (indicates financial capacity)
    profit_margin = company.profit / company.revenue if company.revenue > 0 else 0
    if profit_margin > 0.15:
        score += 0.20
    elif profit_margin > 0.10:
        score += 0.15
    elif profit_margin > 0.05:
        score += 0.10
    else:
        score += 0.05

    # Market position (rank-based)
    if company.rank <= 50:
        score += 0.20
    elif company.rank <= 100:
        score += 0.15
    elif company.rank <= 250:
        score += 0.10
    else:
        score += 0.05

    return min(1.0, score)


def calculate_strategic_value(company: CompanyProfile) -> float:
    """Calculate strategic value score (S component of QII).

    Args:
        company: CompanyProfile with enriched data

    Returns:
        Score from 0.0 to 1.0
    """

    score = 0.0

    # Strategic sectors for quantum/simulation
    high_value_sectors = [
        "Pharmaceuticals",
        "Biotechnology",
        "Aerospace & Defense",
        "Automotive",
        "Energy",
        "Chemicals",
    ]
    medium_value_sectors = [
        "Technology",
        "Manufacturing",
        "Financial Services",
        "Telecommunications",
    ]

    if company.sector in high_value_sectors:
        score += 0.40
    elif company.sector in medium_value_sectors:
        score += 0.25
    else:
        score += 0.10

    # Quantum computing readiness
    if company.has_quantum:
        score += 0.25

    # Digital transformation investment
    if company.has_digital_twin and company.has_ai_ml:
        score += 0.20
    elif company.has_digital_twin or company.has_ai_ml:
        score += 0.10

    # Innovation leadership
    if company.rank <= 100 and company.rnd_percent_revenue > 5:
        score += 0.15

    return min(1.0, score)


def calculate_qii(company: CompanyProfile) -> tuple[QIIComponents, float]:
    """Calculate the QuASIM Integration Index (QII) for a company.

    QII = 0.25*T + 0.25*I + 0.25*E + 0.25*S

    Args:
        company: CompanyProfile with enriched data

    Returns:
        Tuple of (QIIComponents, overall QII score)
    """

    components = QIIComponents(
        technical_feasibility=calculate_technical_feasibility(company),
        integration_compatibility=calculate_integration_compatibility(company),
        economic_leverage=calculate_economic_leverage(company),
        strategic_value=calculate_strategic_value(company),
    )

    qii_score = (
        0.25 * components.technical_feasibility
        + 0.25 * components.integration_compatibility
        + 0.25 * components.economic_leverage
        + 0.25 * components.strategic_value
    )

    return components, round(qii_score, 4)


def identify_integration_pathways(
    company: CompanyProfile, qii_components: QIIComponents
) -> List[str]:
    """Identify specific integration pathways for QuASIM.

    Args:
        company: CompanyProfile with enriched data
        qii_components: Calculated QII components

    Returns:
        List of recommended integration pathways
    """

    pathways = []

    # API-level integration
    if qii_components.integration_compatibility > 0.6:
        pathways.append("API-level: REST/GraphQL endpoints for QuASIM services")

    # Runtime-level integration
    if company.has_hpc and qii_components.technical_feasibility > 0.7:
        pathways.append("Runtime-level: Direct HPC cluster integration with QuASIM kernels")

    # Pipeline fusion
    if company.has_ai_ml and company.has_digital_twin:
        pathways.append("Pipeline fusion: Integrate QuASIM into existing ML/digital-twin workflows")

    # SDK integration
    if qii_components.technical_feasibility > 0.5:
        pathways.append("SDK integration: Python/C++ libraries for quantum simulation")

    # Cloud-native deployment
    if company.cloud_provider in ["AWS", "Azure", "GCP"]:
        pathways.append(f"Cloud-native: Deploy QuASIM containers on {company.cloud_provider}")

    # Custom quantum circuits
    if company.has_quantum:
        pathways.append("Custom circuits: Quantum algorithm development for specific use cases")

    return pathways if pathways else ["Evaluation: Start with pilot program and feasibility study"]


def determine_adoption_timeline(qii_score: float) -> str:
    """Determine adoption timeline based on QII score.

    Args:
        qii_score: Overall QII score (0-1)

    Returns:
        Timeline string (e.g., "2025-2026")
    """

    if qii_score >= 0.75:
        return "2025-2026"
    elif qii_score >= 0.60:
        return "2026-2027"
    elif qii_score >= 0.45:
        return "2027-2028"
    elif qii_score >= 0.30:
        return "2028-2029"
    else:
        return "2029-2030"


def analyze_company(company: CompanyProfile) -> CompanyAnalysis:
    """Perform complete QuASIM integration analysis for a company.

    Args:
        company: CompanyProfile to analyze

    Returns:
        CompanyAnalysis with complete evaluation
    """

    # Enrich data
    company = enrich_company_data(company)

    # Calculate QII
    qii_components, qii_score = calculate_qii(company)

    # Identify pathways
    pathways = identify_integration_pathways(company, qii_components)

    # Timeline
    timeline = determine_adoption_timeline(qii_score)

    # Generate notes
    notes = f"QII Score: {qii_score:.3f} | Sector: {company.sector} | R&D: {company.rnd_percent_revenue:.1f}%"

    return CompanyAnalysis(
        profile=company,
        qii_components=qii_components,
        qii_score=qii_score,
        integration_pathways=pathways,
        adoption_timeline=timeline,
        notes=notes,
    )


def aggregate_sector_analysis(analyses: List[CompanyAnalysis], sector_name: str) -> SectorAnalysis:
    """Aggregate analysis results for a specific sector.

    Args:
        analyses: List of CompanyAnalysis objects for the sector
        sector_name: Name of the sector

    Returns:
        SectorAnalysis with aggregated metrics
    """

    if not analyses:
        return SectorAnalysis(
            sector_name=sector_name,
            company_count=0,
            mean_qii=0.0,
            std_qii=0.0,
            median_qii=0.0,
            top_companies=[],
            integration_potential="Unknown",
            key_challenges=[],
            recommended_approach="",
        )

    qii_scores = [a.qii_score for a in analyses]
    mean_qii = float(np.mean(qii_scores))
    std_qii = float(np.std(qii_scores))
    median_qii = float(np.median(qii_scores))

    # Get top 5 companies by QII
    sorted_analyses = sorted(analyses, key=lambda x: x.qii_score, reverse=True)
    top_companies = [a.profile.name for a in sorted_analyses[:5]]

    # Determine integration potential
    if mean_qii >= 0.65:
        potential = "High"
    elif mean_qii >= 0.50:
        potential = "Medium-High"
    elif mean_qii >= 0.35:
        potential = "Medium"
    else:
        potential = "Low-Medium"

    # Identify key challenges (sector-specific)
    challenges = []
    if mean_qii < 0.50:
        challenges.append("Limited technical infrastructure")
    if std_qii > 0.15:
        challenges.append("High variability in readiness across companies")
    if np.mean([a.qii_components.integration_compatibility for a in analyses]) < 0.50:
        challenges.append("Integration complexity with legacy systems")

    # Recommended approach
    if mean_qii >= 0.60:
        approach = "Direct enterprise sales with pilot programs"
    elif mean_qii >= 0.45:
        approach = "Partnership model with phased rollout"
    else:
        approach = "Education and proof-of-concept demonstrations"

    return SectorAnalysis(
        sector_name=sector_name,
        company_count=len(analyses),
        mean_qii=round(mean_qii, 4),
        std_qii=round(std_qii, 4),
        median_qii=round(median_qii, 4),
        top_companies=top_companies,
        integration_potential=potential,
        key_challenges=challenges,
        recommended_approach=approach,
    )


def calculate_correlation_matrix(
    analyses: List[CompanyAnalysis],
) -> Dict[str, Dict[str, float]]:
    """Calculate correlation matrix between R&D spending % and QII.

    Args:
        analyses: List of all company analyses

    Returns:
        Dictionary containing correlation statistics
    """

    rnd_percents = [a.profile.rnd_percent_revenue for a in analyses]
    qii_scores = [a.qii_score for a in analyses]

    correlation = float(np.corrcoef(rnd_percents, qii_scores)[0, 1])

    return {
        "correlation_rnd_qii": round(correlation, 4),
        "mean_rnd_percent": round(float(np.mean(rnd_percents)), 4),
        "mean_qii": round(float(np.mean(qii_scores)), 4),
    }


def export_data_matrix(analyses: List[CompanyAnalysis], output_path: Path) -> None:
    """Export complete 500x15 data matrix to CSV.

    Args:
        analyses: List of all company analyses
        output_path: Path to output CSV file
    """

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Rank",
            "Company",
            "Sector",
            "Revenue_M",
            "Profit_M",
            "RnD_Percent",
            "Cloud_Provider",
            "Has_HPC",
            "Has_AI_ML",
            "Has_Quantum",
            "QII_Score",
            "Technical_Feasibility",
            "Integration_Compatibility",
            "Economic_Leverage",
            "Strategic_Value",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for analysis in analyses:
            writer.writerow(
                {
                    "Rank": analysis.profile.rank,
                    "Company": analysis.profile.name,
                    "Sector": analysis.profile.sector,
                    "Revenue_M": analysis.profile.revenue,
                    "Profit_M": analysis.profile.profit,
                    "RnD_Percent": round(analysis.profile.rnd_percent_revenue, 2),
                    "Cloud_Provider": analysis.profile.cloud_provider or "Unknown",
                    "Has_HPC": analysis.profile.has_hpc,
                    "Has_AI_ML": analysis.profile.has_ai_ml,
                    "Has_Quantum": analysis.profile.has_quantum,
                    "QII_Score": analysis.qii_score,
                    "Technical_Feasibility": analysis.qii_components.technical_feasibility,
                    "Integration_Compatibility": analysis.qii_components.integration_compatibility,
                    "Economic_Leverage": analysis.qii_components.economic_leverage,
                    "Strategic_Value": analysis.qii_components.strategic_value,
                }
            )


def export_json_summary(
    analyses: List[CompanyAnalysis],
    sector_analyses: Dict[str, SectorAnalysis],
    correlation_data: Dict[str, Dict[str, float]],
    output_path: Path,
) -> None:
    """Export comprehensive JSON summary of the analysis.

    Args:
        analyses: List of all company analyses
        sector_analyses: Dictionary of sector analyses
        correlation_data: Correlation statistics
        output_path: Path to output JSON file
    """

    # Top 20 companies by QII
    top_20 = sorted(analyses, key=lambda x: x.qii_score, reverse=True)[:20]

    summary = {
        "metadata": {
            "total_companies": len(analyses),
            "sectors_analyzed": len(sector_analyses),
            "analysis_date": "2025",
        },
        "top_20_companies": [
            {
                "rank": a.profile.rank,
                "name": a.profile.name,
                "sector": a.profile.sector,
                "qii_score": a.qii_score,
                "adoption_timeline": a.adoption_timeline,
            }
            for a in top_20
        ],
        "sector_summaries": {name: asdict(sector) for name, sector in sector_analyses.items()},
        "correlation_analysis": correlation_data,
        "overall_statistics": {
            "mean_qii": round(float(np.mean([a.qii_score for a in analyses])), 4),
            "median_qii": round(float(np.median([a.qii_score for a in analyses])), 4),
            "std_qii": round(float(np.std([a.qii_score for a in analyses])), 4),
            "min_qii": round(min(a.qii_score for a in analyses), 4),
            "max_qii": round(max(a.qii_score for a in analyses), 4),
        },
    }

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)


def main(input_csv: Optional[Path] = None) -> Dict[str, Path]:
    """Execute the complete Fortune 500 QuASIM Integration Analysis.

    Args:
        input_csv: Optional path to Fortune 500 CSV data file.
                   If None, generates synthetic data.

    Returns:
        Dictionary mapping output names to file paths
    """

    # Create output directories
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    if input_csv and input_csv.exists():
        companies = load_fortune500_data(input_csv)
    else:
        # Generate synthetic Fortune 500 data for demonstration
        companies = generate_synthetic_fortune500()

    # Analyze all companies
    print(f"Analyzing {len(companies)} companies...")
    analyses = [analyze_company(company) for company in companies]

    # Aggregate by sector
    print("Aggregating sector analyses...")
    sectors = {}
    for sector_name in {a.profile.sector for a in analyses}:
        sector_analyses = [a for a in analyses if a.profile.sector == sector_name]
        sectors[sector_name] = aggregate_sector_analysis(sector_analyses, sector_name)

    # Calculate correlations
    print("Computing correlation matrix...")
    correlation_data = calculate_correlation_matrix(analyses)

    # Export data matrix
    matrix_path = DATA_DIR / "fortune500_quasim_matrix.csv"
    print(f"Exporting data matrix to {matrix_path}...")
    export_data_matrix(analyses, matrix_path)

    # Export JSON summary
    json_path = DATA_DIR / "fortune500_quasim_analysis.json"
    print(f"Exporting JSON summary to {json_path}...")
    export_json_summary(analyses, sectors, correlation_data, json_path)

    print("Analysis complete!")

    return {
        "data_matrix": matrix_path,
        "json_summary": json_path,
    }


def generate_synthetic_fortune500() -> List[CompanyProfile]:
    """Generate synthetic Fortune 500 data for demonstration purposes.

    Returns:
        List of 500 CompanyProfile objects with realistic distributions
    """

    np.random.seed(42)

    sectors = [
        "Technology",
        "Financial Services",
        "Healthcare",
        "Pharmaceuticals",
        "Retail",
        "Energy",
        "Automotive",
        "Manufacturing",
        "Telecommunications",
        "Aerospace & Defense",
        "Chemicals",
        "Biotechnology",
        "Insurance",
        "Banking",
        "Consumer Goods",
    ]

    companies = []
    for rank in range(1, 501):
        # Revenue decreases with rank (log scale)
        base_revenue = 600000 * np.exp(-rank / 150)
        revenue = base_revenue * (0.8 + np.random.random() * 0.4)

        # Profit margin varies by rank
        profit_margin = 0.05 + 0.10 * np.exp(-rank / 100) + np.random.normal(0, 0.02)
        profit = revenue * max(0.01, profit_margin)

        sector = np.random.choice(sectors)
        industry = f"{sector} Subsector {np.random.randint(1, 5)}"

        company = CompanyProfile(
            rank=rank,
            name=f"Company_{rank:03d}",
            sector=sector,
            industry=industry,
            revenue=round(revenue, 2),
            profit=round(profit, 2),
            headquarters=f"City_{np.random.randint(1, 100)}",
        )
        companies.append(company)

    return companies


if __name__ == "__main__":
    output_files = main()
    print("\nGenerated files:")
    for name, path in output_files.items():
        print(f"  {name}: {path}")
