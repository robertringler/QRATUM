"""Tests for Fortune 500 QuASIM Integration Analysis modules."""

import csv
import json
import sys
import tempfile
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.fortune500_quasim_integration import (
    CompanyProfile,
    QIIComponents,
    aggregate_sector_analysis,
    analyze_company,
    calculate_economic_leverage,
    calculate_integration_compatibility,
    calculate_qii,
    calculate_strategic_value,
    calculate_technical_feasibility,
    enrich_company_data,
    generate_synthetic_fortune500,
)


def test_company_profile_creation():
    """Test CompanyProfile dataclass creation."""
    company = CompanyProfile(
        rank=1,
        name="Test Company",
        sector="Technology",
        industry="Software",
        revenue=100000.0,
        profit=10000.0,
        headquarters="San Francisco, CA",
    )
    assert company.rank == 1
    assert company.name == "Test Company"
    assert company.sector == "Technology"
    assert company.revenue == 100000.0


def test_enrich_company_data_technology():
    """Test enrichment for technology sector company."""
    company = CompanyProfile(
        rank=1,
        name="Amazon",
        sector="Technology",
        industry="Cloud Services",
        revenue=500000.0,
        profit=50000.0,
        headquarters="Seattle, WA",
    )

    enriched = enrich_company_data(company)

    assert enriched.cloud_provider is not None
    assert enriched.has_hpc is True
    assert enriched.has_ai_ml is True
    assert enriched.rnd_spending > 0
    assert enriched.rnd_percent_revenue > 0


def test_enrich_company_data_pharma():
    """Test enrichment for pharmaceutical company."""
    company = CompanyProfile(
        rank=50,
        name="Pfizer",
        sector="Pharmaceuticals",
        industry="Drug Development",
        revenue=80000.0,
        profit=15000.0,
        headquarters="New York, NY",
    )

    enriched = enrich_company_data(company)

    assert enriched.has_hpc is True
    assert enriched.rnd_percent_revenue >= 15  # Pharma has high R&D
    assert enriched.rnd_spending > 0


def test_calculate_technical_feasibility():
    """Test technical feasibility score calculation."""
    company = CompanyProfile(
        rank=1,
        name="Test",
        sector="Technology",
        industry="Software",
        revenue=100000.0,
        profit=10000.0,
        headquarters="SF",
        cloud_provider="AWS",
        has_hpc=True,
        has_ai_ml=True,
        has_quantum=True,
    )

    score = calculate_technical_feasibility(company)

    assert 0.0 <= score <= 1.0
    assert score >= 0.75  # Should be high with all flags set


def test_calculate_integration_compatibility():
    """Test integration compatibility score calculation."""
    company = CompanyProfile(
        rank=1,
        name="Test",
        sector="Technology",
        industry="Software",
        revenue=100000.0,
        profit=10000.0,
        headquarters="SF",
        cloud_provider="AWS",
        has_digital_twin=True,
        has_predictive_analytics=True,
    )

    score = calculate_integration_compatibility(company)

    assert 0.0 <= score <= 1.0
    assert score >= 0.65  # High compatibility expected


def test_calculate_economic_leverage():
    """Test economic leverage score calculation."""
    # High revenue, high R&D company
    company = CompanyProfile(
        rank=10,
        name="Test",
        sector="Technology",
        industry="Software",
        revenue=150000.0,  # > 100B
        profit=30000.0,  # 20% margin
        headquarters="SF",
        rnd_spending=20000.0,
        rnd_percent_revenue=13.33,  # High R&D
    )

    score = calculate_economic_leverage(company)

    assert 0.0 <= score <= 1.0
    assert score >= 0.70  # Should be high


def test_calculate_strategic_value():
    """Test strategic value score calculation."""
    # Pharmaceutical company with quantum
    company = CompanyProfile(
        rank=20,
        name="Test",
        sector="Pharmaceuticals",
        industry="Biotech",
        revenue=100000.0,
        profit=15000.0,
        headquarters="SF",
        has_quantum=True,
        has_digital_twin=True,
        has_ai_ml=True,
        rnd_percent_revenue=18.0,
    )

    score = calculate_strategic_value(company)

    assert 0.0 <= score <= 1.0
    assert score >= 0.65  # High strategic value for pharma


def test_calculate_qii():
    """Test complete QII calculation."""
    company = CompanyProfile(
        rank=1,
        name="Test Company",
        sector="Technology",
        industry="Software",
        revenue=150000.0,
        profit=30000.0,
        headquarters="SF",
        cloud_provider="AWS",
        has_hpc=True,
        has_ai_ml=True,
        has_quantum=True,
        has_digital_twin=True,
        has_predictive_analytics=True,
        rnd_spending=22500.0,
        rnd_percent_revenue=15.0,
    )

    components, qii_score = calculate_qii(company)

    # Check components
    assert isinstance(components, QIIComponents)
    assert 0.0 <= components.technical_feasibility <= 1.0
    assert 0.0 <= components.integration_compatibility <= 1.0
    assert 0.0 <= components.economic_leverage <= 1.0
    assert 0.0 <= components.strategic_value <= 1.0

    # Check QII score
    assert 0.0 <= qii_score <= 1.0

    # Verify weighted average
    expected_qii = (
        0.25 * components.technical_feasibility
        + 0.25 * components.integration_compatibility
        + 0.25 * components.economic_leverage
        + 0.25 * components.strategic_value
    )
    assert abs(qii_score - expected_qii) < 0.0001


def test_analyze_company():
    """Test complete company analysis."""
    company = CompanyProfile(
        rank=1,
        name="Test Company",
        sector="Technology",
        industry="Software",
        revenue=150000.0,
        profit=30000.0,
        headquarters="SF",
    )

    analysis = analyze_company(company)

    assert analysis.profile.name == "Test Company"
    assert 0.0 <= analysis.qii_score <= 1.0
    assert len(analysis.integration_pathways) > 0
    assert analysis.adoption_timeline is not None
    assert len(analysis.notes) > 0


def test_aggregate_sector_analysis():
    """Test sector aggregation."""
    # Create multiple companies in same sector
    companies = [
        CompanyProfile(
            rank=i,
            name=f"Company_{i}",
            sector="Technology",
            industry="Software",
            revenue=100000.0,
            profit=10000.0,
            headquarters="SF",
        )
        for i in range(1, 11)
    ]

    analyses = [analyze_company(c) for c in companies]
    sector_analysis = aggregate_sector_analysis(analyses, "Technology")

    assert sector_analysis.sector_name == "Technology"
    assert sector_analysis.company_count == 10
    assert 0.0 <= sector_analysis.mean_qii <= 1.0
    assert sector_analysis.std_qii >= 0.0
    assert len(sector_analysis.top_companies) <= 5
    assert sector_analysis.integration_potential in [
        "High",
        "Medium-High",
        "Medium",
        "Low-Medium",
    ]


def test_generate_synthetic_fortune500():
    """Test synthetic data generation."""
    companies = generate_synthetic_fortune500()

    assert len(companies) == 500
    assert all(1 <= c.rank <= 500 for c in companies)
    assert all(c.revenue > 0 for c in companies)

    # Check that rank 1 has highest revenue
    rank_1 = next(c for c in companies if c.rank == 1)
    rank_500 = next(c for c in companies if c.rank == 500)
    assert rank_1.revenue > rank_500.revenue


def test_qii_components_dataclass():
    """Test QIIComponents dataclass."""
    components = QIIComponents(
        technical_feasibility=0.8,
        integration_compatibility=0.7,
        economic_leverage=0.9,
        strategic_value=0.85,
    )

    assert components.technical_feasibility == 0.8
    assert components.integration_compatibility == 0.7
    assert components.economic_leverage == 0.9
    assert components.strategic_value == 0.85


def test_data_matrix_export():
    """Test CSV data matrix export."""
    from analysis.fortune500_quasim_integration import export_data_matrix

    # Create sample analyses
    companies = [
        CompanyProfile(
            rank=i,
            name=f"Company_{i}",
            sector="Technology",
            industry="Software",
            revenue=100000.0 - i * 100,
            profit=10000.0 - i * 10,
            headquarters="SF",
        )
        for i in range(1, 6)
    ]

    analyses = [analyze_company(c) for c in companies]

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        temp_path = Path(f.name)

    try:
        export_data_matrix(analyses, temp_path)

        # Read back and verify
        with open(temp_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 5
        assert "Rank" in rows[0]
        assert "Company" in rows[0]
        assert "QII_Score" in rows[0]
        assert rows[0]["Rank"] == "1"
        assert "Company_1" in rows[0]["Company"]
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_json_export():
    """Test JSON summary export."""
    from analysis.fortune500_quasim_integration import export_json_summary

    companies = generate_synthetic_fortune500()[:50]  # Use subset
    analyses = [analyze_company(c) for c in companies]

    # Aggregate by sector
    sectors = {}
    for sector_name in {a.profile.sector for a in analyses}:
        sector_analyses = [a for a in analyses if a.profile.sector == sector_name]
        sectors[sector_name] = aggregate_sector_analysis(sector_analyses, sector_name)

    # Calculate correlation
    from analysis.fortune500_quasim_integration import calculate_correlation_matrix

    correlation_data = calculate_correlation_matrix(analyses)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        temp_path = Path(f.name)

    try:
        export_json_summary(analyses, sectors, correlation_data, temp_path)

        # Read back and verify
        with open(temp_path) as f:
            data = json.load(f)

        assert "metadata" in data
        assert "top_20_companies" in data
        assert "sector_summaries" in data
        assert "correlation_analysis" in data
        assert "overall_statistics" in data

        assert data["metadata"]["total_companies"] == 50
        assert len(data["top_20_companies"]) == 20
        assert "mean_qii" in data["overall_statistics"]
    finally:
        if temp_path.exists():
            temp_path.unlink()


def test_integration_pathways():
    """Test integration pathway identification."""
    from analysis.fortune500_quasim_integration import identify_integration_pathways

    # High compatibility company
    company = CompanyProfile(
        rank=1,
        name="Test",
        sector="Technology",
        industry="Software",
        revenue=150000.0,
        profit=30000.0,
        headquarters="SF",
        cloud_provider="AWS",
        has_hpc=True,
        has_ai_ml=True,
        has_quantum=True,
        has_digital_twin=True,
    )

    components = QIIComponents(
        technical_feasibility=0.85,
        integration_compatibility=0.75,
        economic_leverage=0.90,
        strategic_value=0.80,
    )

    pathways = identify_integration_pathways(company, components)

    assert len(pathways) > 0
    assert any("API" in p for p in pathways)
    assert any("Cloud" in p for p in pathways)


def test_adoption_timeline():
    """Test adoption timeline determination."""
    from analysis.fortune500_quasim_integration import determine_adoption_timeline

    assert determine_adoption_timeline(0.85) == "2025-2026"
    assert determine_adoption_timeline(0.65) == "2026-2027"
    assert determine_adoption_timeline(0.50) == "2027-2028"
    assert determine_adoption_timeline(0.35) == "2028-2029"
    assert determine_adoption_timeline(0.20) == "2029-2030"


def test_qii_score_bounds():
    """Test that QII scores are always within valid bounds."""
    companies = generate_synthetic_fortune500()

    for company in companies[:100]:  # Test subset
        enriched = enrich_company_data(company)
        components, qii_score = calculate_qii(enriched)

        assert 0.0 <= qii_score <= 1.0
        assert 0.0 <= components.technical_feasibility <= 1.0
        assert 0.0 <= components.integration_compatibility <= 1.0
        assert 0.0 <= components.economic_leverage <= 1.0
        assert 0.0 <= components.strategic_value <= 1.0


def test_sector_diversity():
    """Test that generated data includes diverse sectors."""
    companies = generate_synthetic_fortune500()
    sectors = {c.sector for c in companies}

    assert len(sectors) >= 10  # At least 10 different sectors
    assert "Technology" in sectors
    assert "Financial Services" in sectors
    assert "Healthcare" in sectors or "Pharmaceuticals" in sectors


if __name__ == "__main__":
    # Run tests
    import traceback

    test_functions = [
        test_company_profile_creation,
        test_enrich_company_data_technology,
        test_enrich_company_data_pharma,
        test_calculate_technical_feasibility,
        test_calculate_integration_compatibility,
        test_calculate_economic_leverage,
        test_calculate_strategic_value,
        test_calculate_qii,
        test_analyze_company,
        test_aggregate_sector_analysis,
        test_generate_synthetic_fortune500,
        test_qii_components_dataclass,
        test_data_matrix_export,
        test_json_export,
        test_integration_pathways,
        test_adoption_timeline,
        test_qii_score_bounds,
        test_sector_diversity,
    ]

    passed = 0
    failed = 0

    for test_func in test_functions:
        try:
            test_func()
            print(f"✓ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{passed} passed, {failed} failed out of {len(test_functions)} tests")
    sys.exit(0 if failed == 0 else 1)
