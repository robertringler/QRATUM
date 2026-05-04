"""
Tests for SEER-Medicare claims parsing functionality.

Uses synthetic test data that mimics SEER-Medicare formats
without containing any actual patient data.
"""

from __future__ import annotations

import csv
from datetime import date
from pathlib import Path

import pytest

from qratum.oncology.registry.seer_medicare.medicare_claims import (
    MEDPARParser,
    NCHParser,
    OUTSAFParser,
    PDEParser,
    get_parser_for_file,
    parse_claims_directory,
)
from qratum.oncology.registry.seer_medicare.privacy import PrivacyConfig
from qratum.oncology.registry.seer_medicare.schema import ClaimSetting, CodeSystem
from qratum.oncology.registry.seer_medicare.seer_registry import (
    SEERRegistryParser,
    filter_cases_by_site,
    filter_cases_by_year,
)


class TestSEERRegistryParser:
    """Tests for SEER registry parsing."""

    @pytest.fixture
    def synthetic_seer_file(self, tmp_path: Path) -> Path:
        """Create a synthetic SEER registry file."""
        filepath = tmp_path / "seer_cases.csv"
        rows = [
            {
                "PATIENT_ID": "TEST001",
                "DATE_OF_DIAGNOSIS": "20150315",
                "PRIMARY_SITE": "LUNG",
                "HISTOLOGY": "ADENOCARCINOMA",
                "STAGE": "IV",
                "AGE": "72",
                "SEX": "M",
                "RACE": "01",
                "VITAL_STATUS": "1",
                "SURVIVAL": "24",
                "SEQUENCE_NUMBER": "0",
            },
            {
                "PATIENT_ID": "TEST002",
                "DATE_OF_DIAGNOSIS": "20160820",
                "PRIMARY_SITE": "BREAST",
                "HISTOLOGY": "DUCTAL",
                "STAGE": "IIA",
                "AGE": "68",
                "SEX": "F",
                "RACE": "01",
                "VITAL_STATUS": "1",
                "SURVIVAL": "36",
                "SEQUENCE_NUMBER": "0",
            },
            {
                "PATIENT_ID": "TEST003",
                "DATE_OF_DIAGNOSIS": "20140101",
                "PRIMARY_SITE": "COLON",
                "HISTOLOGY": "ADENOCARCINOMA",
                "STAGE": "III",
                "AGE": "75",
                "SEX": "M",
                "RACE": "02",
                "VITAL_STATUS": "4",
                "SURVIVAL": "18",
                "SEQUENCE_NUMBER": "0",
            },
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        return filepath

    def test_parse_seer_file(self, synthetic_seer_file: Path) -> None:
        """Test parsing a SEER registry file."""
        parser = SEERRegistryParser(privacy_config=PrivacyConfig(salt="test"))
        cases = list(parser.parse_file(synthetic_seer_file))

        assert len(cases) == 3

        # Check first case
        case1 = cases[0]
        assert case1.patient_key  # Should be hashed
        assert case1.dx_date == date(2015, 3, 15)
        assert case1.cancer_site == "LUNG"
        assert case1.histology == "ADENOCARCINOMA"
        assert case1.stage == "IV"
        assert case1.age_at_dx == 72
        assert case1.sex == "M"

    def test_patient_key_is_hashed(self, synthetic_seer_file: Path) -> None:
        """Test that patient keys are hashed, not raw IDs."""
        parser = SEERRegistryParser(privacy_config=PrivacyConfig(salt="test"))
        cases = list(parser.parse_file(synthetic_seer_file))

        # Patient key should not be the raw ID
        assert cases[0].patient_key != "TEST001"
        # Should be 16 characters (truncated hash)
        assert len(cases[0].patient_key) == 16
        # Should be hex characters
        assert all(c in "0123456789abcdef" for c in cases[0].patient_key)

    def test_deterministic_patient_key(self, synthetic_seer_file: Path) -> None:
        """Test that patient keys are deterministic."""
        parser1 = SEERRegistryParser(privacy_config=PrivacyConfig(salt="test"))
        parser2 = SEERRegistryParser(privacy_config=PrivacyConfig(salt="test"))

        cases1 = list(parser1.parse_file(synthetic_seer_file))
        cases2 = list(parser2.parse_file(synthetic_seer_file))

        assert cases1[0].patient_key == cases2[0].patient_key

    def test_different_salt_produces_different_keys(self, synthetic_seer_file: Path) -> None:
        """Test that different salts produce different keys."""
        parser1 = SEERRegistryParser(privacy_config=PrivacyConfig(salt="salt1"))
        parser2 = SEERRegistryParser(privacy_config=PrivacyConfig(salt="salt2"))

        cases1 = list(parser1.parse_file(synthetic_seer_file))
        cases2 = list(parser2.parse_file(synthetic_seer_file))

        assert cases1[0].patient_key != cases2[0].patient_key

    def test_filter_cases_by_site(self, synthetic_seer_file: Path) -> None:
        """Test filtering cases by cancer site."""
        parser = SEERRegistryParser()
        cases = parser.parse_file(synthetic_seer_file)
        filtered = list(filter_cases_by_site(cases, ["LUNG"]))

        assert len(filtered) == 1
        assert filtered[0].cancer_site == "LUNG"

    def test_filter_cases_by_year(self, synthetic_seer_file: Path) -> None:
        """Test filtering cases by diagnosis year."""
        parser = SEERRegistryParser()
        cases = parser.parse_file(synthetic_seer_file)
        filtered = list(filter_cases_by_year(cases, year_min=2015, year_max=2016))

        assert len(filtered) == 2


class TestMEDPARParser:
    """Tests for MEDPAR (inpatient) claims parsing."""

    @pytest.fixture
    def synthetic_medpar_file(self, tmp_path: Path) -> Path:
        """Create a synthetic MEDPAR file."""
        filepath = tmp_path / "medpar_claims.csv"
        rows = [
            {
                "BENE_ID": "TEST001",
                "CLM_ID": "CLM001",
                "ADMSN_DT": "20150401",
                "DSCHRG_DT": "20150405",
                "DRG_CD": "470",
                "PRNCPAL_DGNS_CD": "C3490",
                "ICD_DGNS_CD2": "J449",
                "SRGCL_PRCDR_CD1": "0DBN0ZZ",
                "CLM_PMT_AMT": "15000.00",
            },
            {
                "BENE_ID": "TEST001",
                "CLM_ID": "CLM002",
                "ADMSN_DT": "20150601",
                "DSCHRG_DT": "20150603",
                "DRG_CD": "180",
                "PRNCPAL_DGNS_CD": "C3490",
                "ICD_DGNS_CD2": "",
                "SRGCL_PRCDR_CD1": "",
                "CLM_PMT_AMT": "8000.00",
            },
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        return filepath

    def test_parse_medpar_file(self, synthetic_medpar_file: Path) -> None:
        """Test parsing a MEDPAR file."""
        parser = MEDPARParser(privacy_config=PrivacyConfig(salt="test"))
        events = list(parser.parse_file(synthetic_medpar_file))

        assert len(events) > 0
        # Should have DRG codes and diagnosis codes
        drg_events = [e for e in events if e.code_system == CodeSystem.DRG]
        dx_events = [
            e for e in events if e.code_system in (CodeSystem.ICD9_DX, CodeSystem.ICD10_DX)
        ]

        # We have 2 rows, each with a DRG
        assert len(drg_events) >= 2
        # Diagnosis codes: row 1 has J449 in ICD_DGNS_CD2, row 2 has empty
        assert len(dx_events) >= 1

    def test_medpar_setting_is_inpatient(self, synthetic_medpar_file: Path) -> None:
        """Test that MEDPAR events have inpatient setting."""
        parser = MEDPARParser()
        events = list(parser.parse_file(synthetic_medpar_file))

        for event in events:
            assert event.setting == ClaimSetting.INPATIENT


class TestNCHParser:
    """Tests for NCH (carrier) claims parsing."""

    @pytest.fixture
    def synthetic_nch_file(self, tmp_path: Path) -> Path:
        """Create a synthetic NCH file."""
        filepath = tmp_path / "nch_carrier.csv"
        rows = [
            {
                "BENE_ID": "TEST001",
                "CLM_ID": "NCHCLM001",
                "CLM_FROM_DT": "20150415",
                "HCPCS_CD": "96413",
                "CLM_PMT_AMT": "500.00",
                "PRVDR_SPCLTY": "83",  # Hematology/Oncology
            },
            {
                "BENE_ID": "TEST001",
                "CLM_ID": "NCHCLM002",
                "CLM_FROM_DT": "20150422",
                "HCPCS_CD": "J9271",
                "CLM_PMT_AMT": "8000.00",
                "PRVDR_SPCLTY": "83",
            },
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        return filepath

    def test_parse_nch_file(self, synthetic_nch_file: Path) -> None:
        """Test parsing an NCH file."""
        parser = NCHParser(privacy_config=PrivacyConfig(salt="test"))
        events = list(parser.parse_file(synthetic_nch_file))

        assert len(events) == 2
        assert all(e.setting == ClaimSetting.CARRIER for e in events)

    def test_nch_hcpcs_codes(self, synthetic_nch_file: Path) -> None:
        """Test that HCPCS codes are parsed correctly."""
        parser = NCHParser()
        events = list(parser.parse_file(synthetic_nch_file))

        codes = {e.code for e in events}
        assert "96413" in codes
        assert "J9271" in codes


class TestPDEParser:
    """Tests for PDE (Part D) parsing."""

    @pytest.fixture
    def synthetic_pde_file(self, tmp_path: Path) -> Path:
        """Create a synthetic PDE file."""
        filepath = tmp_path / "pde_partd.csv"
        rows = [
            {
                "BENE_ID": "TEST001",
                "PDE_ID": "PDE001",
                "SRVC_DT": "20150501",
                "PROD_SRVC_ID": "12345678901",  # NDC
                "QTY_DSPNSD_NUM": "30",
                "TOT_RX_CST_AMT": "150.00",
            },
            {
                "BENE_ID": "TEST001",
                "PDE_ID": "PDE002",
                "SRVC_DT": "20150601",
                "PROD_SRVC_ID": "98765432109",
                "QTY_DSPNSD_NUM": "60",
                "TOT_RX_CST_AMT": "300.00",
            },
        ]

        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        return filepath

    def test_parse_pde_file(self, synthetic_pde_file: Path) -> None:
        """Test parsing a PDE file."""
        parser = PDEParser(privacy_config=PrivacyConfig(salt="test"))
        events = list(parser.parse_file(synthetic_pde_file))

        assert len(events) == 2
        assert all(e.setting == ClaimSetting.PDE for e in events)
        assert all(e.code_system == CodeSystem.NDC for e in events)

    def test_pde_quantity(self, synthetic_pde_file: Path) -> None:
        """Test that quantity is parsed correctly."""
        parser = PDEParser()
        events = list(parser.parse_file(synthetic_pde_file))

        quantities = [e.quantity for e in events]
        assert 30 in quantities
        assert 60 in quantities


class TestParserSelection:
    """Tests for automatic parser selection."""

    def test_get_parser_for_medpar(self, tmp_path: Path) -> None:
        """Test parser selection for MEDPAR files."""
        filepath = tmp_path / "medpar_2015.csv"
        filepath.touch()

        parser = get_parser_for_file(filepath)
        assert isinstance(parser, MEDPARParser)

    def test_get_parser_for_nch(self, tmp_path: Path) -> None:
        """Test parser selection for NCH files."""
        filepath = tmp_path / "nch_carrier_claims.csv"
        filepath.touch()

        parser = get_parser_for_file(filepath)
        assert isinstance(parser, NCHParser)

    def test_get_parser_for_outsaf(self, tmp_path: Path) -> None:
        """Test parser selection for OUTSAF files."""
        filepath = tmp_path / "outsaf_outpatient.csv"
        filepath.touch()

        parser = get_parser_for_file(filepath)
        assert isinstance(parser, OUTSAFParser)

    def test_get_parser_for_pde(self, tmp_path: Path) -> None:
        """Test parser selection for PDE files."""
        filepath = tmp_path / "pde_partd_events.csv"
        filepath.touch()

        parser = get_parser_for_file(filepath)
        assert isinstance(parser, PDEParser)

    def test_get_parser_for_unknown(self, tmp_path: Path) -> None:
        """Test parser selection for unknown files."""
        filepath = tmp_path / "unknown_file.csv"
        filepath.touch()

        parser = get_parser_for_file(filepath)
        assert parser is None


class TestClaimsDirectoryParsing:
    """Tests for directory-level claims parsing."""

    @pytest.fixture
    def synthetic_claims_dir(self, tmp_path: Path) -> Path:
        """Create a directory with multiple claim files."""
        claims_dir = tmp_path / "claims"
        claims_dir.mkdir()

        # Create NCH file
        nch_file = claims_dir / "nch_claims.csv"
        nch_rows = [
            {
                "BENE_ID": "TEST001",
                "CLM_ID": "NCH001",
                "CLM_FROM_DT": "20150415",
                "HCPCS_CD": "96413",
                "CLM_PMT_AMT": "500.00",
                "PRVDR_SPCLTY": "83",
            },
        ]
        with open(nch_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=nch_rows[0].keys())
            writer.writeheader()
            writer.writerows(nch_rows)

        # Create PDE file
        pde_file = claims_dir / "pde_events.csv"
        pde_rows = [
            {
                "BENE_ID": "TEST001",
                "PDE_ID": "PDE001",
                "SRVC_DT": "20150501",
                "PROD_SRVC_ID": "12345678901",
                "QTY_DSPNSD_NUM": "30",
                "TOT_RX_CST_AMT": "150.00",
            },
        ]
        with open(pde_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=pde_rows[0].keys())
            writer.writeheader()
            writer.writerows(pde_rows)

        return claims_dir

    def test_parse_claims_directory(self, synthetic_claims_dir: Path) -> None:
        """Test parsing all claims in a directory."""
        events = list(parse_claims_directory(synthetic_claims_dir))

        assert len(events) >= 2
        settings = {e.setting for e in events}
        assert ClaimSetting.CARRIER in settings
        assert ClaimSetting.PDE in settings
