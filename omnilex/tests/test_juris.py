"""Tests for QRATUM-JURIS criminal case analysis engine.

Tests the Ultra-Detailed Legal Analysis & Public-Record Exhaustion Engine.
"""

import pytest

from omnilex.juris import (
    CompletenessRating,
    CriminalCaseIntent,
    PublicRecordSource,
    QRATUMJurisEngine,
    SourceAccessibility,
    create_ohio_v_ringler_intent,
    generate_juris_intent_id,
)


class TestCriminalCaseIntent:
    """Test CriminalCaseIntent dataclass."""

    def test_create_valid_intent(self):
        """Test creating a valid criminal case intent."""
        intent = CriminalCaseIntent(
            intent_id="JURIS-test123",
            case_number="17CR052",
            case_name="STATE OF OHIO v. RINGLER, ROBERT",
            jurisdiction="Holmes County Common Pleas",
            county="Holmes",
            state="Ohio",
            defendant_name="Robert Ringler",
            charges=(
                "R.C. 2903.13(A) — Assault, Felony 4",
                "R.C. 2921.33(B) — Resisting Arrest, M1",
            ),
            plea="Guilty",
            sentence="14 months",
            judge="Robert D. Rinfret",
            defense_counsel="Mark W. Baserman, Jr.",
            key_dates=(("plea", "08/28/2017"),),
        )

        assert intent.case_number == "17CR052"
        assert intent.defendant_name == "Robert Ringler"
        assert len(intent.charges) == 2
        assert intent.analysis_type == "full_forensic"

    def test_intent_is_frozen(self):
        """Test that intent is immutable (frozen)."""
        intent = create_ohio_v_ringler_intent()

        with pytest.raises(AttributeError):
            intent.case_number = "modified"

    def test_compute_hash_deterministic(self):
        """Test that hash computation is deterministic."""
        # Create two intents with same data (except timestamp-based ID)
        intent1 = CriminalCaseIntent(
            intent_id="JURIS-fixed",
            case_number="17CR052",
            case_name="STATE OF OHIO v. RINGLER",
            jurisdiction="Holmes County",
            county="Holmes",
            state="Ohio",
            defendant_name="Robert Ringler",
            charges=("Assault",),
            plea="Guilty",
            sentence="14 months",
            judge="Judge Smith",
            defense_counsel="Attorney Jones",
            key_dates=(("plea", "08/28/2017"),),
        )

        intent2 = CriminalCaseIntent(
            intent_id="JURIS-fixed",
            case_number="17CR052",
            case_name="STATE OF OHIO v. RINGLER",
            jurisdiction="Holmes County",
            county="Holmes",
            state="Ohio",
            defendant_name="Robert Ringler",
            charges=("Assault",),
            plea="Guilty",
            sentence="14 months",
            judge="Judge Smith",
            defense_counsel="Attorney Jones",
            key_dates=(("plea", "08/28/2017"),),
        )

        assert intent1.compute_hash() == intent2.compute_hash()


class TestGenerateJurisIntentId:
    """Test intent ID generation."""

    def test_format(self):
        """Test intent ID has correct format."""
        intent_id = generate_juris_intent_id("17CR052", 1234567890.0)

        assert intent_id.startswith("JURIS-")
        assert len(intent_id) == 22  # "JURIS-" + 16 hex chars

    def test_deterministic(self):
        """Test ID generation is deterministic."""
        id1 = generate_juris_intent_id("17CR052", 1234567890.0)
        id2 = generate_juris_intent_id("17CR052", 1234567890.0)

        assert id1 == id2

    def test_different_inputs(self):
        """Test different inputs produce different IDs."""
        id1 = generate_juris_intent_id("17CR052", 1234567890.0)
        id2 = generate_juris_intent_id("17CR053", 1234567890.0)

        assert id1 != id2


class TestQRATUMJurisEngine:
    """Test QRATUM-JURIS engine."""

    def test_initialization(self):
        """Test engine initialization."""
        engine = QRATUMJurisEngine()

        assert engine.VERSION == "1.0.0"
        assert "DOES NOT PROVIDE LEGAL ADVICE" in engine.CORE_DISCLAIMER

    def test_ohio_statutes_loaded(self):
        """Test Ohio statutes are pre-loaded."""
        engine = QRATUMJurisEngine()

        assert "2903.13(A)" in engine.OHIO_STATUTES
        assert "2921.33(B)" in engine.OHIO_STATUTES
        assert "2909.06(A)(1)" in engine.OHIO_STATUTES

    def test_speedy_trial_limits(self):
        """Test speedy trial limits are configured."""
        engine = QRATUMJurisEngine()

        assert engine.OHIO_SPEEDY_TRIAL["felony"] == 270
        assert engine.OHIO_SPEEDY_TRIAL["M1"] == 90

    def test_analyze_criminal_case(self):
        """Test full criminal case analysis."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)

        # Verify structure
        assert "intent_id" in result
        assert "intent_hash" in result
        assert "case_number" in result
        assert "phase_i_public_record_exhaustion" in result
        assert "phase_ii_legal_analysis" in result
        assert "disclaimer" in result
        assert "result_hash" in result

        # Verify Phase I
        phase_i = result["phase_i_public_record_exhaustion"]
        assert "sources_enumerated" in phase_i
        assert "source_disclosure_table" in phase_i
        assert "negative_certification" in phase_i
        assert "completeness_rating" in phase_i

        # Verify Phase II
        phase_ii = result["phase_ii_legal_analysis"]
        assert "1_procedural_posture" in phase_ii
        assert "2_charge_analysis" in phase_ii
        assert "3_constitutional_issues" in phase_ii
        assert "4_speedy_trial_computation" in phase_ii
        assert "5_plea_sentencing_forensics" in phase_ii
        assert "6_adversarial_dual_analysis" in phase_ii
        assert "7_outcome_probability_matrix" in phase_ii
        assert "8_post_conviction_review" in phase_ii
        assert "9_appellate_risk_profile" in phase_ii
        assert "10_final_audit_statement" in phase_ii

    def test_result_hash_integrity(self):
        """Test result hash is computed correctly."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)

        # Verify hash exists and is 64 chars (SHA-256 hex)
        assert len(result["result_hash"]) == 64
        assert all(c in "0123456789abcdef" for c in result["result_hash"])

    def test_disclaimer_present(self):
        """Test disclaimer is present in results."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)

        assert "DOES NOT PROVIDE LEGAL ADVICE" in result["disclaimer"]
        assert "INFORMATIONAL PURPOSES ONLY" in result["disclaimer"]


class TestPhaseIPublicRecordExhaustion:
    """Test Phase I: Public Record Exhaustion."""

    def test_source_enumeration(self):
        """Test all sources are enumerated."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        phase_i = result["phase_i_public_record_exhaustion"]

        sources = phase_i["sources_enumerated"]
        assert len(sources) >= 10
        assert any("Clerk of Courts" in s for s in sources)
        assert any("Supreme Court" in s for s in sources)

    def test_source_disclosure_table(self):
        """Test source disclosure table is complete."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        phase_i = result["phase_i_public_record_exhaustion"]

        table = phase_i["source_disclosure_table"]
        assert len(table) >= 5

        for row in table:
            assert "source" in row
            assert "searched" in row
            assert "accessible" in row
            assert "information_found" in row

    def test_negative_certification(self):
        """Test negative certification statement."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        phase_i = result["phase_i_public_record_exhaustion"]

        cert = phase_i["negative_certification"]
        assert "exhausting all publicly available sources" in cert

    def test_completeness_rating(self):
        """Test completeness rating is provided."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        phase_i = result["phase_i_public_record_exhaustion"]

        rating = phase_i["completeness_rating"]
        valid_ratings = [r.value for r in CompletenessRating]
        assert rating in valid_ratings


class TestPhaseIILegalAnalysis:
    """Test Phase II: Legal Analysis."""

    def test_charge_analysis_all_charges(self):
        """Test all charges are analyzed."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        charges = result["phase_ii_legal_analysis"]["2_charge_analysis"]

        # Should have 3 charges
        assert len(charges) == 3

        # Each should have statute analysis
        for charge in charges:
            assert "charge" in charge

    def test_constitutional_issues_spotted(self):
        """Test constitutional issues are identified."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        issues = result["phase_ii_legal_analysis"]["3_constitutional_issues"]

        assert len(issues) >= 5

        amendments_spotted = [i["amendment"] for i in issues]
        assert any("Fourth" in a for a in amendments_spotted)
        assert any("Fifth" in a for a in amendments_spotted)
        assert any("Sixth" in a for a in amendments_spotted)

    def test_speedy_trial_computation(self):
        """Test speedy trial calculation."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        st = result["phase_ii_legal_analysis"]["4_speedy_trial_computation"]

        assert st["statutory_limit_days"] == 270
        assert "R.C. 2945.71" in st["statutory_basis"]
        assert "waive" in st["waiver_analysis"].lower()  # "waives" contains "waive"

    def test_outcome_probability_matrix(self):
        """Test outcome probabilities sum correctly."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        outcomes = result["phase_ii_legal_analysis"]["7_outcome_probability_matrix"]

        total = sum(o["probability_percent"] for o in outcomes)
        assert total == 100

    def test_post_conviction_analysis(self):
        """Test post-conviction options are analyzed."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        pcr = result["phase_ii_legal_analysis"]["8_post_conviction_review"]

        assert "crim_r_32_1" in pcr
        assert "rc_2953_21" in pcr
        assert "ineffective_assistance" in pcr
        assert "sealing_eligibility" in pcr

    def test_sealing_eligibility_assault_ineligible(self):
        """Test assault is correctly identified as ineligible for sealing."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        sealing = result["phase_ii_legal_analysis"]["8_post_conviction_review"][
            "sealing_eligibility"
        ]

        assert "ineligible" in sealing
        assert "violence" in sealing["ineligible"].lower()

    def test_final_audit_statement(self):
        """Test final audit statement is complete."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)
        audit = result["phase_ii_legal_analysis"]["10_final_audit_statement"]

        assert "overall_confidence" in audit
        assert "top_3_facts_that_could_flip_outcome" in audit
        assert "documents_critical_to_obtain" in audit
        assert len(audit["top_3_facts_that_could_flip_outcome"]) == 3


class TestOhioVRinglerCase:
    """Test specific Ohio v. Ringler case analysis."""

    def test_case_details_correct(self):
        """Test case details are correct."""
        intent = create_ohio_v_ringler_intent()

        assert intent.case_number == "17CR052"
        assert intent.case_name == "STATE OF OHIO v. RINGLER, ROBERT"
        assert intent.county == "Holmes"
        assert intent.state == "Ohio"
        assert intent.defendant_name == "Robert Ringler"
        assert intent.judge == "Robert D. Rinfret"
        assert intent.defense_counsel == "Mark W. Baserman, Jr."

    def test_charges_correct(self):
        """Test charges are listed correctly."""
        intent = create_ohio_v_ringler_intent()

        assert len(intent.charges) == 3
        assert any("2903.13" in c for c in intent.charges)  # Assault
        assert any("2921.33" in c for c in intent.charges)  # Resisting
        assert any("2909.06" in c for c in intent.charges)  # Damaging

    def test_key_dates_present(self):
        """Test key dates are recorded."""
        intent = create_ohio_v_ringler_intent()

        dates_dict = dict(intent.key_dates)
        assert "plea" in dates_dict
        assert dates_dict["plea"] == "08/28/2017"
        assert "judicial_release" in dates_dict
        assert dates_dict["judicial_release"] == "12/21/2017"

    def test_full_analysis_runs(self):
        """Test full analysis runs without error."""
        engine = QRATUMJurisEngine()
        intent = create_ohio_v_ringler_intent()

        result = engine.analyze_criminal_case(intent)

        assert result is not None
        assert result["case_name"] == "STATE OF OHIO v. RINGLER, ROBERT"


class TestEnumerations:
    """Test enumeration classes."""

    def test_public_record_sources(self):
        """Test PublicRecordSource enum."""
        assert PublicRecordSource.CLERK_OF_COURTS.value == "clerk_of_courts"
        assert PublicRecordSource.CORRECTIONS_DEPARTMENT.value == "corrections_department"

    def test_source_accessibility(self):
        """Test SourceAccessibility enum."""
        assert SourceAccessibility.PUBLIC.value == "public"
        assert SourceAccessibility.RESTRICTED.value == "restricted"
        assert SourceAccessibility.SEALED.value == "sealed"

    def test_completeness_rating(self):
        """Test CompletenessRating enum."""
        assert "Fragmentary" in CompletenessRating.FRAGMENTARY.value
        assert "Partial" in CompletenessRating.PARTIAL.value
        assert "exhausted" in CompletenessRating.EXHAUSTED.value.lower()
