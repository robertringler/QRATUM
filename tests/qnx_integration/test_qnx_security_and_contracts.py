import json

from qnx.security import compute_integrity_hash, validate_security_context, verify_integrity
from qnx.sustainability import estimate_carbon
from qnx.types import SecurityLevel, SubstrateResult


def test_integrity_hash_round_trip():
    payload = {"engine": "quasim_modern", "timesteps": 2}
    digest = compute_integrity_hash(json.dumps(payload, sort_keys=True))

    assert verify_integrity(json.dumps(payload, sort_keys=True), digest)
    assert verify_integrity(json.dumps(payload, sort_keys=True), digest) is True


def test_security_validation_accepts_all_levels():
    for level in (SecurityLevel.LOW, SecurityLevel.MEDIUM, SecurityLevel.HIGH):
        validate_security_context(level)


def test_substrate_result_contract_roundtrip():
    result = SubstrateResult(
        backend="quasim_modern",
        scenario_id="contract",
        timesteps=2,
        seed=1,
        raw_results={"ok": True},
        simulation_hash="hash",
        execution_time_ms=1.2,
        carbon_emissions_kg=0.0,
        errors=[],
        warnings=[],
    )

    assert result.raw_results["ok"] is True
    assert result.seed == 1


def test_carbon_estimation_gracefully_handles_missing_codecarbon(monkeypatch):
    monkeypatch.setattr("qnx.sustainability.EmissionsTracker", None)
    estimate = estimate_carbon({"sample": True})

    assert estimate == 0.0
