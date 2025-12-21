"""
QIL Intent Examples - All Verticals

Comprehensive examples for all 14 vertical modules.
"""

from qratum.platform import PlatformIntent


# JURIS - Legal AI
juris_intent = PlatformIntent(
    vertical="JURIS",
    task="analyze_contract",
    parameters={"contract_text": "Sample contract with indemnification clause..."},
    requester_id="legal_dept",
)

# VITRA - Bioinformatics
vitra_intent = PlatformIntent(
    vertical="VITRA",
    task="predict_structure",
    parameters={"protein_sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQ"},
    requester_id="biotech_lab",
)

# ECORA - Climate & Energy
ecora_intent = PlatformIntent(
    vertical="ECORA",
    task="assess_climate_risk",
    parameters={"asset_location": "Miami, FL", "time_horizon_years": 30},
    requester_id="risk_management",
)

# CAPRA - Financial Risk
capra_intent = PlatformIntent(
    vertical="CAPRA",
    task="calculate_risk_metrics",
    parameters={"portfolio_value": 10000000, "confidence_level": 0.99},
    requester_id="quant_team",
)

# SENTRA - Aerospace & Defense
sentra_intent = PlatformIntent(
    vertical="SENTRA",
    task="plan_mission",
    parameters={
        "mission_type": "reconnaissance",
        "waypoints": [[34.0, -118.0], [35.0, -117.0], [36.0, -116.0]],
    },
    requester_id="mission_control",
)

# NEURA - Neuroscience & BCI
neura_intent = PlatformIntent(
    vertical="NEURA",
    task="process_brain_signals",
    parameters={"type": "EEG", "sample_rate_hz": 250, "duration_s": 60},
    requester_id="neuroscience_lab",
)

# FLUXA - Supply Chain
fluxa_intent = PlatformIntent(
    vertical="FLUXA",
    task="design_network",
    parameters={"num_facilities": 10, "demand_centers": 50, "constraints": ["cost", "time"]},
    requester_id="supply_chain_ops",
)

# CHRONA - Semiconductor
chrona_intent = PlatformIntent(
    vertical="CHRONA",
    task="analyze_timing",
    parameters={"circuit_design": "cpu_core.v", "node": "5nm", "frequency_ghz": 3.5},
    requester_id="chip_design_team",
)

# GEONA - Geospatial
geona_intent = PlatformIntent(
    vertical="GEONA",
    task="predict_disaster",
    parameters={"type": "earthquake", "location": "San Francisco Bay Area", "timeframe_days": 30},
    requester_id="emergency_mgmt",
)

# FUSIA - Nuclear & Fusion
fusia_intent = PlatformIntent(
    vertical="FUSIA",
    task="optimize_reactor",
    parameters={"reactor_type": "PWR", "power_mw": 1000, "fuel_type": "UO2"},
    requester_id="nuclear_engineering",
)

# STRATA - Policy & Macro
strata_intent = PlatformIntent(
    vertical="STRATA",
    task="simulate_policy",
    parameters={"policy": "universal_basic_income", "amount_monthly": 1000, "coverage": 0.95},
    requester_id="policy_research",
)

# VEXOR - Cybersecurity
vexor_intent = PlatformIntent(
    vertical="VEXOR",
    task="simulate_attack",
    parameters={"type": "ransomware", "target_network": "corporate", "defense_level": "moderate"},
    requester_id="security_ops",
)

# COHORA - Robotics
cohora_intent = PlatformIntent(
    vertical="COHORA",
    task="plan_path",
    parameters={"start": [0, 0, 0], "goal": [100, 50, 10], "obstacles": [[50, 25, 5]]},
    requester_id="robotics_control",
)

# ORBIA - Space Systems
orbia_intent = PlatformIntent(
    vertical="ORBIA",
    task="optimize_constellation",
    parameters={"satellites": 300, "coverage_goal": 0.99, "orbit_type": "LEO"},
    requester_id="space_ops",
)


ALL_INTENT_EXAMPLES = [
    ("JURIS", juris_intent),
    ("VITRA", vitra_intent),
    ("ECORA", ecora_intent),
    ("CAPRA", capra_intent),
    ("SENTRA", sentra_intent),
    ("NEURA", neura_intent),
    ("FLUXA", fluxa_intent),
    ("CHRONA", chrona_intent),
    ("GEONA", geona_intent),
    ("FUSIA", fusia_intent),
    ("STRATA", strata_intent),
    ("VEXOR", vexor_intent),
    ("COHORA", cohora_intent),
    ("ORBIA", orbia_intent),
]


def print_all_intents():
    """Print summary of all example intents"""
    
    print("QRATUM Platform - QIL Intent Examples for All 14 Verticals")
    print("=" * 80)
    print()
    
    for name, intent in ALL_INTENT_EXAMPLES:
        print(f"{name:12} | {intent.task:30} | {intent.intent_id}")
    
    print()
    print("=" * 80)
    print(f"Total: {len(ALL_INTENT_EXAMPLES)} vertical modules with example intents")
    print("=" * 80)


if __name__ == "__main__":
    print_all_intents()
