"""Hot staging scenarios."""

from typing import Any, Dict


def load_scenario(name: str) -> Dict[str, Any]:
    """Load predefined scenario.

    Args:
        name: Scenario name

    Returns:
        Scenario configuration dictionary
    """

    scenarios = {
        "starship": {
            "vehicle": "starship",
            "mass_kg": 1200000,  # Starship + SuperHeavy
            "thrust_n": 72000000,  # 33 Raptors on booster
            "isp_s": 330,
            "hot_staging": True,
            "staging_time": 150,
        },
        "falcon9": {
            "vehicle": "falcon9",
            "mass_kg": 549000,
            "thrust_n": 7607000,  # 9 Merlin engines
            "isp_s": 282,
            "hot_staging": False,
            "staging_time": 160,
        },
        "hot_staging_v1": {
            "vehicle": "test_vehicle",
            "mass_kg": 800000,
            "thrust_n": 40000000,
            "isp_s": 310,
            "hot_staging": True,
            "staging_time": 140,
        },
    }

    return scenarios.get(name, scenarios["starship"])
