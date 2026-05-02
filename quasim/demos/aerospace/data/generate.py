"""Synthetic data generator for aerospace demos."""

from pathlib import Path
from typing import Dict

import numpy as np

from quasim.common.seeding import set_global_seed
from quasim.common.serialize import save_npz


def generate_atmosphere_profile(
    altitudes: np.ndarray,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Generate atmospheric density and temperature profile.

    Args:
        altitudes: Altitude points in meters
        seed: Random seed

    Returns:
        Dictionary with 'density' and 'temperature' arrays
    """

    set_global_seed(seed)

    # Standard atmosphere model with noise
    density = 1.225 * np.exp(-altitudes / 8500)
    density += 0.01 * density * np.random.randn(len(altitudes))
    density = np.maximum(density, 1e-6)

    # Temperature (ISA model)
    temp_sl = 288.15  # K
    lapse_rate = -0.0065  # K/m
    temperature = temp_sl + lapse_rate * altitudes
    temperature = np.maximum(temperature, 200)

    return {
        "altitudes": altitudes,
        "density": density,
        "temperature": temperature,
    }


def generate_fixtures(output_dir: Path) -> None:
    """Generate fixture data files.

    Args:
        output_dir: Output directory for fixtures
    """

    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate atmosphere profile
    altitudes = np.linspace(0, 120000, 1000)
    atm_data = generate_atmosphere_profile(altitudes, seed=42)
    save_npz(atm_data, output_dir / "atmosphere.npz")

    print(f"Generated fixtures in {output_dir}")


if __name__ == "__main__":
    fixtures_dir = Path(__file__).parent / "fixtures"
    generate_fixtures(fixtures_dir)
