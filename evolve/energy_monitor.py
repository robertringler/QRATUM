"""Energy-adaptive regulation with thermal control."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ThermalTelemetry:
    """Thermal and energy telemetry data."""

    timestamp: float = field(default_factory=time.time)
    temperature_celsius: float = 0.0
    power_watts: float = 0.0
    energy_joules: float = 0.0
    gflops: float = 0.0
    gflops_per_watt: float = 0.0

    def to_dict(self) -> dict:
        """Convert to dictionary."""

        return {
            "timestamp": self.timestamp,
            "temperature_celsius": self.temperature_celsius,
            "power_watts": self.power_watts,
            "energy_joules": self.energy_joules,
            "gflops": self.gflops,
            "gflops_per_watt": self.gflops_per_watt,
        }


@dataclass
class EfficiencyDashboard:
    """Dashboard data for energy efficiency metrics."""

    total_energy_j: float = 0.0
    average_power_w: float = 0.0
    peak_power_w: float = 0.0
    average_temp_c: float = 0.0
    peak_temp_c: float = 0.0
    average_efficiency: float = 0.0  # GFLOPs/W
    thermal_throttle_events: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary."""

        return {
            "total_energy_j": self.total_energy_j,
            "average_power_w": self.average_power_w,
            "peak_power_w": self.peak_power_w,
            "average_temp_c": self.average_temp_c,
            "peak_temp_c": self.peak_temp_c,
            "average_efficiency": self.average_efficiency,
            "thermal_throttle_events": self.thermal_throttle_events,
        }


class EnergyMonitor:
    """

    Energy-adaptive regulation using thermal telemetry.
    Implements closed-loop control with throttling.
    """

    def __init__(
        self,
        max_temp_celsius: float = 85.0,
        max_power_watts: float = 400.0,
    ):
        self.max_temp_celsius = max_temp_celsius
        self.max_power_watts = max_power_watts
        self.telemetry_history: list[ThermalTelemetry] = []
        self.throttle_events = 0

    def sample_telemetry(
        self, temperature: float, power: float, gflops: float, duration_s: float = 1.0
    ) -> ThermalTelemetry:
        """

        Sample thermal and energy telemetry.
        In real implementation, would query NVML/ROCm APIs.
        """

        energy = power * duration_s
        efficiency = gflops / power if power > 0 else 0.0

        telemetry = ThermalTelemetry(
            temperature_celsius=temperature,
            power_watts=power,
            energy_joules=energy,
            gflops=gflops,
            gflops_per_watt=efficiency,
        )

        self.telemetry_history.append(telemetry)
        return telemetry

    def check_thermal_limits(self, telemetry: ThermalTelemetry) -> bool:
        """

        Check if thermal limits are exceeded.
        Returns True if throttling is needed.
        """

        needs_throttle = False

        if telemetry.temperature_celsius > self.max_temp_celsius:
            needs_throttle = True

        if telemetry.power_watts > self.max_power_watts:
            needs_throttle = True

        if needs_throttle:
            self.throttle_events += 1

        return needs_throttle

    def compute_throttle_factor(self, telemetry: ThermalTelemetry) -> float:
        """

        Compute throttling factor based on thermal conditions.
        Returns value in [0.5, 1.0] where 1.0 = no throttling.
        """

        # Temperature-based throttling
        temp_factor = 1.0
        if telemetry.temperature_celsius > self.max_temp_celsius:
            overheat = telemetry.temperature_celsius - self.max_temp_celsius
            # Linear reduction: 10% per 5C over limit
            temp_factor = max(0.5, 1.0 - (overheat / 50.0))

        # Power-based throttling
        power_factor = 1.0
        if telemetry.power_watts > self.max_power_watts:
            overpower = telemetry.power_watts - self.max_power_watts
            # Linear reduction: 10% per 40W over limit
            power_factor = max(0.5, 1.0 - (overpower / 400.0))

        # Take minimum (most restrictive)
        return min(temp_factor, power_factor)

    def apply_feedback_control(self, telemetry: ThermalTelemetry) -> dict[str, float]:
        """

        Apply feedback control algorithm.
        Returns control parameters for throttling/migration.
        """

        needs_throttle = self.check_thermal_limits(telemetry)

        if needs_throttle:
            throttle_factor = self.compute_throttle_factor(telemetry)

            return {
                "throttle_factor": throttle_factor,
                "needs_migration": throttle_factor < 0.7,
                "recommended_freq_ghz": 1.5 * throttle_factor,
            }
        else:
            return {
                "throttle_factor": 1.0,
                "needs_migration": False,
                "recommended_freq_ghz": 1.5,
            }

    def generate_dashboard(self) -> EfficiencyDashboard:
        """Generate efficiency dashboard from collected telemetry."""

        if not self.telemetry_history:
            return EfficiencyDashboard()

        powers = [t.power_watts for t in self.telemetry_history]
        temps = [t.temperature_celsius for t in self.telemetry_history]
        energies = [t.energy_joules for t in self.telemetry_history]
        efficiencies = [t.gflops_per_watt for t in self.telemetry_history]

        dashboard = EfficiencyDashboard(
            total_energy_j=sum(energies),
            average_power_w=sum(powers) / len(powers),
            peak_power_w=max(powers),
            average_temp_c=sum(temps) / len(temps),
            peak_temp_c=max(temps),
            average_efficiency=sum(efficiencies) / len(efficiencies),
            thermal_throttle_events=self.throttle_events,
        )

        return dashboard

    def save_dashboard(
        self, dashboard: EfficiencyDashboard, output_path: str = "evolve/energy_dashboard.json"
    ) -> Path:
        """Save dashboard to disk."""

        dashboard_path = Path(output_path)
        dashboard_path.parent.mkdir(parents=True, exist_ok=True)

        with open(dashboard_path, "w") as f:
            json.dump(dashboard.to_dict(), f, indent=2)

        return dashboard_path
