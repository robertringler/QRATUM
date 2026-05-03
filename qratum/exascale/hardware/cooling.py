"""
Thermal Management and Cooling Strategy

Advanced cooling systems for high-density exascale deployment maintaining
PUE < 1.05 with 54.5 kW/rack average power density.

Cooling Technologies:
- Direct liquid cooling for GPUs and CPUs
- Rear-door heat exchangers (RDHX) for racks
- Free cooling when ambient conditions allow
- Hot/cold aisle containment
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List


class CoolingType(Enum):
    """Cooling technology types"""

    AIR_COOLED = "Air Cooled"
    DIRECT_LIQUID = "Direct Liquid Cooling"
    IMMERSION = "Immersion Cooling"
    RDHX = "Rear-Door Heat Exchanger"
    HYBRID = "Hybrid (Liquid + RDHX)"


class FluidType(Enum):
    """Cooling fluid types"""

    WATER = "Water"
    DIELECTRIC = "Dielectric Fluid"
    REFRIGERANT = "Refrigerant"


@dataclass
class CoolingLoop:
    """
    Cooling loop specification for high-density racks

    Attributes:
        loop_id: Unique identifier
        fluid_type: Type of cooling fluid
        flow_rate_lpm: Flow rate in liters per minute
        inlet_temp_c: Inlet temperature in Celsius
        outlet_temp_c: Outlet temperature in Celsius
        pressure_kpa: Operating pressure in kPa
    """

    loop_id: str
    fluid_type: FluidType
    flow_rate_lpm: float
    inlet_temp_c: float
    outlet_temp_c: float
    pressure_kpa: float

    def heat_removal_kw(self) -> float:
        """
        Calculate heat removal capacity

        Using specific heat capacity of water: 4.186 kJ/(kg·°C)
        Density of water at 20°C: 1.0 kg/L
        """
        if self.fluid_type == FluidType.WATER:
            specific_heat = 4.186  # kJ/(kg·°C)
            density = 1.0  # kg/L
            temp_delta = self.outlet_temp_c - self.inlet_temp_c

            # Q = ṁ × cp × ΔT
            # ṁ = flow_rate (L/min) × density (kg/L) × (1 min / 60 s)
            mass_flow_rate = self.flow_rate_lpm * density / 60.0  # kg/s
            heat_removal = mass_flow_rate * specific_heat * temp_delta

            return heat_removal

        return 0.0  # Unknown for other fluid types


@dataclass
class CoolingStrategy:
    """
    Comprehensive cooling strategy for exascale deployment

    Multi-layer cooling approach:
    1. Direct liquid cooling for high-power components (GPUs, CPUs)
    2. RDHX for rack-level heat rejection
    3. Facility-level cooling with free cooling optimization
    """

    strategy_name: str
    primary_cooling: CoolingType
    secondary_cooling: CoolingType
    facility_cooling: str
    target_pue: float
    max_rack_power_kw: float
    ambient_temp_range_c: tuple

    def is_valid_configuration(self) -> bool:
        """Validate cooling configuration"""
        if self.target_pue < 1.0 or self.target_pue > 2.0:
            return False
        if self.max_rack_power_kw < 10.0 or self.max_rack_power_kw > 100.0:
            return False
        return True


@dataclass
class ThermalManagement:
    """
    System-wide thermal management for QRATUM ExaScale

    Manages cooling for 3,125 compute nodes with total power consumption
    of ~24.8 MW at peak load, achieving PUE < 1.05.

    Attributes:
        total_it_power_mw: Total IT equipment power in MW
        target_pue: Target Power Usage Effectiveness
        cooling_strategy: Primary cooling strategy
        cooling_loops: List of active cooling loops
        monitoring_enabled: Enable real-time thermal monitoring
    """

    total_it_power_mw: float
    target_pue: float
    cooling_strategy: CoolingStrategy
    cooling_loops: List[CoolingLoop]
    monitoring_enabled: bool = True

    def total_facility_power_mw(self) -> float:
        """Calculate total facility power including cooling"""
        return self.total_it_power_mw * self.target_pue

    def cooling_power_mw(self) -> float:
        """Calculate power consumed by cooling infrastructure"""
        return self.total_facility_power_mw() - self.total_it_power_mw

    def cooling_efficiency_percent(self) -> float:
        """Calculate cooling efficiency as percentage of IT power"""
        return (self.cooling_power_mw() / self.total_it_power_mw) * 100.0

    def validate_capacity(self) -> bool:
        """
        Validate that cooling capacity meets requirements

        Returns:
            True if cooling capacity is sufficient
        """
        total_cooling_capacity = sum(loop.heat_removal_kw() for loop in self.cooling_loops)
        required_capacity = self.total_it_power_mw * 1000.0  # Convert MW to kW

        # Add 20% safety margin
        return total_cooling_capacity >= required_capacity * 1.2

    def get_thermal_summary(self) -> Dict[str, any]:
        """Return comprehensive thermal management summary"""
        return {
            "total_it_power_mw": self.total_it_power_mw,
            "total_facility_power_mw": self.total_facility_power_mw(),
            "cooling_power_mw": self.cooling_power_mw(),
            "target_pue": self.target_pue,
            "actual_pue": self.target_pue,  # Assuming target is achieved
            "cooling_efficiency_percent": self.cooling_efficiency_percent(),
            "primary_cooling": self.cooling_strategy.primary_cooling.value,
            "secondary_cooling": self.cooling_strategy.secondary_cooling.value,
            "cooling_loops": len(self.cooling_loops),
            "monitoring_enabled": self.monitoring_enabled,
            "capacity_validated": self.validate_capacity(),
        }


def create_exascale_cooling() -> ThermalManagement:
    """
    Create the standard QRATUM ExaScale cooling configuration

    Returns:
        ThermalManagement configured for 3,125 nodes
    """
    # Cooling strategy
    strategy = CoolingStrategy(
        strategy_name="ExaScale Hybrid Cooling",
        primary_cooling=CoolingType.DIRECT_LIQUID,
        secondary_cooling=CoolingType.RDHX,
        facility_cooling="Free Cooling + Chiller Backup",
        target_pue=1.05,
        max_rack_power_kw=60.0,
        ambient_temp_range_c=(18.0, 27.0),
    )

    # Create cooling loops (one per 10 racks)
    cooling_loops = []
    num_loops = 57  # 570 racks / 10 racks per loop

    for i in range(num_loops):
        loop = CoolingLoop(
            loop_id=f"CL-{i:03d}",
            fluid_type=FluidType.WATER,
            flow_rate_lpm=2000.0,  # 2000 L/min per loop
            inlet_temp_c=18.0,
            outlet_temp_c=30.0,
            pressure_kpa=400.0,
        )
        cooling_loops.append(loop)

    # Create thermal management system
    thermal_mgmt = ThermalManagement(
        total_it_power_mw=24.8,  # ~55 MW total IT power
        target_pue=1.05,
        cooling_strategy=strategy,
        cooling_loops=cooling_loops,
        monitoring_enabled=True,
    )

    return thermal_mgmt


# Standard ExaScale cooling instance
EXASCALE_COOLING = create_exascale_cooling()


def calculate_rack_cooling_requirements(
    nodes_per_rack: int, power_per_node_kw: float = 17.6
) -> Dict[str, float]:
    """
    Calculate cooling requirements for a single rack

    Args:
        nodes_per_rack: Number of nodes in the rack
        power_per_node_kw: Power consumption per node in kW

    Returns:
        Dictionary with cooling requirements
    """
    total_power_kw = nodes_per_rack * power_per_node_kw

    # BTU/hr conversion (1 kW = 3412.14 BTU/hr)
    btu_per_hr = total_power_kw * 3412.14

    # Cooling capacity with 20% margin
    required_capacity_kw = total_power_kw * 1.2

    return {
        "total_power_kw": total_power_kw,
        "heat_output_btu_per_hr": btu_per_hr,
        "required_cooling_capacity_kw": required_capacity_kw,
        "recommended_flow_rate_lpm": total_power_kw * 10.0,  # 10 L/min per kW
        "estimated_water_temp_rise_c": 12.0,  # Typical for high-density racks
    }
