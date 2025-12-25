"""ECORA - Climate & Energy Module for QRATUM Platform.

Provides climate projection, energy grid optimization, carbon footprint analysis,
weather prediction, and renewable site assessment.
"""

import math
from typing import Any, Dict, List

from qratum_platform.core import (ComputeSubstrate, PlatformContract,
                                  VerticalModuleBase)
from qratum_platform.substrates import VerticalModule, get_optimal_substrate


class ECORAModule(VerticalModuleBase):
    """Climate and Energy module for environmental analysis."""

    MODULE_NAME = "ECORA"
    MODULE_VERSION = "2.0.0"
    SAFETY_DISCLAIMER = """
    ECORA Climate & Energy Disclaimer:
    - Climate projections are subject to uncertainty
    - Not a substitute for official meteorological services
    - Energy recommendations require professional engineering review
    - Grid operations must comply with regulatory requirements
    - Carbon calculations are estimates - verify with standards
    - Always consult qualified environmental engineers
    """

    PROHIBITED_USES = [
        "bypassing environmental regulations",
        "false carbon credit claims",
        "unauthorized grid manipulation",
        "environmental fraud",
        "greenwashing without data",
    ]

    # Emission factors (kg CO2e per unit)
    EMISSION_FACTORS = {
        "electricity_grid": 0.5,  # kg CO2e per kWh
        "natural_gas": 0.185,  # kg CO2e per kWh
        "coal": 0.34,  # kg CO2e per kWh
        "gasoline": 2.31,  # kg CO2e per liter
        "diesel": 2.68,  # kg CO2e per liter
        "aviation_fuel": 2.52,  # kg CO2e per liter
    }

    # SSP scenarios (Shared Socioeconomic Pathways)
    SSP_SCENARIOS = {
        "SSP1-1.9": {"warming_2100": 1.5, "description": "Very low emissions"},
        "SSP1-2.6": {"warming_2100": 1.8, "description": "Low emissions"},
        "SSP2-4.5": {"warming_2100": 2.7, "description": "Intermediate emissions"},
        "SSP3-7.0": {"warming_2100": 3.6, "description": "High emissions"},
        "SSP5-8.5": {"warming_2100": 4.4, "description": "Very high emissions"},
    }

    def execute(self, contract: PlatformContract) -> Dict[str, Any]:
        """Execute climate and energy operation.

        Args:
            contract: Immutable execution contract

        Returns:
            Results of climate/energy analysis
        """
        operation = contract.intent.operation
        parameters = contract.intent.parameters

        self.emit_event("ecora_execution_start", contract.contract_id, {"operation": operation})

        try:
            if operation == "climate_projection":
                result = self._project_climate(parameters)
            elif operation == "grid_optimization":
                result = self._optimize_energy_grid(parameters)
            elif operation == "carbon_analysis":
                result = self._analyze_carbon_footprint(parameters)
            elif operation == "weather_prediction":
                result = self._predict_weather(parameters)
            elif operation == "renewable_assessment":
                result = self._assess_renewable_site(parameters)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            self.emit_event(
                "ecora_execution_complete",
                contract.contract_id,
                {"operation": operation, "success": True},
            )

            return result

        except Exception as e:
            self.emit_event(
                "ecora_execution_failed",
                contract.contract_id,
                {"operation": operation, "error": str(e)},
            )
            raise

    def _project_climate(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Project climate scenarios using SSP pathways.

        Args:
            parameters: Must contain 'scenario' and 'region'

        Returns:
            Climate projection results
        """
        scenario = parameters.get("scenario", "SSP2-4.5")
        region = parameters.get("region", "global")
        target_year = parameters.get("target_year", 2100)

        ssp_data = self.SSP_SCENARIOS.get(scenario, self.SSP_SCENARIOS["SSP2-4.5"])

        # Calculate radiative forcing
        years_from_now = target_year - 2024
        forcing = self._calculate_radiative_forcing(scenario, years_from_now)

        return {
            "projection_type": "climate_scenario",
            "scenario": scenario,
            "region": region,
            "target_year": target_year,
            "projected_warming_c": ssp_data["warming_2100"],
            "radiative_forcing_w_m2": forcing,
            "impacts": {
                "sea_level_rise_cm": ssp_data["warming_2100"] * 20,  # Simplified
                "extreme_heat_days_increase": int(ssp_data["warming_2100"] * 10),
                "precipitation_change_percent": ssp_data["warming_2100"] * 5,
            },
            "confidence_level": "medium",
            "note": "Simplified projection - use CMIP6 models for detailed analysis",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _calculate_radiative_forcing(self, scenario: str, years: int) -> float:
        """Calculate radiative forcing for scenario."""
        # Simplified calculation
        base_forcing = {"SSP1-1.9": 1.9, "SSP1-2.6": 2.6, "SSP2-4.5": 4.5, "SSP3-7.0": 7.0, "SSP5-8.5": 8.5}
        return base_forcing.get(scenario, 4.5)

    def _optimize_energy_grid(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize energy grid generation mix.

        Args:
            parameters: Must contain 'demand_mw' and 'available_sources'

        Returns:
            Optimal generation mix
        """
        demand_mw = parameters.get("demand_mw", 1000)
        available_sources = parameters.get(
            "available_sources",
            [
                {"type": "solar", "capacity_mw": 300, "cost_per_mwh": 40},
                {"type": "wind", "capacity_mw": 250, "cost_per_mwh": 45},
                {"type": "natural_gas", "capacity_mw": 500, "cost_per_mwh": 60},
                {"type": "battery", "capacity_mw": 100, "cost_per_mwh": 80},
            ],
        )

        # Simple optimization: minimize cost while meeting demand
        generation_mix = self._optimize_generation_mix(demand_mw, available_sources)

        total_cost = sum(s["generation_mw"] * s["cost_per_mwh"] for s in generation_mix)
        total_generation = sum(s["generation_mw"] for s in generation_mix)

        # Calculate emissions
        emissions = self._calculate_grid_emissions(generation_mix)

        return {
            "optimization_type": "grid_dispatch",
            "demand_mw": demand_mw,
            "total_generation_mw": total_generation,
            "generation_mix": generation_mix,
            "total_cost_per_hour": total_cost,
            "emissions_kg_co2_per_hour": emissions,
            "renewable_percentage": self._calculate_renewable_percentage(generation_mix),
            "note": "Simplified optimization - use advanced OPF tools for production",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _optimize_generation_mix(
        self, demand: float, sources: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Optimize generation mix (simplified merit order)."""
        # Sort by cost (merit order)
        sorted_sources = sorted(sources, key=lambda x: x["cost_per_mwh"])

        mix = []
        remaining_demand = demand

        for source in sorted_sources:
            generation = min(remaining_demand, source["capacity_mw"])
            if generation > 0:
                mix.append(
                    {
                        "type": source["type"],
                        "generation_mw": generation,
                        "cost_per_mwh": source["cost_per_mwh"],
                    }
                )
                remaining_demand -= generation

            if remaining_demand <= 0:
                break

        return mix

    def _calculate_grid_emissions(self, generation_mix: List[Dict[str, Any]]) -> float:
        """Calculate emissions from generation mix."""
        emission_rates = {
            "solar": 0.0,
            "wind": 0.0,
            "natural_gas": 0.185,
            "coal": 0.34,
            "battery": 0.0,  # Assuming charged from renewables
        }

        total_emissions = 0.0
        for source in generation_mix:
            rate = emission_rates.get(source["type"], 0.5)
            total_emissions += source["generation_mw"] * rate

        return total_emissions

    def _calculate_renewable_percentage(self, generation_mix: List[Dict[str, Any]]) -> float:
        """Calculate percentage of renewable generation."""
        renewable_types = {"solar", "wind", "hydro", "battery"}

        total = sum(s["generation_mw"] for s in generation_mix)
        renewable = sum(
            s["generation_mw"] for s in generation_mix if s["type"] in renewable_types
        )

        return (renewable / total * 100) if total > 0 else 0.0

    def _analyze_carbon_footprint(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze carbon footprint of activities.

        Args:
            parameters: Must contain 'activities' list

        Returns:
            Carbon footprint analysis
        """
        activities = parameters.get("activities", [])

        emissions_by_category = {}
        total_emissions = 0.0

        for activity in activities:
            category = activity.get("category", "other")
            emissions = self._calculate_activity_emissions(activity)
            emissions_by_category[category] = emissions_by_category.get(category, 0.0) + emissions
            total_emissions += emissions

        return {
            "analysis_type": "carbon_footprint",
            "total_emissions_kg_co2e": total_emissions,
            "emissions_by_category": emissions_by_category,
            "equivalent_trees_to_offset": int(total_emissions / 20),  # ~20 kg CO2/tree/year
            "recommendations": self._generate_reduction_recommendations(emissions_by_category),
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _calculate_activity_emissions(self, activity: Dict[str, Any]) -> float:
        """Calculate emissions for a single activity."""
        activity_type = activity.get("type", "electricity_grid")
        amount = activity.get("amount", 0)

        emission_factor = self.EMISSION_FACTORS.get(activity_type, 0.5)
        return amount * emission_factor

    def _generate_reduction_recommendations(
        self, emissions_by_category: Dict[str, float]
    ) -> List[str]:
        """Generate emissions reduction recommendations."""
        recommendations = []

        # Sort by highest emissions
        sorted_categories = sorted(emissions_by_category.items(), key=lambda x: x[1], reverse=True)

        for category, emissions in sorted_categories[:3]:
            if category == "electricity_grid":
                recommendations.append("Switch to renewable energy sources")
            elif category in ["gasoline", "diesel"]:
                recommendations.append("Consider electric vehicles or public transport")
            elif category == "natural_gas":
                recommendations.append("Improve building insulation and heating efficiency")

        return recommendations

    def _predict_weather(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Predict weather patterns.

        Args:
            parameters: Must contain 'location' and 'forecast_hours'

        Returns:
            Weather prediction
        """
        location = parameters.get("location", "unknown")
        forecast_hours = parameters.get("forecast_hours", 24)

        # Simplified weather prediction
        # In production, this would use numerical weather prediction models
        return {
            "prediction_type": "weather_forecast",
            "location": location,
            "forecast_hours": forecast_hours,
            "forecast": [
                {
                    "hour": i,
                    "temperature_c": 20 + 5 * math.sin(i / 12 * math.pi),
                    "precipitation_mm": max(0, (i % 6 - 3) * 0.5),
                    "wind_speed_ms": 5 + 2 * math.sin(i / 6 * math.pi),
                    "cloud_cover_percent": 30 + 20 * math.sin(i / 8 * math.pi),
                }
                for i in range(min(forecast_hours, 48))
            ],
            "confidence": "medium",
            "note": "Simplified forecast - use WRF/GFS for production",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _assess_renewable_site(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Assess site for renewable energy installation.

        Args:
            parameters: Must contain 'location', 'renewable_type', 'site_area_m2'

        Returns:
            Site assessment results
        """
        location = parameters.get("location", "unknown")
        renewable_type = parameters.get("renewable_type", "solar")
        site_area_m2 = parameters.get("site_area_m2", 10000)
        latitude = parameters.get("latitude", 40.0)

        if renewable_type == "solar":
            result = self._assess_solar_site(site_area_m2, latitude)
        elif renewable_type == "wind":
            result = self._assess_wind_site(site_area_m2, location)
        else:
            raise ValueError(f"Unknown renewable type: {renewable_type}")

        result.update(
            {
                "assessment_type": "renewable_site",
                "location": location,
                "renewable_type": renewable_type,
                "site_area_m2": site_area_m2,
                "disclaimer": self.SAFETY_DISCLAIMER,
            }
        )

        return result

    def _assess_solar_site(self, area_m2: float, latitude: float) -> Dict[str, Any]:
        """Assess solar installation potential."""
        # Solar irradiance varies by latitude
        avg_irradiance = 1000 * math.cos(math.radians(abs(latitude)))  # W/m2
        panel_efficiency = 0.20  # 20% efficient panels
        system_losses = 0.85  # 15% system losses

        capacity_kw = area_m2 * panel_efficiency
        annual_generation_kwh = capacity_kw * avg_irradiance * 0.001 * 365 * 5 * system_losses

        return {
            "estimated_capacity_kw": capacity_kw,
            "annual_generation_kwh": annual_generation_kwh,
            "capacity_factor": 0.15 + 0.05 * (1 - abs(latitude) / 90),
            "levelized_cost_per_kwh": 0.04,
            "payback_period_years": 8,
            "co2_avoided_kg_per_year": annual_generation_kwh * self.EMISSION_FACTORS["electricity_grid"],
        }

    def _assess_wind_site(self, area_m2: float, location: str) -> Dict[str, Any]:
        """Assess wind installation potential."""
        # Simplified wind assessment
        turbine_capacity_kw = 2000  # 2 MW turbines
        num_turbines = max(1, int(area_m2 / 50000))  # One turbine per 5 hectares

        capacity_kw = num_turbines * turbine_capacity_kw
        annual_generation_kwh = capacity_kw * 8760 * 0.30  # 30% capacity factor

        return {
            "estimated_capacity_kw": capacity_kw,
            "num_turbines": num_turbines,
            "annual_generation_kwh": annual_generation_kwh,
            "capacity_factor": 0.30,
            "levelized_cost_per_kwh": 0.05,
            "payback_period_years": 10,
            "co2_avoided_kg_per_year": annual_generation_kwh * self.EMISSION_FACTORS["electricity_grid"],
        }

    def get_optimal_substrate(
        self, operation: str, parameters: Dict[str, Any]
    ) -> ComputeSubstrate:
        """Get optimal compute substrate for climate/energy operation."""
        return get_optimal_substrate(VerticalModule.ECORA, operation)
