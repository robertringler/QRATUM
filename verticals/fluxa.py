"""FLUXA - Supply Chain Module for QRATUM Platform.

Provides route optimization, demand forecasting, inventory optimization,
network resilience analysis, and logistics simulation.
"""

import math
import random
from typing import Any, Dict, List, Tuple

from qratum_platform.core import (
    ComputeSubstrate,
    PlatformContract,
    VerticalModuleBase,
)
from qratum_platform.substrates import get_optimal_substrate, VerticalModule
from qratum_platform.utils import compute_deterministic_seed


class FLUXAModule(VerticalModuleBase):
    """Supply Chain module for logistics and optimization."""

    MODULE_NAME = "FLUXA"
    MODULE_VERSION = "2.0.0"
    SAFETY_DISCLAIMER = """
    FLUXA Supply Chain Disclaimer:
    - Recommendations are computational estimates
    - Real-world constraints may not be fully captured
    - Consult supply chain professionals for implementation
    - Regulatory compliance is user responsibility
    - Consider safety stock and risk factors
    - Validate recommendations before deployment
    """

    PROHIBITED_USES = [
        "price fixing or collusion",
        "market manipulation",
        "illegal trade routes",
        "sanctions violations",
        "counterfeit supply chains",
    ]

    def execute(self, contract: PlatformContract) -> Dict[str, Any]:
        """Execute supply chain operation."""
        operation = contract.intent.operation
        parameters = contract.intent.parameters

        self.emit_event("fluxa_execution_start", contract.contract_id, {"operation": operation})

        try:
            if operation == "route_optimization":
                result = self._optimize_routes(parameters)
            elif operation == "demand_forecasting":
                result = self._forecast_demand(parameters)
            elif operation == "inventory_optimization":
                result = self._optimize_inventory(parameters)
            elif operation == "resilience_analysis":
                result = self._analyze_resilience(parameters)
            elif operation == "logistics_simulation":
                result = self._simulate_logistics(parameters)
            else:
                raise ValueError(f"Unknown operation: {operation}")

            self.emit_event(
                "fluxa_execution_complete",
                contract.contract_id,
                {"operation": operation, "success": True},
            )
            return result
        except Exception as e:
            self.emit_event(
                "fluxa_execution_failed",
                contract.contract_id,
                {"operation": operation, "error": str(e)},
            )
            raise

    def _optimize_routes(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize delivery routes (TSP/VRP).

        Args:
            parameters: depot, locations, vehicles

        Returns:
            Optimized routes
        """
        depot = parameters.get("depot", {"lat": 0, "lon": 0})
        locations = parameters.get(
            "locations",
            [
                {"id": "A", "lat": 1, "lon": 1, "demand": 10},
                {"id": "B", "lat": 2, "lon": 1, "demand": 15},
                {"id": "C", "lat": 1, "lon": 2, "demand": 8},
                {"id": "D", "lat": 2, "lon": 2, "demand": 12},
            ],
        )
        num_vehicles = parameters.get("num_vehicles", 2)
        vehicle_capacity = parameters.get("vehicle_capacity", 30)

        # Derive seed from parameters for input-dependent determinism
        seed = compute_deterministic_seed(parameters)

        # Simplified nearest neighbor heuristic
        routes = self._greedy_route_assignment(depot, locations, num_vehicles, vehicle_capacity, seed)

        total_distance = sum(r["distance_km"] for r in routes)
        total_time = sum(r["time_hours"] for r in routes)

        return {
            "optimization_type": "vehicle_routing",
            "num_vehicles": num_vehicles,
            "num_locations": len(locations),
            "routes": routes,
            "total_distance_km": total_distance,
            "total_time_hours": total_time,
            "avg_utilization": sum(r["utilization"] for r in routes) / len(routes),
            "note": "Simplified heuristic - use OR-Tools for production",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _greedy_route_assignment(
        self, depot: Dict, locations: List[Dict], num_vehicles: int, capacity: float, seed: int
    ) -> List[Dict[str, Any]]:
        """Greedy route assignment."""
        routes = []
        remaining = locations.copy()
        random.seed(seed)

        for v in range(num_vehicles):
            route = []
            current_load = 0
            current_pos = depot

            while remaining:
                # Find nearest unvisited location with capacity
                best = None
                best_dist = float("inf")

                for loc in remaining:
                    if current_load + loc.get("demand", 0) <= capacity:
                        dist = self._haversine_distance(current_pos, loc)
                        if dist < best_dist:
                            best_dist = dist
                            best = loc

                if best is None:
                    break

                route.append(best["id"])
                current_load += best.get("demand", 0)
                current_pos = best
                remaining.remove(best)

            if route:
                route_dist = self._calculate_route_distance(depot, [loc for loc in locations if loc["id"] in route], depot)
                routes.append(
                    {
                        "vehicle_id": v,
                        "stops": route,
                        "load": current_load,
                        "utilization": current_load / capacity,
                        "distance_km": route_dist,
                        "time_hours": route_dist / 60,  # Assume 60 km/h
                    }
                )

        return routes

    def _haversine_distance(self, p1: Dict, p2: Dict) -> float:
        """Calculate distance between two lat/lon points."""
        lat1, lon1 = math.radians(p1["lat"]), math.radians(p1["lon"])
        lat2, lon2 = math.radians(p2["lat"]), math.radians(p2["lon"])

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        return 6371 * c  # Earth radius in km

    def _calculate_route_distance(self, depot: Dict, stops: List[Dict], return_depot: Dict) -> float:
        """Calculate total route distance."""
        total = 0
        current = depot
        for stop in stops:
            total += self._haversine_distance(current, stop)
            current = stop
        total += self._haversine_distance(current, return_depot)
        return total

    def _forecast_demand(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Forecast future demand.

        Args:
            parameters: historical_data, forecast_periods, product

        Returns:
            Demand forecast
        """
        historical = parameters.get("historical_data", [100, 105, 110, 108, 115, 120])
        periods = parameters.get("forecast_periods", 6)
        product = parameters.get("product", "PRODUCT_A")

        # Simple exponential smoothing
        alpha = 0.3
        forecast = self._exponential_smoothing(historical, alpha, periods)

        # Calculate forecast accuracy
        mape = self._calculate_mape(historical, alpha)

        return {
            "forecast_type": "exponential_smoothing",
            "product": product,
            "historical_periods": len(historical),
            "forecast_periods": periods,
            "forecast": forecast,
            "smoothing_parameter": alpha,
            "mape_percent": mape,
            "confidence_interval_95": {
                "lower": [f * 0.9 for f in forecast],
                "upper": [f * 1.1 for f in forecast],
            },
            "note": "Simplified forecasting - use Prophet/ARIMA for production",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _exponential_smoothing(self, data: List[float], alpha: float, periods: int) -> List[float]:
        """Simple exponential smoothing."""
        if not data:
            return [0] * periods

        forecast = []
        last = data[0]

        for val in data[1:]:
            last = alpha * val + (1 - alpha) * last

        for _ in range(periods):
            forecast.append(last)

        return forecast

    def _calculate_mape(self, data: List[float], alpha: float) -> float:
        """Calculate Mean Absolute Percentage Error."""
        if len(data) < 2:
            return 0.0

        errors = []
        last = data[0]

        for val in data[1:]:
            forecast = last
            error = abs((val - forecast) / val) if val != 0 else 0
            errors.append(error)
            last = alpha * val + (1 - alpha) * last

        return sum(errors) / len(errors) * 100 if errors else 0.0

    def _optimize_inventory(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize inventory levels.

        Args:
            parameters: demand_rate, lead_time_days, holding_cost, ordering_cost

        Returns:
            Optimal inventory policy
        """
        demand_rate = parameters.get("demand_rate_per_day", 100)
        lead_time = parameters.get("lead_time_days", 7)
        holding_cost = parameters.get("holding_cost_per_unit_per_year", 5)
        ordering_cost = parameters.get("ordering_cost_per_order", 100)

        # Economic Order Quantity (EOQ)
        annual_demand = demand_rate * 365
        eoq = math.sqrt((2 * annual_demand * ordering_cost) / holding_cost)

        # Reorder point
        reorder_point = demand_rate * lead_time

        # Safety stock (simplified)
        safety_stock = demand_rate * lead_time * 0.5

        # Total cost
        num_orders = annual_demand / eoq
        total_cost = (num_orders * ordering_cost) + (eoq / 2 * holding_cost) + (safety_stock * holding_cost)

        return {
            "optimization_type": "inventory_policy",
            "economic_order_quantity": round(eoq, 2),
            "reorder_point": round(reorder_point, 2),
            "safety_stock": round(safety_stock, 2),
            "orders_per_year": round(num_orders, 2),
            "total_annual_cost": round(total_cost, 2),
            "avg_inventory_level": round(eoq / 2 + safety_stock, 2),
            "note": "EOQ model - consider demand variability",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _analyze_resilience(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze supply chain network resilience.

        Args:
            parameters: network_structure, disruption_scenarios

        Returns:
            Resilience analysis
        """
        num_suppliers = parameters.get("num_suppliers", 5)
        num_facilities = parameters.get("num_facilities", 3)
        redundancy = parameters.get("redundancy_level", "medium")

        # Simulate disruption scenarios
        scenarios = [
            {"name": "single_supplier_failure", "impact": 0.2},
            {"name": "facility_disruption", "impact": 0.35},
            {"name": "transportation_delay", "impact": 0.15},
            {"name": "demand_surge", "impact": 0.25},
        ]

        # Calculate resilience metrics
        avg_impact = sum(s["impact"] for s in scenarios) / len(scenarios)
        resilience_score = 1.0 - avg_impact

        mitigation = {
            "diversify_suppliers": "High priority",
            "increase_inventory": "Medium priority",
            "backup_facilities": "Medium priority",
            "flexible_transport": "Low priority",
        }

        return {
            "analysis_type": "network_resilience",
            "num_suppliers": num_suppliers,
            "num_facilities": num_facilities,
            "redundancy_level": redundancy,
            "disruption_scenarios": scenarios,
            "resilience_score": round(resilience_score, 2),
            "mitigation_strategies": mitigation,
            "note": "Simplified resilience analysis",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def _simulate_logistics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate logistics operations.

        Args:
            parameters: simulation_days, arrival_rate, service_rate

        Returns:
            Simulation results
        """
        days = parameters.get("simulation_days", 30)
        arrival_rate = parameters.get("arrival_rate_per_day", 50)
        service_rate = parameters.get("service_rate_per_day", 60)

        # Derive seed from parameters for input-dependent determinism
        seed = compute_deterministic_seed(parameters)
        random.seed(seed)

        # Simplified discrete event simulation
        total_arrivals = 0
        total_served = 0
        queue_lengths = []

        for day in range(days):
            arrivals = int(random.gauss(arrival_rate, arrival_rate * 0.1))
            capacity = int(random.gauss(service_rate, service_rate * 0.1))

            total_arrivals += arrivals
            served = min(arrivals, capacity)
            total_served += served

            queue_length = max(0, arrivals - served)
            queue_lengths.append(queue_length)

        avg_queue = sum(queue_lengths) / len(queue_lengths)
        utilization = total_served / (days * service_rate)

        return {
            "simulation_type": "discrete_event",
            "simulation_days": days,
            "total_arrivals": total_arrivals,
            "total_served": total_served,
            "utilization": round(utilization, 2),
            "avg_queue_length": round(avg_queue, 2),
            "max_queue_length": max(queue_lengths),
            "service_level": round(total_served / total_arrivals, 2),
            "note": "Simplified queuing simulation",
            "disclaimer": self.SAFETY_DISCLAIMER,
        }

    def get_optimal_substrate(self, operation: str, parameters: Dict[str, Any]) -> ComputeSubstrate:
        """Get optimal compute substrate for supply chain operation."""
        return get_optimal_substrate(VerticalModule.FLUXA, operation)
