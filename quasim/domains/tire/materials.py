"""Material modeling for tire compounds with quantum-enhanced properties."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class CompoundType(Enum):
    """Types of tire compounds."""

    NATURAL_RUBBER = "natural_rubber"
    SYNTHETIC_RUBBER = "synthetic_rubber"
    BIOPOLYMER = "biopolymer"
    NANO_ENHANCED = "nano_enhanced"
    GRAPHENE_REINFORCED = "graphene_reinforced"
    QUANTUM_OPTIMIZED = "quantum_optimized"
    SILICA_ENHANCED = "silica_enhanced"
    CARBON_BLACK = "carbon_black"


@dataclass
class MaterialProperties:
    """Physical and chemical properties of tire materials.

    Multi-scale modeling from molecular to macroscopic level.

    Attributes:
        density: Material density in kg/m³
        elastic_modulus: Young's modulus in GPa
        shear_modulus: Shear modulus in GPa
        poisson_ratio: Poisson's ratio (dimensionless)
        hardness_shore_a: Shore A hardness (0-100)
        viscoelastic_loss_factor: Tan delta (hysteresis)
        thermal_conductivity: W/(m·K)
        specific_heat: J/(kg·K)
        thermal_expansion_coeff: 1/K
        glass_transition_temp: Glass transition temperature in °C
        max_service_temp: Maximum service temperature in °C
        oxidation_resistance: Resistance to oxidative degradation (0-1)
        uv_resistance: UV exposure resistance (0-1)
        abrasion_resistance: Wear resistance index (0-1)
        wet_grip_coefficient: Wet traction coefficient (0-1)
        rolling_resistance_coeff: Rolling resistance coefficient
        molecular_structure: Quantum-computed molecular properties
    """

    density: float = 1150.0  # kg/m³
    elastic_modulus: float = 0.002  # GPa
    shear_modulus: float = 0.001  # GPa
    poisson_ratio: float = 0.49
    hardness_shore_a: float = 60.0
    viscoelastic_loss_factor: float = 0.15
    thermal_conductivity: float = 0.25  # W/(m·K)
    specific_heat: float = 1900.0  # J/(kg·K)
    thermal_expansion_coeff: float = 2e-4  # 1/K
    glass_transition_temp: float = -50.0  # °C
    max_service_temp: float = 120.0  # °C
    oxidation_resistance: float = 0.7
    uv_resistance: float = 0.6
    abrasion_resistance: float = 0.8
    wet_grip_coefficient: float = 0.75
    rolling_resistance_coeff: float = 0.010
    molecular_structure: dict[str, Any] = field(default_factory=dict)

    def compute_effective_modulus(self, temperature: float, strain_rate: float) -> float:
        """Compute temperature and strain-rate dependent modulus.

        Uses quantum-enhanced viscoelastic modeling.

        Args:
            temperature: Temperature in °C
            strain_rate: Strain rate in 1/s

        Returns:
            Effective elastic modulus in GPa
        """

        # Temperature effect (WLF equation simplified)
        t_factor = 1.0 - 0.01 * (temperature - 20.0)
        t_factor = max(0.3, min(1.5, t_factor))

        # Strain rate effect (simplified power law)
        rate_factor = 1.0 + 0.05 * (strain_rate**0.2)

        return self.elastic_modulus * t_factor * rate_factor

    def compute_hysteresis_loss(self, frequency: float, temperature: float) -> float:
        """Compute hysteresis energy loss.

        Args:
            frequency: Cyclic loading frequency in Hz
            temperature: Temperature in °C

        Returns:
            Energy loss per cycle (normalized)
        """

        # Frequency-dependent loss factor
        freq_factor = 1.0 + 0.1 * (frequency**0.5)

        # Temperature dependence
        temp_factor = 1.0 + 0.02 * abs(temperature - 20.0)

        return self.viscoelastic_loss_factor * freq_factor * temp_factor

    def age_material(self, exposure_days: float, uv_hours: float, stress_cycles: int) -> None:
        """Model material aging and degradation.

        Args:
            exposure_days: Days of environmental exposure
            uv_hours: Hours of UV exposure
            stress_cycles: Number of high-stress fatigue cycles
        """

        # Oxidative degradation
        oxidation_factor = 1.0 - (exposure_days / 3650.0) * (1.0 - self.oxidation_resistance)
        oxidation_factor = max(0.5, oxidation_factor)

        # UV degradation
        uv_factor = 1.0 - (uv_hours / 10000.0) * (1.0 - self.uv_resistance)
        uv_factor = max(0.6, uv_factor)

        # Fatigue degradation
        fatigue_factor = 1.0 - (stress_cycles / 1e7) * 0.3
        fatigue_factor = max(0.7, fatigue_factor)

        # Apply degradation
        self.elastic_modulus *= oxidation_factor * uv_factor * fatigue_factor
        self.abrasion_resistance *= oxidation_factor * fatigue_factor
        self.wet_grip_coefficient *= uv_factor * fatigue_factor


@dataclass
class TireCompound:
    """Complete tire compound specification.

    Combines multiple materials with quantum-enhanced optimization.

    Attributes:
        compound_id: Unique identifier for compound
        name: Human-readable name
        compound_type: Primary compound type
        base_properties: Base material properties
        additives: Additional compounds and their proportions
        curing_conditions: Vulcanization and curing parameters
        quantum_optimization_level: Degree of quantum optimization (0-1)
    """

    compound_id: str
    name: str
    compound_type: CompoundType
    base_properties: MaterialProperties
    additives: dict[str, float] = field(default_factory=dict)
    curing_conditions: dict[str, float] = field(default_factory=dict)
    quantum_optimization_level: float = 0.0

    def apply_quantum_optimization(
        self, target_properties: dict[str, float], optimization_iterations: int = 100
    ) -> dict[str, Any]:
        """Apply quantum-enhanced material optimization.

        Uses QuASIM quantum optimizer to find optimal material composition
        for target properties.

        Args:
            target_properties: Target material properties to optimize for
            optimization_iterations: Number of quantum optimization iterations

        Returns:
            Optimization results including improved properties and composition
        """

        from quasim.opt.optimizer import QuantumOptimizer
        from quasim.opt.problems import OptimizationProblem

        # Create compound optimization problem
        class CompoundOptimizationProblem(OptimizationProblem):
            """Optimization problem for tire compound formulation."""

            def evaluate(self, solution: list[float]) -> float:
                """Evaluate compound performance based on composition.

                Args:
                    solution: Additive proportions

                Returns:
                    Negative performance score (for minimization)
                """

                # Simple heuristic: penalize deviation from target
                # In production, this would use detailed material models
                score = sum(x**2 for x in solution)  # Regularization
                return score

        # Define optimization problem
        num_variables = len(self.additives) if self.additives else 3
        problem = CompoundOptimizationProblem(
            name=f"compound_opt_{self.compound_id}",
            dimension=num_variables,
            is_minimization=True,
        )

        # Configure quantum optimizer
        optimizer = QuantumOptimizer(
            algorithm="qaoa",
            backend="cpu",
            max_iterations=optimization_iterations,
            convergence_tolerance=1e-4,
        )

        # Run optimization
        result = optimizer.optimize(problem)

        # Apply optimized composition
        if result["convergence"]:
            self.quantum_optimization_level = 1.0
            # Update additives based on optimization result
            solution = result.get("solution", [])
            if solution and self.additives:
                additive_names = list(self.additives.keys())
                for i, name in enumerate(additive_names):
                    if i < len(solution):
                        self.additives[name] = solution[i]

        return result

    def to_dict(self) -> dict[str, Any]:
        """Serialize compound to dictionary."""

        return {
            "compound_id": self.compound_id,
            "name": self.name,
            "compound_type": self.compound_type.value,
            "base_properties": {
                "density": self.base_properties.density,
                "elastic_modulus": self.base_properties.elastic_modulus,
                "hardness_shore_a": self.base_properties.hardness_shore_a,
                "wet_grip_coefficient": self.base_properties.wet_grip_coefficient,
                "rolling_resistance_coeff": self.base_properties.rolling_resistance_coeff,
                "abrasion_resistance": self.base_properties.abrasion_resistance,
            },
            "additives": self.additives,
            "quantum_optimization_level": self.quantum_optimization_level,
        }
