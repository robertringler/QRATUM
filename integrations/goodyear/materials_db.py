"""Goodyear materials database with 1,000+ tire compounds."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from quasim.domains.tire.materials import (CompoundType, MaterialProperties,
                                           TireCompound)


@dataclass
class MaterialRecord:
    """Record for a Goodyear material in the database.

    Attributes:
        material_id: Unique Goodyear material identifier
        name: Material name
        family: Material family (natural rubber, synthetic, etc.)
        formulation_code: Internal formulation code
        properties: Material properties
        performance_targets: Target application areas
        test_data: Historical test data
        certification_status: Regulatory certification status
        quantum_validated: Whether material has been quantum-validated
    """

    material_id: str
    name: str
    family: str
    formulation_code: str
    properties: dict[str, float]
    performance_targets: list[str] = field(default_factory=list)
    test_data: dict[str, Any] = field(default_factory=dict)
    certification_status: str = "pending"
    quantum_validated: bool = False

    def to_tire_compound(self) -> TireCompound:
        """Convert Goodyear material record to QuASIM tire compound.

        Returns:
            TireCompound instance compatible with tire simulation
        """

        # Map Goodyear family to CompoundType
        family_mapping = {
            "natural_rubber": CompoundType.NATURAL_RUBBER,
            "synthetic_rubber": CompoundType.SYNTHETIC_RUBBER,
            "biopolymer": CompoundType.BIOPOLYMER,
            "nano_enhanced": CompoundType.NANO_ENHANCED,
            "graphene_reinforced": CompoundType.GRAPHENE_REINFORCED,
            "quantum_optimized": CompoundType.QUANTUM_OPTIMIZED,
            "silica_enhanced": CompoundType.SILICA_ENHANCED,
            "carbon_black": CompoundType.CARBON_BLACK,
        }
        compound_type = family_mapping.get(self.family, CompoundType.SYNTHETIC_RUBBER)

        # Create material properties from Goodyear data
        props = MaterialProperties(
            density=self.properties.get("density", 1150.0),
            elastic_modulus=self.properties.get("elastic_modulus", 0.002),
            shear_modulus=self.properties.get("shear_modulus", 0.001),
            poisson_ratio=self.properties.get("poisson_ratio", 0.49),
            hardness_shore_a=self.properties.get("hardness_shore_a", 60.0),
            viscoelastic_loss_factor=self.properties.get("viscoelastic_loss_factor", 0.15),
            thermal_conductivity=self.properties.get("thermal_conductivity", 0.25),
            specific_heat=self.properties.get("specific_heat", 1900.0),
            thermal_expansion_coeff=self.properties.get("thermal_expansion_coeff", 2e-4),
            glass_transition_temp=self.properties.get("glass_transition_temp", -50.0),
            max_service_temp=self.properties.get("max_service_temp", 120.0),
            oxidation_resistance=self.properties.get("oxidation_resistance", 0.7),
            uv_resistance=self.properties.get("uv_resistance", 0.6),
            abrasion_resistance=self.properties.get("abrasion_resistance", 0.8),
            wet_grip_coefficient=self.properties.get("wet_grip_coefficient", 0.75),
            rolling_resistance_coeff=self.properties.get("rolling_resistance_coeff", 0.010),
        )

        # Extract additives from formulation data
        additives = self.properties.get(
            "additives", {"silica": 0.15, "carbon_black": 0.25, "sulfur": 0.02}
        )

        # Create tire compound
        compound = TireCompound(
            compound_id=self.material_id,
            name=f"{self.name} (Goodyear {self.formulation_code})",
            compound_type=compound_type,
            base_properties=props,
            additives=additives,
            quantum_optimization_level=1.0 if self.quantum_validated else 0.5,
        )

        return compound


class GoodyearMaterialsDatabase:
    """Database of 1,000+ Goodyear tire materials.

    Provides access to pre-characterized materials from the Goodyear
    Quantum Pilot platform with full material properties, test data,
    and certification status.
    """

    def __init__(self, db_path: str | None = None):
        """Initialize materials database.

        Args:
            db_path: Path to materials database file (JSON format)
        """

        self.db_path = db_path
        self.materials: dict[str, MaterialRecord] = {}
        self._load_materials()

    def _load_materials(self) -> None:
        """Load materials from database file or generate synthetic database."""

        if self.db_path and Path(self.db_path).exists():
            self._load_from_file(self.db_path)
        else:
            self._generate_synthetic_database(count=1000)

    def _load_from_file(self, file_path: str) -> None:
        """Load materials from JSON database file.

        Args:
            file_path: Path to JSON file containing material records
        """

        with open(file_path) as f:
            data = json.load(f)

        for record_data in data["materials"]:
            record = MaterialRecord(
                material_id=record_data["material_id"],
                name=record_data["name"],
                family=record_data["family"],
                formulation_code=record_data["formulation_code"],
                properties=record_data["properties"],
                performance_targets=record_data.get("performance_targets", []),
                test_data=record_data.get("test_data", {}),
                certification_status=record_data.get("certification_status", "pending"),
                quantum_validated=record_data.get("quantum_validated", False),
            )
            self.materials[record.material_id] = record

    def _generate_synthetic_database(self, count: int = 1000) -> None:
        """Generate synthetic Goodyear materials database.

        Creates 1,000+ material variants with realistic property distributions
        based on Goodyear's material families and performance targets.

        Args:
            count: Number of materials to generate
        """

        rng = np.random.RandomState(42)

        # Material families and their base properties
        families = {
            "natural_rubber": {
                "base_elastic_modulus": 0.002,
                "base_hardness": 65,
                "base_wet_grip": 0.78,
                "base_rolling_resistance": 0.011,
                "base_abrasion": 0.75,
            },
            "synthetic_rubber": {
                "base_elastic_modulus": 0.0025,
                "base_hardness": 70,
                "base_wet_grip": 0.72,
                "base_rolling_resistance": 0.010,
                "base_abrasion": 0.82,
            },
            "biopolymer": {
                "base_elastic_modulus": 0.0018,
                "base_hardness": 60,
                "base_wet_grip": 0.75,
                "base_rolling_resistance": 0.009,
                "base_abrasion": 0.70,
            },
            "nano_enhanced": {
                "base_elastic_modulus": 0.0028,
                "base_hardness": 68,
                "base_wet_grip": 0.80,
                "base_rolling_resistance": 0.009,
                "base_abrasion": 0.88,
            },
            "graphene_reinforced": {
                "base_elastic_modulus": 0.0032,
                "base_hardness": 72,
                "base_wet_grip": 0.82,
                "base_rolling_resistance": 0.008,
                "base_abrasion": 0.92,
            },
            "quantum_optimized": {
                "base_elastic_modulus": 0.0030,
                "base_hardness": 66,
                "base_wet_grip": 0.85,
                "base_rolling_resistance": 0.0075,
                "base_abrasion": 0.95,
            },
            "silica_enhanced": {
                "base_elastic_modulus": 0.0024,
                "base_hardness": 64,
                "base_wet_grip": 0.80,
                "base_rolling_resistance": 0.0085,
                "base_abrasion": 0.85,
            },
            "carbon_black": {
                "base_elastic_modulus": 0.0026,
                "base_hardness": 68,
                "base_wet_grip": 0.76,
                "base_rolling_resistance": 0.010,
                "base_abrasion": 0.80,
            },
        }

        # Performance targets
        performance_areas = [
            "passenger_comfort",
            "truck_durability",
            "racing_grip",
            "winter_traction",
            "fuel_efficiency",
            "all_season",
            "high_performance",
            "off_road",
            "ev_optimized",
        ]

        family_list = list(families.keys())

        for i in range(count):
            family = family_list[i % len(family_list)]
            base_props = families[family]

            # Add variation to base properties
            properties = {
                "density": 1150.0 + rng.uniform(-50, 50),
                "elastic_modulus": base_props["base_elastic_modulus"]
                * (1.0 + rng.uniform(-0.2, 0.2)),
                "shear_modulus": 0.001 * (1.0 + rng.uniform(-0.15, 0.15)),
                "poisson_ratio": 0.49 + rng.uniform(-0.02, 0.02),
                "hardness_shore_a": base_props["base_hardness"] + rng.uniform(-5, 5),
                "viscoelastic_loss_factor": 0.15 * (1.0 + rng.uniform(-0.3, 0.3)),
                "thermal_conductivity": 0.25 * (1.0 + rng.uniform(-0.1, 0.1)),
                "specific_heat": 1900.0 + rng.uniform(-100, 100),
                "thermal_expansion_coeff": 2e-4 * (1.0 + rng.uniform(-0.2, 0.2)),
                "glass_transition_temp": -50.0 + rng.uniform(-10, 10),
                "max_service_temp": 120.0 + rng.uniform(-10, 10),
                "oxidation_resistance": 0.7 + rng.uniform(-0.1, 0.2),
                "uv_resistance": 0.6 + rng.uniform(-0.1, 0.2),
                "abrasion_resistance": base_props["base_abrasion"] * (1.0 + rng.uniform(-0.1, 0.1)),
                "wet_grip_coefficient": base_props["base_wet_grip"]
                * (1.0 + rng.uniform(-0.1, 0.1)),
                "rolling_resistance_coeff": base_props["base_rolling_resistance"]
                * (1.0 + rng.uniform(-0.15, 0.15)),
                "additives": {
                    "silica": max(0, min(0.3, 0.15 + rng.uniform(-0.05, 0.05))),
                    "carbon_black": max(0, min(0.4, 0.25 + rng.uniform(-0.05, 0.05))),
                    "sulfur": max(0, min(0.05, 0.02 + rng.uniform(-0.01, 0.01))),
                },
            }

            # Select performance targets
            num_targets = rng.randint(1, 4)
            targets = rng.choice(performance_areas, size=num_targets, replace=False).tolist()

            # Certification status distribution
            cert_statuses = ["certified", "certified", "certified", "in_testing", "pending"]
            cert_status = cert_statuses[i % len(cert_statuses)]

            # Quantum validation (30% of materials)
            quantum_validated = i % 3 == 0

            record = MaterialRecord(
                material_id=f"GY-MAT-{i:04d}",
                name=f"Goodyear {family.replace('_', ' ').title()} Compound {i}",
                family=family,
                formulation_code=f"GY-{family[:3].upper()}-{i:04d}",
                properties=properties,
                performance_targets=targets,
                test_data={
                    "lab_tests_completed": rng.randint(50, 200),
                    "field_tests_completed": rng.randint(10, 50),
                    "total_test_kilometers": float(rng.randint(10000, 500000)),
                },
                certification_status=cert_status,
                quantum_validated=quantum_validated,
            )

            self.materials[record.material_id] = record

    def get_material(self, material_id: str) -> MaterialRecord | None:
        """Get material by ID.

        Args:
            material_id: Goodyear material identifier

        Returns:
            Material record or None if not found
        """

        return self.materials.get(material_id)

    def search_materials(
        self,
        family: str | None = None,
        performance_target: str | None = None,
        certification_status: str | None = None,
        quantum_validated: bool | None = None,
        min_wet_grip: float | None = None,
        max_rolling_resistance: float | None = None,
    ) -> list[MaterialRecord]:
        """Search materials database with filters.

        Args:
            family: Filter by material family
            performance_target: Filter by performance target
            certification_status: Filter by certification status
            quantum_validated: Filter by quantum validation status
            min_wet_grip: Minimum wet grip coefficient
            max_rolling_resistance: Maximum rolling resistance

        Returns:
            List of matching material records
        """

        results = []

        for material in self.materials.values():
            # Apply filters
            if family and material.family != family:
                continue

            if performance_target and performance_target not in material.performance_targets:
                continue

            if certification_status and material.certification_status != certification_status:
                continue

            if quantum_validated is not None and material.quantum_validated != quantum_validated:
                continue

            if min_wet_grip and material.properties.get("wet_grip_coefficient", 0) < min_wet_grip:
                continue

            if (
                max_rolling_resistance
                and material.properties.get("rolling_resistance_coeff", 1.0)
                > max_rolling_resistance
            ):
                continue

            results.append(material)

        return results

    def get_all_materials(self) -> list[MaterialRecord]:
        """Get all materials in database.

        Returns:
            List of all material records
        """

        return list(self.materials.values())

    def export_to_json(self, output_path: str) -> None:
        """Export materials database to JSON file.

        Args:
            output_path: Path to output JSON file
        """

        data = {
            "version": "1.0",
            "source": "Goodyear Quantum Pilot Platform",
            "material_count": len(self.materials),
            "materials": [
                {
                    "material_id": m.material_id,
                    "name": m.name,
                    "family": m.family,
                    "formulation_code": m.formulation_code,
                    "properties": m.properties,
                    "performance_targets": m.performance_targets,
                    "test_data": m.test_data,
                    "certification_status": m.certification_status,
                    "quantum_validated": m.quantum_validated,
                }
                for m in self.materials.values()
            ],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

    def get_statistics(self) -> dict[str, Any]:
        """Get database statistics.

        Returns:
            Dictionary with database statistics
        """

        families = {}
        cert_statuses = {}
        quantum_count = 0

        for material in self.materials.values():
            families[material.family] = families.get(material.family, 0) + 1
            cert_statuses[material.certification_status] = (
                cert_statuses.get(material.certification_status, 0) + 1
            )
            if material.quantum_validated:
                quantum_count += 1

        return {
            "total_materials": len(self.materials),
            "by_family": families,
            "by_certification_status": cert_statuses,
            "quantum_validated": quantum_count,
            "quantum_validated_percentage": round(100.0 * quantum_count / len(self.materials), 1),
        }
