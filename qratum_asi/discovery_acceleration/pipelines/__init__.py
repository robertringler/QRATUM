"""Discovery Acceleration Workflow Pipelines.

Dedicated workflow implementations for each of the 6 discovery types:
1. Personalized Drug Design (Discovery 2)
2. Climate-Gene Connections (Discovery 3)
3. Natural Compound Discovery (Discovery 4)
4. Economic-Biological Models (Discovery 5)
5. Anti-Aging/Longevity Pathways (Discovery 6)

Note: Discovery 1 (Complex Disease Genetics) is implemented
in federated_gwas.py at the module root.

Version: 1.0.0
Status: Production Ready
QuASIM: v2025.12.26
"""

from qratum_asi.discovery_acceleration.pipelines.climate_gene import (
    ClimateGenePipeline,
)
from qratum_asi.discovery_acceleration.pipelines.economic_bio import (
    EconomicBioPipeline,
)
from qratum_asi.discovery_acceleration.pipelines.longevity import (
    LongevityPipeline,
)
from qratum_asi.discovery_acceleration.pipelines.natural_compound import (
    NaturalCompoundPipeline,
)
from qratum_asi.discovery_acceleration.pipelines.personalized_drug import (
    PersonalizedDrugPipeline,
)

__all__ = [
    "PersonalizedDrugPipeline",
    "ClimateGenePipeline",
    "NaturalCompoundPipeline",
    "EconomicBioPipeline",
    "LongevityPipeline",
]
