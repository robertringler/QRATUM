"""Domain scenario templates."""
from qscenario.domains.aerospace import aerospace_launch_anomaly
from qscenario.domains.energy import energy_grid_instability
from qscenario.domains.finance import finance_liquidity_crunch
from qscenario.domains.network import network_partition
from qscenario.domains.pharma import pharma_trial_outcome

__all__ = [
    "aerospace_launch_anomaly",
    "finance_liquidity_crunch",
    "energy_grid_instability",
    "network_partition",
    "pharma_trial_outcome",
]
