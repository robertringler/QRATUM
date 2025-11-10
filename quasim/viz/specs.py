# quasim/viz/specs.py
# Data specifications for QuASIM flow visualization

from dataclasses import dataclass


@dataclass
class FlowFrameSpec:
    """
    Specification for a single frame of the QuASIM flow visualization.

    Attributes:
        frame_idx: Frame index in the sequence
        time: Current time in the simulation
        control: Control parameter value a(t) at this time
        objective: Current objective function value
        w2: Wasserstein-2 distance
        fr_speed: Fisher-Rao speed
        bures_dist: Bures distance
        qfi: Quantum Fisher Information
        fidelity: Fidelity value
        free_energy: Free energy value
    """
    frame_idx: int
    time: float
    control: float
    objective: float
    w2: float
    fr_speed: float
    bures_dist: float
    qfi: float
    fidelity: float
    free_energy: float
