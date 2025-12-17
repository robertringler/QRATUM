"""QuASIM quantum simulation utilities (PLACEHOLDER - NOT REAL QUANTUM SIMULATION).

WARNING: Despite the name, this does NOT perform quantum circuit simulation.
The Runtime.simulate() method just averages complex numbers.

This is a placeholder showing the intended API structure for when actual
quantum simulation is implemented using proper quantum frameworks.

TODO: Replace with genuine quantum circuit simulation:
- Use Qiskit Aer for state vector simulation
- Or use Qiskit Runtime for cloud execution
- Implement proper gate application and state evolution
- Add measurement and sampling capabilities
"""

from __future__ import annotations

from quasim import Config, runtime


def simulate(circuit, *, precision: str = "fp8"):
    """PLACEHOLDER: Not real quantum simulation.

    Current behavior: Averages complex numbers (see quasim/__init__.py Runtime.simulate).

    Real quantum simulation would:
    - Apply quantum gates sequentially to state vector
    - Maintain quantum superposition and entanglement
    - Support measurement and sampling
    - Use tensor network contraction or state vector simulation

    Args:
        circuit: "Circuit" specification (not actually used as quantum gates)
        precision: Precision level (currently ignored)

    Returns:
        List of complex numbers (NOT quantum amplitudes)
    """
    cfg = Config(simulation_precision=precision, max_workspace_mb=1024)
    tensors = [[complex(value) for value in gate] for gate in circuit]
    with runtime(cfg) as rt:
        return rt.simulate(tensors)  # Just averages values, not quantum simulation
