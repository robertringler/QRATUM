from __future__ import annotations

from typing import Any, Callable



class SimulationEngine:
    """Deterministic simulation engine supporting named kernels and circuits."""

    def __init__(self):
        self._steps: list[Callable[[Any], Any]] = []
        self._kernels: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}

    def add_step(self, fn: Callable[[Any], Any]) -> None:
        self._steps.append(fn)

    def register_kernel(self, name: str, fn: Callable[[dict[str, Any]], dict[str, Any]]) -> None:
        self._kernels[name] = fn

    def run(self, seed_state: Any) -> list[Any]:
        state = seed_state
        outputs: list[Any] = []
        for step in self._steps:
            state = step(state)
            outputs.append(state)
        return outputs

    def run_kernel(self, name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if name not in self._kernels:
            raise ValueError(f"kernel {name} not registered")
        return self._kernels[name](payload)

    def evaluate_circuit(self, gates: list[tuple[str, float]], vector: list[float]) -> list[float]:
        """Simple deterministic circuit evaluation: sequential scalar ops."""
        output = list(vector)
        for gate, weight in gates:
            output = [v * weight if gate == "scale" else v + weight for v in output]
        return output

    def mlir_lower(self, operations: list[str]) -> str:
        """Deterministic MLIR-like lowering stub."""
        return "\n".join(f"op_{idx}:{op}" for idx, op in enumerate(operations))

    def telemetry_kernel(self, telemetry: dict[str, float]) -> dict[str, float]:
        altitude = telemetry.get("altitude", 0.0)
        velocity = telemetry.get("velocity", 0.0)
        energy = 0.5 * velocity * velocity
        return {"altitude": altitude + velocity * 0.1, "velocity": velocity, "energy": energy}

    def finance_kernel(self, snapshot: dict[str, float]) -> dict[str, float]:
        price = snapshot.get("price", 0.0)
        shock = snapshot.get("shock", 0.0)
        liquidity = snapshot.get("liquidity", 1.0)
        adjusted = price * (1.0 + shock) * (1.0 - 0.1 * (1.0 - liquidity))
        return {"price": adjusted, "liquidity": liquidity}

    def pharma_kernel(self, species: dict[str, float]) -> dict[str, float]:
        # Simple reaction: A -> B with deterministic rate k
        k = 0.05
        a = species.get("A", 0.0)
        b = species.get("B", 0.0)
        delta = k * a
        return {"A": a - delta, "B": b + delta}


def build_default_engine() -> SimulationEngine:
    engine = SimulationEngine()
    engine.register_kernel("telemetry", engine.telemetry_kernel)
    engine.register_kernel("finance", engine.finance_kernel)
    engine.register_kernel("pharma", engine.pharma_kernel)
    return engine
