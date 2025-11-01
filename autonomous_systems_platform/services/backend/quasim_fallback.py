"""Fallback implementation when JAX is not available."""
import random

def autonomous_systems_kernel(seed: int = 0, scale: float = 1.0):
    random.seed(seed)
    state = [random.gauss(0, scale) for _ in range(10)]
    energy = sum(x**2 for x in state)
    convergence = sum(abs(x) for x in state) / len(state)
    return {
        "state_vector": state,
        "energy": energy,
        "convergence": convergence,
    }
