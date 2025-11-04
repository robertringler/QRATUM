"""QuASIM kernel for autonomous systems simulation."""

import random as py_random

try:
    import jax.numpy as jnp
    from jax import random as jax_random
    JAX_AVAILABLE = True
except (ImportError, AttributeError, RuntimeError):
    JAX_AVAILABLE = False


def autonomous_systems_kernel(seed: int = 0, scale: float = 1.0):
    """Simulated autonomous systems kernel."""
    if JAX_AVAILABLE:
        try:
            key = jax_random.PRNGKey(seed)
            state = jax_random.normal(key, (10,)) * scale
            result = {
                "state_vector": state.tolist(),
                "energy": float(jnp.sum(state**2)),
                "convergence": float(jnp.mean(jnp.abs(state))),
            }
            return result
        except (ValueError, AttributeError, RuntimeError):
            # JAX import succeeded but runtime failed, fall back
            pass

    # Fallback implementation
    py_random.seed(seed)
    state = [py_random.gauss(0, scale) for _ in range(10)]
    energy = sum(x**2 for x in state)
    convergence = sum(abs(x) for x in state) / len(state)
    return {
        "state_vector": state,
        "energy": energy,
        "convergence": convergence,
    }
