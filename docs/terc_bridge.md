# TERC Bridge Integration

## Overview

The TERC Bridge connects REVULTRA and QGH algorithms to TERC (Tiered Evaluation & Robustness Certification) validation tiers, exposing instrumentation hooks for observable extraction and validation.

## Architecture

```
REVULTRA/QGH Algorithms
         ↓
  TERC Observables
         ↓
  Observable Registry
         ↓
TERC Tier-1 / Tier-5 Validation
```

## Observable Types

### 1. Beta Metrics (from REVULTRA)

Synthetic state surrogates for TERC validation when quantum observables unavailable.

**Metrics**:

- `beta_entropy`: Normalized character entropy (0-1)
- `beta_complexity`: Normalized complexity score (0-1)
- `beta_coherence`: Inverse IoC variance (coherence proxy)
- `beta_periodicity`: Strength of detected periodicities (0-1)

**Usage**:

```python
from quasim.terc_bridge import beta_metrics_from_cipher

text = "CRYPTOTEXT" * 20
metrics = beta_metrics_from_cipher(text)

print(f"Entropy: {metrics['beta_entropy']:.3f}")
print(f"Complexity: {metrics['beta_complexity']:.3f}")
print(f"Coherence: {metrics['beta_coherence']:.3f}")
```

**TERC Mapping**:

- Tier-1 State Coherence: `beta_entropy`, `beta_coherence`
- Tier-5 Robustness: `beta_complexity`, `beta_periodicity`

### 2. IoC Period Candidates

Detects candidate periods for polyalphabetic ciphers.

**Usage**:

```python
from quasim.terc_bridge import ioc_period_candidates

text = "VIGENERETEXT" * 10
periods = ioc_period_candidates(text, max_period=20, threshold=0.2)

print(f"Candidate periods: {periods}")
```

**TERC Application**: Validate periodic structure consistency.

### 3. Emergent Complexity

Overall complexity assessment with components.

**Usage**:

```python
from quasim.terc_bridge import emergent_complexity

complexity = emergent_complexity(text)
print(f"Score: {complexity['score']:.2f}")
print(f"Components: entropy={complexity['entropy']:.2f}, "
      f"pattern_density={complexity['pattern_density']:.3f}")
```

**TERC Application**: Assess system state complexity for robustness tests.

### 4. QGH Consensus Status

Convergence and stability metrics from distributed consensus.

**Usage**:

```python
from quasim.terc_bridge import qgh_consensus_status
import numpy as np

# Simulate distributed nodes
states = np.random.rand(10, 5)
status = qgh_consensus_status(states, damping=0.5, max_iterations=100)

print(f"Converged: {status['converged']}")
print(f"Stability: {status['stability']:.3f}")
print(f"Robustness: {status['robustness']:.3f}")
```

**TERC Mapping**:

- Tier-1 Consensus: `converged`, `iterations`
- Tier-5 Robustness: `stability`, `robustness`, `final_variance`

### 5. Stream Synchronization

Detects synchronization patterns in distributed streams.

**Usage**:

```python
from quasim.terc_bridge import stream_synchronization_metrics

stream_data = {
    0: [1.0, 1.2, 0.9, ...],
    1: [1.1, 1.3, 0.8, ...],
    2: [5.0, 5.1, 5.2, ...]
}

metrics = stream_synchronization_metrics(stream_data, threshold=0.7)
print(f"Sync ratio: {metrics['sync_ratio']:.2f}")
print(f"Synchronized: {metrics['synchronized']}")
```

**TERC Application**: Validate distributed system coherence.

### 6. Stability Assessment

Assesses stability of metric time series.

**Usage**:

```python
from quasim.terc_bridge import stability_assessment

metrics = [1.0, 1.1, 0.9, 1.2, ...]  # Time series
assessment = stability_assessment(metrics, window_size=50)

print(f"Stable: {assessment['is_stable']}")
print(f"Trend: {assessment['trend']:.4f}")
```

**TERC Application**: Monitor system divergence, detect instabilities.

## Observable Registry

Centralized registry for observable functions.

**Auto-Registration**:

```python
from quasim.terc_bridge.registry import list_observables, compute_observable

# List all registered observables
print(list_observables())
# ['beta_metrics', 'ioc_periods', 'emergent_complexity', 
#  'consensus_status', 'stream_sync', 'stability']

# Compute by name
result = compute_observable('beta_metrics', text="EXAMPLE")
```

**Manual Registration**:

```python
from quasim.terc_bridge.registry import register_observable

def my_custom_observable(data):
    return {"custom_metric": len(data)}

register_observable("my_observable", my_custom_observable)
```

## CLI Usage

### Emit Observables

```bash
# From text
quasim-terc-obs emit --text "ATTACKATDAWN" --out observables.json

# From state file
quasim-terc-obs emit --state-file quasim_state.json --out obs.json

# Specific observable
quasim-terc-obs emit --text "CRYPTO" --out obs.json --observable beta_metrics
```

### List Registered Observables

```bash
quasim-terc-obs list
```

### Compute Consensus

```bash
quasim-terc-obs consensus --num-nodes 10 --state-dim 5 --out consensus.json
```

### Validate Observable File

```bash
quasim-terc-obs validate --obs-file observables.json
```

## JSON Output Schema

### Standard Format

```json
{
  "observables": {
    "beta_metrics": {
      "beta_entropy": 0.745,
      "beta_complexity": 0.523,
      "beta_coherence": 0.812,
      "beta_periodicity": 0.234
    },
    "emergent_complexity": {
      "score": 45.2,
      "entropy": 3.14,
      "pattern_density": 0.15,
      "ioc_variance": 0.023
    }
  },
  "format_version": "1.0",
  "source": "quasim-revultra-qgh"
}
```

### Consensus Status

```json
{
  "observables": {
    "consensus_status": {
      "converged": true,
      "iterations": 23,
      "final_variance": 0.0012,
      "stability": 0.945,
      "robustness": 0.770
    }
  },
  "format_version": "1.0",
  "source": "quasim-revultra-qgh"
}
```

## Adapters

### QuASIM State Adaptation

Adapters convert QuASIM agent states to algorithm inputs.

**Example**:

```python
from quasim.terc_bridge.adapters import from_quasim_state

# From file
state = from_quasim_state("quasim_state.json")

# From dictionary
state_dict = {"trajectory": "ABCDEF", "type": "text"}
text = from_quasim_state(state_dict)

# From path (string)
text = from_quasim_state("path/to/state.json")
```

**Supported Formats**:

- Text trajectories: `{"trajectory": "..."}` → string
- State vectors: `{"state_vector": [...]}` → np.ndarray
- Tensor data: `{"tensor": [...]}` → np.ndarray

### TERC Format Conversion

```python
from quasim.terc_bridge.adapters import to_terc_observable_format

results = {"metric1": 1.5, "metric2": [1, 2, 3]}
formatted = to_terc_observable_format(results)

# Output:
# {
#   "observables": {...},
#   "format_version": "1.0",
#   "source": "quasim-revultra-qgh"
# }
```

## Integration with TERC Tiers

### Tier-1: Initial State Validation

Uses:

- Beta entropy/coherence → State coherence checks
- IoC period candidates → Periodic structure validation
- Consensus status → Distributed initialization

### Tier-5: Robustness Testing

Uses:

- Beta complexity → Overall robustness metric
- Stability assessment → Long-term stability
- Stream synchronization → Distributed coherence

### Example Pipeline

```python
from quasim.terc_bridge import (
    beta_metrics_from_cipher,
    emergent_complexity,
    qgh_consensus_status
)

# Tier-1 validation
text_state = "QUANTUMSTATE" * 20
tier1_obs = {
    "beta": beta_metrics_from_cipher(text_state),
    "complexity": emergent_complexity(text_state)
}

# Tier-5 validation
node_states = np.random.rand(20, 10)
tier5_obs = {
    "consensus": qgh_consensus_status(node_states)
}

# Validate
if tier1_obs["beta"]["beta_coherence"] > 0.7:
    print("✓ Tier-1 passed: State coherence acceptable")

if tier5_obs["consensus"]["stability"] > 0.8:
    print("✓ Tier-5 passed: System stable")
```

## Runtime Integration

Register observables with QuASIM runtime:

```python
from quasim.terc_bridge.registry import register_observables

# Auto-register all standard observables
register_observables(runtime=None)

# Or with runtime object (future)
# register_observables(runtime=quasim_runtime)
```

## Custom Observables

Define and register custom observables:

```python
from quasim.terc_bridge.registry import register_observable

def my_metric(text: str) -> dict:
    """Custom metric for specific use case."""
    return {
        "length": len(text),
        "unique_chars": len(set(text)),
        "ratio": len(set(text)) / len(text)
    }

register_observable("my_metric", my_metric)

# Use via registry
from quasim.terc_bridge.registry import compute_observable
result = compute_observable("my_metric", text="EXAMPLE")
```

## Performance Considerations

- **Beta metrics**: O(n log n) for FFT operations
- **Consensus status**: O(nodes² * iterations) for propagation
- **Stream sync**: O(streams² * buffer_size) for correlation
- **Stability**: O(window_size) for trend fitting

For large-scale deployments:

- Use smaller max_period for IoC (default: 20)
- Limit buffer sizes for streams (default: 1000)
- Set reasonable max_iterations for consensus (default: 100)

## Testing

All observables have comprehensive tests:

```bash
pytest tests/test_terc_bridge_observables.py -v
```

Coverage: 100% for core observable functions.
