# Custom Backends

Implement custom quantum backends for QRATUM.

## Backend Interface

All backends must implement the `QuantumBackend` protocol:

```python
from typing import Protocol
from dataclasses import dataclass
import numpy as np

@dataclass
class CircuitResult:
    """Result from circuit execution."""
    counts: dict[str, int]    # Measurement outcomes
    statevector: np.ndarray | None  # State (if available)
    metadata: dict            # Backend-specific metadata

class QuantumBackend(Protocol):
    """Protocol for quantum backends."""
    
    def execute(
        self,
        circuit: "QuantumCircuit",
        shots: int,
        seed: int | None = None
    ) -> CircuitResult:
        """Execute a quantum circuit."""
        ...
    
    def get_backend_info(self) -> dict:
        """Return backend metadata."""
        ...
    
    @property
    def max_qubits(self) -> int:
        """Maximum supported qubit count."""
        ...
```

## Example: Custom Simulator

```python
from quasim.quantum.backend import QuantumBackend, CircuitResult
import numpy as np

class MyCustomBackend(QuantumBackend):
    """Custom quantum simulator backend."""
    
    def __init__(self, precision: str = "double"):
        self.precision = precision
        self._dtype = np.float64 if precision == "double" else np.float32
    
    def execute(
        self,
        circuit: "QuantumCircuit",
        shots: int,
        seed: int | None = None
    ) -> CircuitResult:
        """Execute circuit using custom simulation."""
        if seed is not None:
            np.random.seed(seed)
        
        # Initialize state
        n_qubits = circuit.num_qubits
        state = np.zeros(2**n_qubits, dtype=np.complex128)
        state[0] = 1.0  # |00...0>
        
        # Apply gates
        for instruction in circuit.data:
            state = self._apply_gate(state, instruction)
        
        # Sample measurements
        probs = np.abs(state)**2
        samples = np.random.choice(
            2**n_qubits,
            size=shots,
            p=probs
        )
        
        # Convert to counts
        counts = {}
        for sample in samples:
            bitstring = format(sample, f'0{n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return CircuitResult(
            counts=counts,
            statevector=state,
            metadata={'precision': self.precision}
        )
    
    def _apply_gate(self, state, instruction):
        """Apply a gate to the state vector."""
        # Implementation depends on gate type
        gate = instruction.operation
        qubits = instruction.qubits
        
        if gate.name == 'h':
            return self._apply_hadamard(state, qubits[0].index)
        elif gate.name == 'x':
            return self._apply_x(state, qubits[0].index)
        elif gate.name == 'cx':
            return self._apply_cnot(state, qubits[0].index, qubits[1].index)
        # ... more gates
        
        return state
    
    def _apply_hadamard(self, state, qubit):
        """Apply Hadamard gate."""
        n_qubits = int(np.log2(len(state)))
        h_factor = 1 / np.sqrt(2)
        new_state = np.zeros_like(state)
        
        for i in range(len(state)):
            bit = (i >> qubit) & 1
            j = i ^ (1 << qubit)  # Flip bit
            if bit == 0:
                new_state[i] += h_factor * state[i]
                new_state[j] += h_factor * state[i]
            else:
                new_state[i] += h_factor * state[j]
                new_state[j] -= h_factor * state[j]
        
        return new_state
    
    def get_backend_info(self) -> dict:
        return {
            'name': 'MyCustomBackend',
            'version': '1.0.0',
            'precision': self.precision,
            'max_qubits': 20
        }
    
    @property
    def max_qubits(self) -> int:
        return 20
```

## Registering Custom Backends

```python
from quasim.quantum.registry import register_backend

# Register the backend
register_backend('my_custom', MyCustomBackend)

# Use it
config = QuantumConfig(backend_type='my_custom')
vqe = MolecularVQE(config)
```

## Example: Cloud Backend

```python
import requests

class CloudQuantumBackend(QuantumBackend):
    """Backend that executes on a cloud service."""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
    
    def execute(
        self,
        circuit: "QuantumCircuit",
        shots: int,
        seed: int | None = None
    ) -> CircuitResult:
        # Serialize circuit
        circuit_json = circuit.to_json()
        
        # Submit to cloud
        response = requests.post(
            f"{self.api_url}/execute",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "circuit": circuit_json,
                "shots": shots,
                "seed": seed
            }
        )
        response.raise_for_status()
        
        result = response.json()
        
        return CircuitResult(
            counts=result['counts'],
            statevector=None,  # Not available from cloud
            metadata=result.get('metadata', {})
        )
    
    def get_backend_info(self) -> dict:
        response = requests.get(
            f"{self.api_url}/info",
            headers={"Authorization": f"Bearer {self.api_key}"}
        )
        return response.json()
    
    @property
    def max_qubits(self) -> int:
        info = self.get_backend_info()
        return info.get('max_qubits', 127)
```

## Example: GPU Backend with cuQuantum

```python
try:
    import cuquantum
    CUQUANTUM_AVAILABLE = True
except ImportError:
    CUQUANTUM_AVAILABLE = False

class CuQuantumBackend(QuantumBackend):
    """NVIDIA cuQuantum GPU-accelerated backend."""
    
    def __init__(self, device: int = 0):
        if not CUQUANTUM_AVAILABLE:
            raise ImportError("cuQuantum not available")
        
        self.device = device
        import cupy as cp
        cp.cuda.Device(device).use()
    
    def execute(
        self,
        circuit: "QuantumCircuit",
        shots: int,
        seed: int | None = None
    ) -> CircuitResult:
        import cupy as cp
        from cuquantum import CircuitToEinsum
        
        if seed is not None:
            cp.random.seed(seed)
        
        # Convert circuit to tensor network
        converter = CircuitToEinsum(circuit)
        
        # Contract tensor network on GPU
        statevector = converter.state_vector()
        
        # Sample on GPU
        probs = cp.abs(statevector)**2
        samples = cp.random.choice(
            len(statevector),
            size=shots,
            p=probs
        )
        
        # Convert to counts (move to CPU)
        counts = {}
        n_qubits = circuit.num_qubits
        for sample in samples.get():
            bitstring = format(int(sample), f'0{n_qubits}b')
            counts[bitstring] = counts.get(bitstring, 0) + 1
        
        return CircuitResult(
            counts=counts,
            statevector=statevector.get(),
            metadata={'device': self.device}
        )
    
    def get_backend_info(self) -> dict:
        import cupy as cp
        device = cp.cuda.Device(self.device)
        return {
            'name': 'CuQuantumBackend',
            'device': self.device,
            'gpu_name': device.name,
            'gpu_memory': device.mem_info[1]
        }
    
    @property
    def max_qubits(self) -> int:
        # Limited by GPU memory
        import cupy as cp
        device = cp.cuda.Device(self.device)
        memory_gb = device.mem_info[1] / (1024**3)
        # Rough estimate: 16 bytes per amplitude
        return int(np.log2(memory_gb * (1024**3) / 16))
```

## Testing Custom Backends

```python
import pytest
from quasim.quantum.backend import CircuitResult

class TestMyCustomBackend:
    
    def test_hadamard(self):
        backend = MyCustomBackend()
        circuit = QuantumCircuit(1)
        circuit.h(0)
        circuit.measure_all()
        
        result = backend.execute(circuit, shots=1000, seed=42)
        
        # Should get ~50% 0 and ~50% 1
        assert '0' in result.counts
        assert '1' in result.counts
        assert abs(result.counts.get('0', 0) - 500) < 100
    
    def test_bell_state(self):
        backend = MyCustomBackend()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        result = backend.execute(circuit, shots=1000, seed=42)
        
        # Should get ~50% 00 and ~50% 11
        assert result.counts.get('00', 0) + result.counts.get('11', 0) > 900
    
    def test_deterministic_with_seed(self):
        backend = MyCustomBackend()
        circuit = QuantumCircuit(2)
        circuit.h(0)
        circuit.cx(0, 1)
        circuit.measure_all()
        
        result1 = backend.execute(circuit, shots=100, seed=42)
        result2 = backend.execute(circuit, shots=100, seed=42)
        
        assert result1.counts == result2.counts
```

## Backend Configuration

```yaml
# config/backends.yaml
backends:
  my_custom:
    class: mymodule.MyCustomBackend
    options:
      precision: double
      
  cloud:
    class: mymodule.CloudQuantumBackend
    options:
      api_url: https://quantum.example.com
      api_key: ${QUANTUM_API_KEY}
      
  gpu:
    class: mymodule.CuQuantumBackend
    options:
      device: 0
```

## Next Steps

- [Performance Tuning](performance-tuning.md) - Optimize your backend
- [API Reference](../reference/api-reference.md) - Full API documentation
- [Contributing](../contributing.md) - Contribute your backend
