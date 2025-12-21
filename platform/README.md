# QRATUM Sovereign AI Platform v2.0

A unified multi-vertical AI platform with cryptographic audit trails, deterministic execution, and domain-specific safety controls.

## Overview

The QRATUM Sovereign AI Platform extends QRATUM core with a unified vertical module system designed for production-grade rigor, determinism, auditability, and full QRATUM integration.

## Architecture

### Core Components

1. **PlatformIntent**: Universal intent structure for all verticals
2. **PlatformContract**: Immutable execution authorization with cryptographic commitment
3. **MerkleEventChain**: Cryptographic audit trail for all operations
4. **VerticalModuleBase**: Abstract base class enforcing 8 fatal invariants
5. **QRATUMPlatform**: Central orchestrator routing intents to modules
6. **ComputeSubstrate**: Optimal hardware selection based on task characteristics

### Eight Fatal Invariants

All vertical modules enforce these invariants:

1. **Deterministic Execution**: All computations must be reproducible
2. **Contract Validation**: Every operation validates contract before execution
3. **Full Event Emission**: Every step emits events to Merkle chain
4. **Safety Disclaimers**: Present in all outputs
5. **Prohibited Uses**: Explicitly checked and refused
6. **Compliance Attestations**: Validated before execution
7. **Optimal Substrate Selection**: Task-specific hardware mapping
8. **Complete Audit Trail**: Results include full event chain

## Vertical Modules

### 1. JURIS - Legal AI
**Capabilities:**
- Contract analysis and review
- Legal research and precedent matching
- Compliance checking
- Regulatory interpretation
- Risk assessment

**Safety Disclaimer:** NOT legal advice - requires attorney review

### 2. VITRA - Bioinformatics & Drug Discovery
**Capabilities:**
- Genomic sequence analysis
- Protein structure prediction
- Drug-target interaction modeling
- Molecular dynamics simulation
- Clinical trial optimization
- Pharmacokinetics modeling

**Safety Disclaimer:** NOT for clinical diagnosis - requires researcher validation

### 3. ECORA - Climate & Energy Systems
**Capabilities:**
- Climate projection modeling
- Renewable energy optimization
- Carbon footprint analysis
- Grid stability simulation
- Weather prediction

**Safety Disclaimer:** Models are approximations - requires expert review

### 4. CAPRA - Financial Risk & Derivatives
**Capabilities:**
- Options pricing (Black-Scholes, Monte Carlo)
- VaR/CVaR risk calculation
- Portfolio optimization
- Credit risk modeling
- Stress testing

**Safety Disclaimer:** NOT investment advice - requires financial advisor

### 5. SENTRA - Aerospace & Defense
**Capabilities:**
- Trajectory simulation
- Radar cross-section analysis
- Orbit propagation
- Aerodynamic analysis
- Mission planning

**Safety Disclaimer:** Export controls may apply - requires authorization

### 6. NEURA - Neuroscience & BCI
**Capabilities:**
- Neural network simulation
- EEG/MEG signal analysis
- Brain connectivity mapping
- BCI signal processing
- Cognitive modeling

**Safety Disclaimer:** NOT for clinical use - requires IRB approval

### 7. FLUXA - Supply Chain & Logistics
**Capabilities:**
- Route optimization
- Demand forecasting
- Inventory optimization
- Supplier risk analysis
- Network resilience assessment

**Safety Disclaimer:** Optimization results require operational validation

## Installation

```bash
# Install QRATUM platform
pip install -e .

# Or install with specific dependencies
pip install -e ".[quantum,dev,viz]"
```

## Quick Start

### Basic Usage

```python
from platform.core.intent import PlatformIntent, VerticalType
from platform.core.orchestrator import QRATUMPlatform
from platform.verticals.juris import JurisModule

# Initialize platform
platform = QRATUMPlatform(seed=42)

# Register vertical modules
juris = JurisModule(seed=42)
platform.register_module(VerticalType.JURIS, juris)

# Create and execute intent
intent = PlatformIntent(
    vertical=VerticalType.JURIS,
    operation="analyze_contract",
    parameters={
        "contract_text": "Your contract text here...",
        "jurisdiction": "California"
    },
    compliance_attestations=frozenset([
        "not_legal_advice_acknowledged",
        "attorney_review_required",
        "contract_jurisdiction_known"
    ])
)

# Execute through platform
result = platform.execute_intent(intent)

print(f"Success: {result.success}")
print(f"Safety Disclaimer: {result.safety_disclaimer}")
print(f"Audit Chain Root: {result.event_chain_root}")
```

### Using Individual Modules

```python
from platform.verticals.capra import CapraModule
from platform.core.intent import PlatformIntent, PlatformContract, VerticalType

# Initialize module
capra = CapraModule(seed=42)

# Create intent
intent = PlatformIntent(
    vertical=VerticalType.CAPRA,
    operation="price_option",
    parameters={
        "type": "call",
        "spot_price": 100.0,
        "strike_price": 105.0,
        "time_to_expiry_years": 0.5,
        "volatility": 0.25,
        "risk_free_rate": 0.04,
        "method": "black_scholes"
    },
    compliance_attestations=frozenset([
        "not_investment_advice",
        "financial_advisor_consultation",
        "risk_disclosure_acknowledged"
    ])
)

# Create and execute contract
import hashlib
contract = PlatformContract(
    intent=intent,
    contract_hash=hashlib.sha256(str(intent).encode()).hexdigest(),
    authorized=True
)

result = capra.execute(contract)
print(f"Option Price: ${result.result_data['option_price']:.2f}")
```

## Examples

See the `platform/examples/` directory for QIL (Quantum Intent Language) examples for each vertical:

- `juris_examples.qil` - Legal AI examples
- `vitra_examples.qil` - Bioinformatics examples
- `ecora_examples.qil` - Climate & Energy examples
- `capra_examples.qil` - Financial Risk examples
- `sentra_examples.qil` - Aerospace & Defense examples
- `neura_examples.qil` - Neuroscience & BCI examples
- `fluxa_examples.qil` - Supply Chain examples

## Testing

```bash
# Run platform core tests
python3 -c "
import sys
sys.path.insert(0, '.')
# Import and run tests
from platform.tests.test_platform import *
# Manual test execution
"

# Test all verticals
python3 -c "
import sys
sys.path.insert(0, '.')
from platform.core.orchestrator import QRATUMPlatform
from platform.core.intent import VerticalType
# ... test code
"
```

## Compliance and Safety

### Required Attestations

Each vertical module requires specific compliance attestations before execution. These must be provided in the `compliance_attestations` field of the intent.

### Safety Disclaimers

Every execution result includes a domain-specific safety disclaimer. These disclaimers are immutable and cannot be bypassed.

### Prohibited Uses

Each module explicitly checks for and refuses prohibited uses. Attempting prohibited operations will result in execution failure with audit trail.

### Audit Trail

All operations emit events to a cryptographic Merkle chain, providing:
- Complete execution history
- Tamper detection
- Cryptographic verification
- Compliance documentation

## API Reference

### Core Classes

#### PlatformIntent
Immutable intent structure for all operations.

**Fields:**
- `vertical`: Target vertical module (VerticalType)
- `operation`: Specific operation within vertical
- `parameters`: Operation parameters (Dict)
- `compliance_attestations`: Required compliance confirmations (FrozenSet)
- `user_id`: User identifier for audit trail
- `session_id`: Session identifier for grouping operations
- `timestamp`: Intent creation timestamp (UTC)

#### PlatformContract
Immutable execution authorization contract.

**Fields:**
- `intent`: The original platform intent
- `contract_hash`: SHA-256 hash of intent serialization
- `authorized`: Whether execution is authorized
- `authorization_timestamp`: When authorization was granted
- `expiry_timestamp`: When authorization expires
- `restrictions`: Any restrictions on execution

**Methods:**
- `is_valid()`: Check if contract is currently valid
- `has_attestation(attestation)`: Check if attestation is present

#### ExecutionResult
Result of a vertical module execution.

**Fields:**
- `success`: Whether execution succeeded
- `result_data`: The actual result data
- `warnings`: Any warnings generated
- `safety_disclaimer`: Required safety disclaimer
- `event_chain_root`: Merkle root of event chain
- `execution_time_ms`: Execution time in milliseconds
- `substrate_used`: Compute substrate that was used

## Development

### Adding a New Vertical

1. Create new module file in `platform/verticals/`
2. Inherit from `VerticalModuleBase`
3. Implement required abstract methods:
   - `get_safety_disclaimer()`
   - `get_prohibited_uses()`
   - `get_required_attestations(operation)`
   - `_execute_operation(contract, substrate)`
4. Ensure all operations are deterministic
5. Emit events for all computation steps
6. Add tests in `platform/tests/`

### Substrate Selection

The platform automatically selects optimal compute substrates based on:
- Problem size
- Task type
- Required availability
- Substrate capabilities

Available substrates:
- CPU_SERIAL, CPU_PARALLEL
- GPU_CUDA, GPU_METAL
- TPU
- QUANTUM_SIM, QUANTUM_HW
- DISTRIBUTED
- NEUROMORPHIC

## Requirements

- Python 3.12+
- NumPy (for numerical operations)
- Frozen dataclasses (immutability)
- Thread-safe event chains
- QRATUM core v1.2 integration

## License

Apache-2.0

## Citation

If you use QRATUM Sovereign AI Platform in your research, please cite:

```bibtex
@software{qratum_platform_v2,
  title = {QRATUM Sovereign AI Platform v2.0},
  author = {QRATUM Team},
  year = {2024},
  version = {2.0.0}
}
```

## Support

For issues, questions, or contributions, please visit the QRATUM repository.
