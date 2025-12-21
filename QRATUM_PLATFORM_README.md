# QRATUM Sovereign AI Platform v2.0 - Core Vertical Modules

## Overview

The QRATUM Sovereign AI Platform provides 14 specialized vertical AI modules that execute as deterministic, auditable QRADLE contracts on the Frankenstein Cluster. This implementation includes the first 7 vertical modules with complete safety controls, event tracking, and Merkle chain audit trails.

## Architecture

### Core Platform (`qratum_platform/`)

- **`core.py`** - Foundational classes and types
  - `VerticalModule` enum (14 verticals)
  - `ComputeSubstrate` enum (7 hardware types)
  - `PlatformIntent` - Immutable intent structure
  - `PlatformContract` - Frozen execution contract
  - `ExecutionEvent` - Audit trail events
  - `MerkleEventChain` - Append-only event chain
  - `VerticalModuleBase` - Abstract module base class
  - `QRATUMPlatform` - Main orchestrator

- **`substrates.py`** - Hardware mapping for optimal execution
- **`utils.py`** - Cryptographic utilities

### Vertical Modules (`verticals/`)

#### 1. JURIS - Legal AI
**Capabilities:**
- IRAC legal reasoning (Issue, Rule, Application, Conclusion)
- Contract analysis with risk identification
- Litigation outcome prediction
- Compliance checking (GDPR, CCPA, HIPAA, SOX)

**Safety:** Not a substitute for licensed legal counsel

#### 2. VITRA - Bioinformatics
**Capabilities:**
- DNA/RNA/Protein sequence analysis
- Codon translation and ORF detection
- Protein structure prediction
- Drug candidate screening
- Molecular dynamics simulation
- Pharmacokinetics (ADME) modeling

**Safety:** Research use only - not for clinical diagnostics

#### 3. ECORA - Climate & Energy
**Capabilities:**
- Climate projection with SSP scenarios
- Energy grid optimization
- Carbon footprint analysis
- Weather prediction
- Renewable site assessment

**Safety:** Projections subject to uncertainty

#### 4. CAPRA - Financial Risk
**Capabilities:**
- Black-Scholes option pricing with Greeks
- Monte Carlo simulation
- Value at Risk (VaR) and CVaR calculation
- Portfolio optimization
- Credit risk analysis
- Stress testing

**Safety:** Not investment advice - consult licensed advisors

#### 5. SENTRA - Aerospace & Defense
**Capabilities:**
- Ballistic trajectory simulation
- Radar cross-section (RCS) analysis
- Two-body orbital propagation
- Aerodynamic analysis
- Mission planning

**Safety:** Research only - export controls may apply (ITAR/EAR)

#### 6. NEURA - Neuroscience & BCI
**Capabilities:**
- Spiking neural network simulation
- EEG/MEG signal analysis
- Brain connectivity mapping
- BCI signal processing
- Cognitive modeling

**Safety:** Research only - IRB approval required

#### 7. FLUXA - Supply Chain
**Capabilities:**
- Vehicle routing optimization (VRP/TSP)
- Demand forecasting
- Inventory optimization (EOQ)
- Network resilience analysis
- Discrete event logistics simulation

**Safety:** Validate recommendations before deployment

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/robertringler/QRATUM
cd QRATUM

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/test_platform.py tests/test_*.py -v
```

### Basic Usage

```python
from qratum_platform.core import QRATUMPlatform, PlatformIntent, VerticalModule
from verticals import JURISModule

# Create platform and register module
platform = QRATUMPlatform()
juris = JURISModule()
platform.register_module(VerticalModule.JURIS, juris)

# Create intent
intent = PlatformIntent(
    vertical=VerticalModule.JURIS,
    operation="legal_reasoning",
    parameters={
        "facts": "Party A breached contract with Party B",
        "area_of_law": "contract"
    },
    user_id="user_001"
)

# Execute
contract = platform.create_contract(intent)
result = platform.execute_contract(contract.contract_id)

print(result['conclusion'])
```

### Run Demo

```bash
python demo_qratum_platform.py
```

## QRATUM Fatal Invariants

The platform enforces these invariants:

1. **Contract Immutability** - Frozen dataclasses prevent modification
2. **Zero Policy Logic in Adapters** - All logic in vertical modules
3. **Mandatory Event Emission** - Every operation emits events
4. **Hash-Chain Integrity** - Merkle chain for audit trail
5. **Causal Traceability** - Events linked via previous_hash
6. **Authorized Execution Only** - Safety checks before execution
7. **Deterministic Serialization** - Consistent JSON serialization
8. **Temporal Constraint Enforcement** - Timestamps on all events

## Safety Features

### Disclaimers
Every module includes domain-specific safety disclaimers warning users about:
- Limitations of AI predictions
- Need for professional review
- Regulatory requirements
- Appropriate use cases

### Prohibited Uses
Each module detects and blocks prohibited operations:
- Unauthorized practice of professions
- Violations of regulations
- Malicious applications
- Privacy violations

### Example
```python
# This will raise SafetyViolation
intent = PlatformIntent(
    vertical=VerticalModule.VITRA,
    operation="bioweapon development",  # Prohibited!
    parameters={},
    user_id="user_001"
)
```

## Hardware Substrate Mapping

The platform automatically selects optimal hardware:

| Module | Default Substrate | Rationale |
|--------|------------------|-----------|
| JURIS | CPU | Logic-heavy, not compute-bound |
| VITRA | MI300X | Memory-intensive genomics |
| ECORA | CEREBRAS | Large-scale climate models |
| CAPRA | GB200 | Parallel financial simulations |
| SENTRA | GB200 | Physics simulations |
| NEURA | CEREBRAS | Neural network models |
| FLUXA | QPU | Optimization problems |

## Examples

See `examples/` directory for QIL (QRATUM Intent Language) files demonstrating each vertical:

- `juris_intent.qil` - Contract analysis
- `vitra_intent.qil` - DNA sequence analysis
- `ecora_intent.qil` - Climate projection
- `capra_intent.qil` - Option pricing
- `sentra_intent.qil` - Orbit propagation
- `neura_intent.qil` - EEG analysis
- `fluxa_intent.qil` - Route optimization

## Testing

Comprehensive test coverage (58 tests):
- 26 platform core tests
- 32 vertical module tests

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_juris.py -v
```

## Event Chain Verification

```python
platform = QRATUMPlatform()
# ... execute contracts ...

# Verify integrity
assert platform.verify_integrity()

# Get events for specific contract
events = platform.event_chain.get_events_for_contract(contract_id)
```

## Directory Structure

```
qratum_platform/
├── __init__.py
├── core.py              # Base classes, contracts, events
├── substrates.py        # Hardware mappings
└── utils.py             # Crypto utilities

verticals/
├── __init__.py
├── juris.py             # Legal AI
├── vitra.py             # Bioinformatics
├── ecora.py             # Climate & Energy
├── capra.py             # Financial Risk
├── sentra.py            # Aerospace & Defense
├── neura.py             # Neuroscience & BCI
└── fluxa.py             # Supply Chain

examples/
└── *_intent.qil         # Example intent files

tests/
├── test_platform.py     # Core platform tests
└── test_*.py            # Module-specific tests
```

## Future Work

The remaining 7 vertical modules to be implemented:
- SPECTRA - Spectrum Management
- AEGIS - Cybersecurity
- LOGOS - Education & Training
- SYNTHOS - Materials Science
- TERAGON - Geospatial Intelligence
- HELIX - Genomic Medicine
- NEXUS - Cross-domain Intelligence

## License

See LICENSE file for details.

## Contributing

See CONTRIBUTING.md for guidelines.

## Security

Report security vulnerabilities to security@qratum.io

## Acknowledgments

Built on the QRATUM platform architecture with inspiration from domain-specific AI systems across multiple industries.
