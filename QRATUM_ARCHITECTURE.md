# QRATUM Canonical Architecture v1.0

## Overview

QRATUM implements the complete **QRADLE + QRATUM Canonical Architecture** as a sovereign control plane and deterministic execution substrate for heterogeneous cluster computing.

## Execution Flow

```
Intent → Authority → Contract → Time → Execution → Event → Audit
```

## Architecture Components

### 1. QIL (Q Intent Language)
Location: `qil/`

High-level intent language for expressing computational requirements:
- **grammar.py**: EBNF grammar definition
- **ast.py**: Abstract Syntax Tree models
- **parser.py**: Parser with validation
- **serializer.py**: Deterministic serialization

**Example QIL Intent:**
```qil
INTENT hybrid_ai_quantum {
    OBJECTIVE optimize_molecular_design
    HARDWARE ONLY GB200 AND QPU NOT IPU
    CONSTRAINT GPU_VRAM >= 500
    CAPABILITY llm_training
    CAPABILITY quantum_optimizer
    TIME deadline: 10800s
    AUTHORITY user: research_lead
    TRUST level: verified
}
```

### 2. Contract System
Location: `contracts/`

Immutable, hash-addressed contracts for execution:
- **base.py**: Base contract with hash-addressing (frozen=True)
- **intent.py**: IntentContract (authorized intent)
- **capability.py**: CapabilityContract with ClusterTopology
- **temporal.py**: TemporalContract (time as a resource)
- **event.py**: EventContract (expected event sequence)

All contracts are:
- Immutable (frozen dataclasses)
- Hash-addressed (SHA-256)
- Deterministically serializable
- Append-only

### 3. Q-Core (Sovereign Control Plane)
Location: `qcore/`

Authorization and policy engine that NEVER executes:
- **authority.py**: Authorization engine (raises AuthorizationError on failure)
- **policy.py**: Policy evaluation
- **resolver.py**: Capability resolution to hardware
- **issuer.py**: Issues all 4 contract types from authorized intent

**CRITICAL**: Q-Core authorizes but NEVER executes workloads.

### 4. Events (Causality Fabric)
Location: `events/`

Global append-only event log:
- **log.py**: Immutable event log with cryptographic chaining

Every action emits deterministic, hash-chained events for:
- Audit trails
- Deterministic replay
- Causal verification

### 5. Spine (Execution Dispatcher)
Location: `spine/`

Deterministic execution spine:
- **executor.py**: Dispatches validated contracts to adapters

The spine:
- Validates contract bundles
- Dispatches to appropriate adapters
- Logs all execution events
- Returns execution proofs

### 6. Adapters (Frankenstein Clusters)
Location: `adapters/`

Zero-policy substrate adapters:
- **base.py**: Base adapter interface (enforces ZERO policy)
- **gb200.py**: NVIDIA GB200 NVL72 Blackwell
- **mi300x.py**: AMD MI300X GPU
- **cerebras.py**: Cerebras CS-3 Wafer-Scale
- **qpu.py**: Quantum Processing Units
- **ipu.py**: Graphcore IPU
- **gaudi.py**: Intel Gaudi 3
- **cpu.py**: High-core-count CPU
- **registry.py**: Adapter registry

**CRITICAL**: Adapters contain ZERO policy logic. They:
- Accept ONLY valid contracts
- Emit deterministic events
- Return ExecutionProof with cryptographic verification
- Raise AdapterError (FATAL) on any violation

## Hard Constraints (FATAL Enforcement)

All 8 hard constraints are enforced as FATAL errors:

1. ✅ **QRATUM does not execute workloads** - Q-Core only authorizes
2. ✅ **QRATUM does not call Kubernetes/Slurm/GPU APIs** - Adapters handle execution
3. ✅ **QRADLE (adapters) reject execution without valid contracts** - BaseAdapter validates
4. ✅ **All execution traces back to Intent** - Contract references enforce lineage
5. ✅ **Every action emits deterministic events** - Event log enforces this
6. ✅ **Time is explicit and enforceable** - TemporalContract enforces deadlines
7. ✅ **Adapters contain zero policy logic** - BaseAdapter enforces contract-only operation
8. ✅ **All contracts are immutable and append-only** - frozen=True dataclasses

## Usage Example

```python
from qil import parse_intent
from qcore import AuthorizationEngine, CapabilityResolver, ContractIssuer
from spine import ContractExecutor
from adapters import AdapterRegistry

# 1. Parse QIL Intent
qil_text = """
INTENT llm_training {
    OBJECTIVE train_llama3_70b
    HARDWARE ONLY GB200
    CAPABILITY llm_training
    TIME deadline: 7200s
    AUTHORITY user: mlops_team
    TRUST level: verified
}
"""
intent = parse_intent(qil_text)

# 2. Authorize Intent
auth_engine = AuthorizationEngine()
auth_result = auth_engine.authorize_intent(intent)

# 3. Issue All 4 Contracts
resolver = CapabilityResolver()
issuer = ContractIssuer(resolver=resolver)
contract_bundle = issuer.issue_contracts(intent, auth_result)

# 4. Execute via Adapter
adapter_registry = AdapterRegistry()
executor = ContractExecutor(adapter_registry=adapter_registry)
execution_result = executor.execute(contract_bundle)

# 5. Verify Event Log
from events import get_global_event_log
event_log = get_global_event_log()
events = event_log.get_contract_events(contract_bundle.intent_contract.contract_id)
assert event_log.verify_causal_chain(contract_bundle.intent_contract.contract_id)
```

## QIL Examples

Location: `examples/intents/`

- **llm_training_gb200.qil**: LLM training on NVIDIA GB200
- **quantum_chemistry.qil**: Quantum molecular simulation
- **hybrid_ai_quantum.qil**: Hybrid AI + Quantum optimization

## Testing

Run comprehensive test suite:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all QRATUM tests (43 tests)
pytest tests/unit/test_qil_parser.py tests/unit/test_contracts.py tests/integration/test_end_to_end.py -v

# Run with coverage
pytest tests/unit/ tests/integration/test_end_to_end.py --cov=qil --cov=qcore --cov=contracts --cov=events --cov=spine --cov=adapters
```

### Test Coverage
- **17 unit tests** for QIL parser
- **18 unit tests** for contracts
- **8 integration tests** for end-to-end flow

All 43 tests pass with enforcement of all 8 hard constraints.

## Supported Cluster Types

| Cluster Type | Description | Use Cases |
|--------------|-------------|-----------|
| GB200 | NVIDIA GB200 NVL72 Blackwell | LLM training, AI inference |
| MI300X | AMD MI300X GPU | AI/ML workloads |
| CEREBRAS | Cerebras CS-3 Wafer-Scale | Large-scale neural networks |
| QPU | Quantum Processing Unit | VQE, QAOA, quantum chemistry |
| IPU | Graphcore IPU | Graph neural networks |
| GAUDI3 | Intel Gaudi 3 | AI training/inference |
| CPU | High-core-count CPU | General compute |

## Contract Flow

```
┌─────────────┐
│   Intent    │ (User-defined QIL)
└──────┬──────┘
       │
       v
┌─────────────┐
│  Authority  │ (Authorization + Policy)
└──────┬──────┘
       │
       v
┌─────────────────────────────────┐
│     4 Immutable Contracts       │
│  ┌─────────────────────────┐   │
│  │  1. IntentContract      │   │
│  │  2. CapabilityContract  │   │
│  │  3. TemporalContract    │   │
│  │  4. EventContract       │   │
│  └─────────────────────────┘   │
└──────────────┬──────────────────┘
               │
               v
       ┌───────────────┐
       │  Executor     │
       │  (Spine)      │
       └───────┬───────┘
               │
               v
       ┌───────────────┐
       │   Adapter     │ (Zero Policy)
       │  (QRADLE)     │
       └───────┬───────┘
               │
               v
       ┌───────────────┐
       │  Event Log    │ (Causal Chain)
       └───────────────┘
```

## Key Design Principles

1. **Separation of Concerns**: Q-Core authorizes, adapters execute
2. **Immutability**: All contracts are frozen and hash-addressed
3. **Determinism**: All events are deterministic and reproducible
4. **Zero Policy in Adapters**: Adapters accept contracts, no decisions
5. **Time as a Resource**: Temporal contracts enforce time budgets
6. **Audit Trail**: Every action logged with causal chains
7. **FATAL Enforcement**: All violations are FATAL errors, not warnings

## Error Handling

All constraint violations raise FATAL errors:

```python
# AuthorizationError - Intent not authorized
try:
    auth_engine.authorize_intent(untrusted_intent)
except AuthorizationError as e:
    # FATAL: Cannot proceed without authorization
    pass

# AdapterError - Invalid contract or execution failure
try:
    adapter.execute(invalid_contract)
except AdapterError as e:
    # FATAL: Adapter cannot execute invalid contract
    pass

# CapabilityResolutionError - Cannot resolve capability
try:
    resolver.resolve_capabilities(invalid_intent)
except CapabilityResolutionError as e:
    # FATAL: Cannot map capability to hardware
    pass
```

## Project Structure

```
qratum/
├── qil/                     # Q Intent Language
│   ├── grammar.py
│   ├── ast.py
│   ├── parser.py
│   └── serializer.py
├── qcore/                   # Q-Core Control Plane
│   ├── authority.py
│   ├── policy.py
│   ├── resolver.py
│   └── issuer.py
├── contracts/               # Immutable Contracts
│   ├── base.py
│   ├── intent.py
│   ├── capability.py
│   ├── temporal.py
│   └── event.py
├── events/                  # Event Fabric
│   └── log.py
├── spine/                   # Execution Spine
│   └── executor.py
├── adapters/                # Cluster Adapters
│   ├── base.py
│   ├── gb200.py
│   ├── mi300x.py
│   ├── cerebras.py
│   ├── qpu.py
│   ├── ipu.py
│   ├── gaudi.py
│   ├── cpu.py
│   └── registry.py
├── examples/intents/        # Example QIL
│   ├── llm_training_gb200.qil
│   ├── quantum_chemistry.qil
│   └── hybrid_ai_quantum.qil
└── tests/                   # Test Suite
    ├── unit/
    │   ├── test_qil_parser.py
    │   └── test_contracts.py
    └── integration/
        └── test_end_to_end.py
```

## Version

**QRATUM Canonical Architecture v1.0** - Production Ready

## License

Apache 2.0 License

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.

**Note**: All contributions must maintain the 8 hard constraints and FATAL error enforcement.
