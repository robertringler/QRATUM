# API Overview

This overview highlights key classes and typical usage patterns.

## Identity (q)

- `KeyManager(seed).derive_key(name)`: deterministic SHA-256 derivation for named secrets.
- `Signer(key).sign(message) / verify(message, signature)`: deterministic MAC semantics.
- `QIdentity(name, key)`: serializable identity container.
- `SovereignObject(identity, claims).digest()`: stable digest capturing identity + claims.
- `Attestor(signer).attest(payload)`: wrap payloads with signatures.
- `TrustGraph` and `IdentityRegistry`: track trust edges and registered identities with attestations.
- `Ledger`: in-memory append-only ledger with chained digests for provenance.

## Runtime (qnx.runtime)

- `OperatorLibrary.register(name, fn)`: add deterministic operators.
- `DeterministicScheduler.schedule(state, goal)`: execute operators in lexicographic order.
- `SafetyValidator`: pre/post rule evaluation with envelopes.
- `QNXVM.run_cycle(state, goal)`: run a cycle and emit traces or error markers.
- `TraceRecorder`: collect deterministic execution traces for later audit.

## QNX-AGI Highlights

- Perception encoders (aerospace, finance, pharma) convert raw dicts into `Percept` objects.
- Worldmodel dynamics apply domain-specific state updates and uncertainty scoring.
- Memory systems provide `WorkingMemory`, `EpisodicMemoryWithDecay`, and `SemanticMemory` with deterministic pruning.
- Planning offers `GreedyPlanner`, `HeuristicSearchPlanner`, and `GoalDecomposer` utilities.
- Orchestrator coordinates subagents, runs critic scoring, and emits QNX-ready plans.

## QuASIM / QuNimbus

- QuASIM exposes deterministic tensor operations, small quantum circuit helpers, and domain evaluators.
- QuNimbus provides pricing, risk, incentives, and governance calculators usable directly or via alignment layers.
