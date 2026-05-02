# Q-Stack

Q-Stack is a deterministic research playground that layers identity, a deterministic runtime substrate, a compact cognition stack, and domain adapters for simulation and economics. Everything is dependency-light, side-effect explicit, and designed to be auditable in safety-critical environments.

## Architectural slices
- **Q (Identity and sovereignty)**: deterministic key derivation, signing, attestations, trust graphs, and an in-memory ledger/registry to anchor provenance for every payload.
- **QNX (Runtime substrate)**: lexicographic operator scheduling, explicit safety constraints/envelopes, trace capture, and deterministic VM-style execution. Simulation and valuation operators from QuASIM/QuNimbus can be registered for cross-layer execution.
- **QNX-AGI (Cognition)**: perception encoders, world models with domain dynamics and QuASIM/QuNimbus adapters, deterministic memory systems (working/episodic/semantic), planners (greedy, heuristic, A*, beam, MPC), orchestrators with critic scoring, and alignment policies.
- **QuASIM (Simulation)**: structured tensor kernels with classical evaluators for aerospace/finance/pharma scenarios, circuit evaluation helpers, and deterministic MLIR-lowering stubs.
- **QuNimbus (Economics)**: deterministic valuation primitives for pricing, risk, incentives, governance, and QNX operator adapters.

## Determinism and safety
- No ambient randomness in core logic; all stochastic behaviors are seeded and explicit in tests only.
- Operator ordering, safety envelopes, and validation steps are deterministic and inspectable with trace capture.
- Memory and planner behaviors use pure functions or bounded-state machines to keep reasoning reproducible.

## Layout
- `q/`: identity, signing, sovereignty, attestation, trust graph, ledger, and identity registry helpers.
- `qnx/`: deterministic runtime operators, scheduler, safety envelopes, diagnostics, and tracing utilities.
- `qnx_agi/`: perception encoders, world-model dynamics, memory systems, planning stack, orchestrators, alignment policies, and integration adapters.
- `quasim/`: simulation kernels, tensor operations, circuits, evaluators, and domain-specific helpers with QNX operator adapters.
- `qunimbus/`: deterministic valuation primitives for pricing, risk, incentives, governance, and QNX operator adapters.
- `docs/`: architecture and API overviews describing flows and design constraints.
- `tests/`: deterministic unit and integration coverage for identity, runtime, cognition flows, simulation, valuation, and cross-layer integration.

Import via `import qstack` or submodules (e.g., `from qstack.q import KeyManager`) to explore the building blocks. The code is self-contained and does not modify the rest of the QuASIM tree.

## Launching a deterministic demo
Use the lightweight CLI to wire up the identity layer and the deterministic QNX runtime in one shot:

```bash
python -m qstack.launch --goal "stabilize" --energy 5
```

What happens under the hood:
1. Seed an orchestrator identity, sign it, and record an attested ledger entry.
2. Construct a trust graph with an initial self-trust edge.
3. Bind the requested goal, advance the deterministic clock, and debit the energy budget once.
4. Capture the operator trace alongside the runtime state for inspection.

The command prints a JSON payload with the orchestrator record, ledger head, trust relationships, runtime state, execution result, and trace events so you can validate the stack wiring end-to-end or embed it into tests.
