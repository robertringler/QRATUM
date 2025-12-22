# Q-Stack Architecture (Expansion Pass 2)

Q-Stack layers deterministic identity, runtime, cognition, simulation, and valuation. Each layer is intentionally simple yet composable so that cross-layer flows can be traced, replayed, and audited.

## Layered overview

- **Q (Identity & Sovereignty)**: deterministic key derivation, signing, attestations, trust graph edges, registries, and append-only ledgers. Every payload passed to higher layers carries provenance digests.
- **QNX (Deterministic Runtime)**: operator library, lexicographic scheduler, safety constraints/envelopes, trace recorder, and replay-friendly VM cycles. Operators from QuASIM and QuNimbus can be registered to make simulation and valuation first-class runtime actions.
- **QNX-AGI (Cognitive Loop)**: perception ➜ world modeling ➜ memory ➜ planning ➜ orchestration. World models use domain-specific dynamics and can call QuASIM/QuNimbus adapters for simulation-informed predictions. Planners reason with safety envelopes and deterministic heuristics (greedy, heuristic search, A*, beam, MPC).
- **QuASIM (Simulation)**: tensor and circuit primitives plus deterministic classical kernels (aerospace telemetry, financial shock propagation, pharma reaction kinetics). Exposes operators to QNX and adapters to QNX-AGI world models.
- **QuNimbus (Economic/Governance)**: deterministic macro/sector valuation, risk, incentives, and governance scoring. Provides QNX operators and is consumable by planners for economic alignment.
- **Infra**: configuration, telemetry, deterministic logging (counter-based timestamps), health monitoring, and network transport shims.
- **Cert**: traceability matrices, compliance checks (DO-178C, NIST 800-53), and verification hooks.
- **Deployment**: deterministic Docker/Kubernetes assets and bootstrap scripts wiring QNX, QuASIM, and QuNimbus components together.

## Cross-layer flows

1. **Provenance propagation**: Q identities sign percepts and plan outputs; ledger entries chain digests so any world-state update can be traced to identities and decisions.
2. **Perception ➜ Worldmodel**: Encoders normalize domain-specific telemetry into `Percept` objects, attach provenance hashes, and hand them to world models. World models apply deterministic dynamics or call QuASIM kernels for forward predictions, producing new `WorldState` nodes in a `WorldStateGraph`.
3. **Memory**: Working memory caches the latest percepts; episodic memory stores structured time-stamped events with decay policies; semantic memory writes facts into a small deterministic graph DB with embedding hashes.
4. **Planning**: Planners compute candidate plans using deterministic heuristics and constraints. A* and beam search operate over symbolic states; MPC rolls deterministic dynamics forward with safety-envelope checks before emitting QNX goals.
5. **Orchestration**: Sub-agents receive decomposed goals, critics score them, and the orchestrator emits an execution bundle for the QNX VM. TraceRecorder captures execution traces for audit and feeds semantic memory.
6. **Runtime execution**: QNX VM runs registered operators (including QuASIM/QuNimbus) in lexicographic order, applying pre/post safety validation and persisting traces to the ledger.
7. **Economic alignment**: QuNimbus valuations and governance scores are consulted by alignment policies, ensuring actions adhere to deterministic policy stores and economic constraints.

## Determinism & safety practices

- All randomness is removed; any stochastic-like behavior uses fixed seeds or hashing-based sampling.
- Safety envelopes set explicit bounds; constraint functions are pure and composable.
- Traces include deterministic counters, hashes, and operator results to support replay.
- Domain models avoid side effects and rely on pure data transformations.

## Extension points

- **Operators**: register new simulation or valuation operators with QNX using deterministic functions.
- **Dynamics**: add domain-specific `simulate_step` functions and attach them to `WorldModel` instances.
- **Planners**: implement new planners against the `Planner` interface to integrate novel heuristics.
- **Adapters**: add integration adapters that translate QuASIM/QuNimbus outputs into world-state updates or planner cost signals.
