# Sovereign Node Architecture

A sovereign Q-Stack node binds identity, deterministic execution, and governance into a reproducible unit. The node couples Q identity material with QNX runtime operators, the QDL/QIR pipeline, and the QSK kernel to provide ordered system calls and verifiable traces.

## Identity Binding
- `qnode.identity_binding.NodeIdentity` binds a node identifier and public key.
- Attestations are generated deterministically and verified via simple string-based signatures.

## Lifecycle State Machine
- Implemented in `qnode.lifecycle.NodeLifecycle` with states `init -> running -> paused -> shutdown`.
- All transitions are explicitly validated and recorded to allow deterministic replay of node behavior.

## Policy-Enforced Syscalls
- `qnode.syscalls.SyscallRouter` exposes high-level calls into QNX/QDL/QSK subsystems.
- Policies (`qnode.policies`) enforce allowlists and resource budgets before a call executes.
- Each invocation is captured in `qnode.incident_log.IncidentLog` for auditability.

## Health Monitoring
- `qnode.monitor.HealthMonitor` tracks deterministic counters and alerts using `infra.metrics` and `infra.alerts`.
- Incident logs are produced for errors and triggered alerts, supporting post-incident analysis.

## Orchestration
- `qnode.orchestration.NodeOrchestrator` wires node policies, syscall handlers, and incident logging.
- The orchestrator is intentionally minimal to support self-hosting and bootstrap flows in deployment scripts.

## Deterministic Guarantees
- No real-world IO or nondeterministic sources are used; all behavior is pure Python state updates.
- State traces (lifecycle history, incident log, metrics snapshots) can be serialized and replayed for verification.
