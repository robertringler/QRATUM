# Q-Stack OS-like Layer

The OS-like layer inside QSK provides deterministic abstractions without invoking host system resources.

## Processes and Scheduling

- `qsk.processes.DeterministicProcess` wraps DAG workloads and orders steps via `qsk.scheduler.DeterministicScheduler`.
- Execution traces are reproducible and verifiable.

## Virtual Filesystem

- `qsk.fs_sim.VirtualFileSystem` maintains an in-memory tree with deterministic hashing via `fingerprint()`.

## IPC

- `qsk.ipc.Mailbox` and `Channel` implement ordered message delivery through queues.

## Resource Accounting

- `qsk.resources.ResourceBudget` tracks logical CPU and memory consumption to prevent overuse in simulations.

All components avoid nondeterminism and external side effects, making them suitable for sovereign node scenarios.
