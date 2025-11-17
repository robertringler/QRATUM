# QNX backend validation report

## Interface discovery

* **Entrypoints**
  * `qnx.core.QNXSubstrate` lifecycle hooks: `initialise_runtime()`, `boot_backend()`, `dispatch()`, `teardown_backend()`, and `lifecycle_run()` orchestrate init → boot → dispatch → teardown across backends.
  * CLI: `python -m qnx.cli simulate --scenario <id>` (see `qnx/cli.py`).
  * Universal contract: `quasim.api.simulate.run_scenario()` dispatches QuASIM engines consumed by `QuasimModernBackend`.
* **Backends**
  * `QuasimModernBackend` → calls `run_scenario()` via `ScenarioSpec` (current engine).
  * `QuasimLegacyV120Backend` → stubbed shim returning `not_implemented` for legacy engine wiring.
  * `QVRWinBackend` → Windows-only bridge to external QVR CLI (validated via environment probe).
* **Lifecycle & IPC hooks**
  * Lifecycle instrumentation lives in `QNXSubstrate` with optional RTOS hooks and integrity hashing in `qnx/security.py`.
  * Mock RTOS provided for tests (`tests/qnx_integration/mock_qnx_rtos.py`) exercises boot, scheduler ticks, IPC bus, and teardown flows.
* **Contracts & data structures**
  * `SimulationConfig`, `SubstrateResult`, and `SecurityLevel` in `qnx/types.py` define runtime configuration and outputs.
  * Deterministic hashing and validation live in `qnx/security.py`; sustainability metrics in `qnx/sustainability.py`.

## Interface compliance and gaps

* **Compliant**
  * Modern backend executes QuASIM scenarios deterministically and returns normalized payloads.
  * Lifecycle hooks exposed for init → boot → dispatch → teardown and IPC logging.
  * CLI exercises substrate without hardware dependencies.
* **Known gaps**
  * Legacy backend remains a stub (`status: not_implemented`).
  * `QVRWinBackend` is disabled on non-Windows platforms and relies on an external `qvr.exe` binary.
  * Security validation currently performs placeholder checks only; enforcement hooks remain TODO.

## Test harness summary

* **Unit coverage**
  * Public QNX module functions: CLI surface, backends, hashing/integrity helpers, sustainability fallback, and lifecycle orchestration.
* **Integration coverage**
  * Lifecycle flow over mock RTOS with IPC bus assertions.
  * Deterministic QuASIM scenario execution and hash repeatability via substrate dispatch.
  * Windows backend simulated with monkeypatched process execution.
* **Location**: `tests/qnx/` (backend adapters) and `tests/qnx_integration/` (lifecycle, RTOS, CLI, contracts).

## Validation results

* Latest run: `pytest tests/qnx tests/qnx_integration -v` → 18 tests passed headlessly with deterministic mock RTOS coverage.

## Recommendations

* Implement the real legacy v1.2.0 adapter or deprecate the backend stub.
* Add hardened security-context enforcement (policy checks, capability gating).
* Provide a Windows CI lane to exercise `QVRWinBackend` end-to-end when infrastructure is available.
