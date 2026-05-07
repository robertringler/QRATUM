# Hardware tools (P0.6)

Minimal, deterministic helpers used by `hardware-replay.yml` and
`topology-validation.yml`. They print machine-readable lines and
exit non-zero on any inconsistency.

| Tool                         | Purpose                                      |
|------------------------------|----------------------------------------------|
| `collect_topology.{ps1,sh}`  | Print canonical topology JSON                |
| `replay_capture.{ps1,sh}`    | Capture event-stream hash                    |
| `validate_receipts.{ps1,sh}` | Validate Merkle roots against host re-hash   |

Real-hardware capture is **deferred** until SMP-real bring-up; the
scripts emit a `status: deferred` line in that case.
