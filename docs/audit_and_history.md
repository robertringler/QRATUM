# Audit and Historical Reconstruction

By combining `qledger` with `qtime` snapshots, operators can answer: *what constitution and policies were in force when an event occurred?* Queries use ledger records to locate the relevant constitution version and snapshot ids to rebuild state.

## Procedure

1. Identify the event tick and retrieve relevant ledger records.
2. Determine the active constitution via `LedgerQuery.active_constitution_at`.
3. Restore the closest snapshot from `SnapshotRegistry.latest_for_tick`.
4. Apply diffs or subsequent ledger events to replay the exact conditions.

All steps are deterministic and avoid wall-clock time, enabling reproducible audits.
