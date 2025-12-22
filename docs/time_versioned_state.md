# Time-Versioned State

`qtime` captures deterministic snapshots of runtime, scenario, node, and cluster state. Each snapshot is canonically serialized, hashed, and can be diffed or restored to support replay and audit.

## Components

- **Snapshot**: wraps state, tick, metadata with a stable hash id.
- **SnapshotDiffer**: generates key-wise diffs between snapshots.
- **SnapshotRegistry**: stores and retrieves snapshots by id and tick.
- **Restore**: rebuilds a `Snapshot` object from serialized content.

## Workflow

1. Capture snapshots at important ticks.
2. Register them in the registry and optionally store their ids in the ledger.
3. Use `diff` to understand change sets and `restore` to replay state transitions.
