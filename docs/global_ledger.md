# Global Ledger

`qledger` provides an append-only, hash-chained record of constitutional changes, node events, and policy decisions. Records are deterministic, tick-stamped, and can be replayed to reconstruct historical worldstate and policy context.

## Structure

- **LedgerRecord**: canonical payload containing tick, record type, node id, metadata, and previous hash.
- **LedgerChain**: ordered list of records with deterministic linking.
- **LedgerStore**: multi-chain storage for node, cluster, and constitution streams.
- **LedgerQuery**: helper to answer time-travel questions like active constitution at tick _T_.

## Guarantees

- Append-only semantics enforced via chained hashes.
- Canonical JSON serialization for reproducible hashing.
- Integration with snapshots to tie worldstate to specific ledger entries.
