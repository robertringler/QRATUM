#!/usr/bin/env python3
"""Reproduce the arithmetic behind QRATUM's headline AHTC-K reduction claim.

The registered claim `ahtc-k.scheduler.10x` is gated by
`crates/qratum-arbiter/tests/ahtc_k_real_10x_validation.rs`, which asserts
`scheduling_reduction_ratio >= 10.0` on a stream produced by
`performance::workload(N, dup_pct, seed)`.

That generator emits exactly one "hot" canonical key plus `COLD_POOL`
cold keys, so the AHTC-K enqueue count is bounded by `1 + COLD_POOL`
independent of N. The reduction ratio is therefore N / (1 + COLD_POOL):
it is a property of the generator's key-space size, not a measurement of
any workload.

Run from the repository root:
    python tools/value_analysis/ahtc_k_claim_probe.py
"""

from __future__ import annotations

COLD_POOL = 32  # os/qratum-os/crates/qratum-arbiter/src/performance.rs:25
GATE = 10.0  # the asserted threshold


def model_ratio(n: int, cold_pool: int = COLD_POOL) -> float:
    """Reduction ratio implied by a key space of size 1 + cold_pool."""
    return n / float(1 + cold_pool)


def main() -> None:
    print("AHTC-K scheduling-reduction claim, modelled from the generator")
    print(f"  COLD_POOL = {COLD_POOL}  ->  distinct canonical keys = {1 + COLD_POOL}")
    print(f"  CI gate   = >= {GATE}x\n")
    print(f"{'events N':>12} {'ahtc enqueues':>15} {'ratio':>12} {'gate margin':>13}")
    for n in (1_000, 10_000, 50_000, 100_000, 1_000_000):
        r = model_ratio(n)
        print(f"{n:>12,} {1 + COLD_POOL:>15} {r:>12,.1f} {r / GATE:>12.0f}x")

    print("\nObservations:")
    print("  * The ratio is linear in N and unbounded; the 10x gate cannot fail")
    print("    for any N > 330, whatever the scheduler does.")
    print("  * The published figures 1515 (N=50,000) and 3030 (N=100,000) are")
    print("    registered as claim doc_phrases in proof_gate::CLAIMS, so the")
    print("    claim-provenance machinery certifies them faithfully.")
    print("  * Measured values from `cargo test --release`:")
    print("      performance_truth           N=50,000  -> 1515.15x")
    print("      ahtc_k_real_10x_validation  N=100,000 -> 3030.303x")
    print("    Both equal N/33 exactly, confirming the generator is the cause.")
    print("\nConclusion: the mechanism that enforces the claim is real; the")
    print("workload that substantiates it is not representative of anything.")


if __name__ == "__main__":
    main()
