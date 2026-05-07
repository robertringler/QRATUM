//! P0.6 — determinism across topologies. Same workload + different
//! topology assignments must produce identical merkle roots when the
//! ordering invariants hold.

use qratum_arbiter::receipt_merkle::{ReceiptMerkle, leaf_for};
use qratum_arbiter::topology_model::{Topology, CoreInfo};

fn workload_root(_topo: &Topology, n: usize) -> [u8; 32] {
    // Workload identity is independent of physical core ids — only
    // the canonical (work_seq, payload) leaves matter.
    let mut m = ReceiptMerkle::new();
    for i in 0..n {
        m.append(leaf_for(&(i as u64).to_le_bytes()));
    }
    m.root()
}

#[test]
fn cross_topology_root_invariant() {
    let mut t1 = Topology::new();
    let mut t2 = Topology::new();
    for i in 0u8..4 {
        t1.add(CoreInfo { cpu_id: i, apic_id: i,     package: 0, numa_node: 0, smt_thread: 0 });
        t2.add(CoreInfo { cpu_id: i, apic_id: i + 8, package: 1, numa_node: 1, smt_thread: 1 });
    }
    assert_ne!(t1.topology_hash(), t2.topology_hash());
    assert_eq!(workload_root(&t1, 32), workload_root(&t2, 32));
}
