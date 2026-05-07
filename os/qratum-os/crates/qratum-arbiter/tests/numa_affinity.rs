//! P0.5 — NUMA affinity preservation.

use qratum_arbiter::numa_model::NumaTopology;

#[test]
fn dual_node_topology_is_balanced() {
    let t = NumaTopology::dual(8);
    assert_eq!(t.nodes, 2);
    let n0 = (0..8).filter(|c| t.node_of(*c) == 0).count();
    let n1 = (0..8).filter(|c| t.node_of(*c) == 1).count();
    assert_eq!(n0, n1);
}

#[test]
fn distance_is_symmetric_and_zero_diagonal() {
    let t = NumaTopology::dual(8);
    for a in 0u8..t.nodes { for b in 0u8..t.nodes {
        assert_eq!(t.distance(a, b), t.distance(b, a));
        if a == b { assert_eq!(t.distance(a, b), 0); }
    }}
}
