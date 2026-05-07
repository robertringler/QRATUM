//! P0.5 — INV-R. Lock-acquisition graph contains no directed cycles.
//! Host-side cycle detector identical algorithm to kernel's
//! `smp::lock_graph::has_cycle`.

fn has_cycle(edges: &[(u32, u32)], n: usize) -> bool {
    let mut color = vec![0u8; n];
    fn dfs(v: u32, edges: &[(u32, u32)], color: &mut [u8]) -> bool {
        color[v as usize] = 1;
        for &(f, t) in edges {
            if f != v { continue; }
            match color[t as usize] {
                1 => return true,
                0 => if dfs(t, edges, color) { return true; },
                _ => {}
            }
        }
        color[v as usize] = 2;
        false
    }
    for v in 0..n as u32 {
        if color[v as usize] == 0 && dfs(v, edges, &mut color) { return true; }
    }
    false
}

#[test]
fn linear_chain_no_cycle() {
    let edges = [(0u32, 1), (1, 2), (2, 3)];
    assert!(!has_cycle(&edges, 4));
}

#[test]
fn back_edge_is_cycle() {
    let edges = [(0u32, 1), (1, 2), (2, 0)];
    assert!(has_cycle(&edges, 3));
}

#[test]
fn self_loop_is_cycle() {
    assert!(has_cycle(&[(5u32, 5)], 8));
}
