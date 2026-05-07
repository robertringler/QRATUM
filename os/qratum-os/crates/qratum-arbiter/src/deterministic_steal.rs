//! P0.6 — Host model of `kernel::smp::deterministic_work_steal`.

#[derive(Clone, Debug, Default)]
pub struct StealRecord {
    pub seq:       u64,
    pub src:       u8,
    pub dst:       u8,
    pub depth_src: u32,
    pub depth_dst: u32,
}

pub fn pick_steal(depths: &[u32]) -> Option<(u8, u8)> {
    if depths.len() < 2 { return None; }
    let mut max_d = 0u32; let mut min_d = u32::MAX;
    let mut src = 0u8;    let mut dst = 0u8;
    for (i, &d) in depths.iter().enumerate() {
        if d > max_d { max_d = d; src = i as u8; }
        if d < min_d { min_d = d; dst = i as u8; }
    }
    if max_d <= min_d.saturating_add(1) { return None; }
    if src == dst { return None; }
    Some((src, dst))
}

/// Replay-stable schedule: same input depth-trace ⇒ same steal record list.
pub fn schedule(traces: &[Vec<u32>]) -> Vec<StealRecord> {
    let mut out = Vec::new();
    for (i, depths) in traces.iter().enumerate() {
        if let Some((src, dst)) = pick_steal(depths) {
            out.push(StealRecord {
                seq: i as u64, src, dst,
                depth_src: depths[src as usize],
                depth_dst: depths[dst as usize],
            });
        }
    }
    out
}
