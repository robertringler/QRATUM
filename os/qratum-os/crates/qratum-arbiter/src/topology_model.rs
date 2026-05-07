//! P0.6 — Topology model. Mirrors `kernel::smp::core_topology`.

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
pub struct CoreInfo {
    pub cpu_id:     u8,
    pub apic_id:    u8,
    pub package:    u8,
    pub numa_node:  u8,
    pub smt_thread: u8,
}

#[derive(Clone, Debug, Default)]
pub struct Topology {
    pub cores: Vec<CoreInfo>,
}

impl Topology {
    pub fn new() -> Self { Self { cores: Vec::new() } }

    pub fn add(&mut self, c: CoreInfo) { self.cores.push(c); }

    pub fn topology_hash(&self) -> u64 {
        let mut h: u64 = 0xCBF2_9CE4_8422_2325;
        for c in &self.cores {
            for b in [c.cpu_id, c.apic_id, c.package, c.numa_node, c.smt_thread] {
                h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3);
            }
        }
        h
    }

    /// Two topologies are *equivalent* if they produce the same
    /// canonical (cpu_id, apic_id) mapping under sort.
    pub fn equivalent(&self, other: &Self) -> bool {
        let mut a: Vec<_> = self.cores.iter().map(|c| (c.cpu_id, c.apic_id)).collect();
        let mut b: Vec<_> = other.cores.iter().map(|c| (c.cpu_id, c.apic_id)).collect();
        a.sort(); b.sort();
        a == b
    }
}
