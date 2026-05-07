//! P0.6 — replay barrier rendezvous tests.

#[derive(Default)]
struct Barrier { arrived: Vec<u8>, generation: u64 }

impl Barrier {
    fn arrive(&mut self, cpu: u8) {
        if !self.arrived.contains(&cpu) { self.arrived.push(cpu); }
    }
    fn all_in(&self, expected: usize) -> bool { self.arrived.len() >= expected }
    fn release(&mut self) -> u64 {
        self.generation += 1;
        self.arrived.clear();
        self.generation
    }
}

#[test]
fn barrier_releases_after_all_arrive() {
    let mut b = Barrier::default();
    for cpu in 0..4 { b.arrive(cpu); }
    assert!(b.all_in(4));
    assert_eq!(b.release(), 1);
    assert!(!b.all_in(4));
}

#[test]
fn duplicate_arrivals_idempotent() {
    let mut b = Barrier::default();
    for _ in 0..5 { b.arrive(2); }
    assert!(!b.all_in(4));
}
