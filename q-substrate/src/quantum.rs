//! Mini QuASIM - 12-Qubit Deterministic Quantum Simulation Module
//!
//! Ultra-lightweight quantum simulator supporting:
//! - 12 qubits (4096 complex amplitudes)
//! - Full gate set: H, X, Y, Z, S, T, T†, CNOT, CZ, SWAP, Toffoli
//! - Rotation gates: RX, RY, RZ
//! - Fixed-point arithmetic option for micro-devices
//! - Deterministic state vector representation
//!
//! Memory footprint: ~32KB for state vector + minimal overhead

extern crate alloc;

use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Number of qubits in Mini QuASIM
pub const QUBITS: usize = 12;

/// State vector size: 2^12 = 4096
pub const STATE_SIZE: usize = 1 << QUBITS;

/// Complex number representation (8 bytes per amplitude)
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct Complex {
    pub re: f32,
    pub im: f32,
}

impl Complex {
    /// Create a new complex number
    #[inline(always)]
    pub const fn new(re: f32, im: f32) -> Self {
        Complex { re, im }
    }

    /// Zero constant
    pub const ZERO: Complex = Complex { re: 0.0, im: 0.0 };
    
    /// One constant
    pub const ONE: Complex = Complex { re: 1.0, im: 0.0 };
    
    /// Imaginary unit
    pub const I: Complex = Complex { re: 0.0, im: 1.0 };

    /// Complex multiplication
    #[inline(always)]
    pub fn mul(self, other: Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    /// Complex addition
    #[inline(always)]
    pub fn add(self, other: Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    /// Complex subtraction
    #[inline(always)]
    pub fn sub(self, other: Complex) -> Complex {
        Complex {
            re: self.re - other.re,
            im: self.im - other.im,
        }
    }

    /// Scale by real factor
    #[inline(always)]
    pub fn scale(self, factor: f32) -> Complex {
        Complex {
            re: self.re * factor,
            im: self.im * factor,
        }
    }

    /// Squared norm |z|²
    #[inline(always)]
    pub fn norm_sq(self) -> f32 {
        self.re * self.re + self.im * self.im
    }

    /// Phase angle arg(z)
    #[inline(always)]
    pub fn phase(self) -> f32 {
        self.im.atan2(self.re)
    }

    /// Complex conjugate
    #[inline(always)]
    pub fn conj(self) -> Complex {
        Complex { re: self.re, im: -self.im }
    }
}

/// Quantum gates supported by Mini QuASIM
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumGate {
    /// Hadamard gate
    Hadamard(usize),
    /// Pauli-X (NOT) gate
    PauliX(usize),
    /// Pauli-Y gate
    PauliY(usize),
    /// Pauli-Z gate
    PauliZ(usize),
    /// Phase gate (S)
    Phase(usize),
    /// T gate (π/8)
    T(usize),
    /// T-dagger gate
    TDagger(usize),
    /// CNOT (controlled-NOT) gate
    CNOT(usize, usize),
    /// Controlled-Z gate
    CZ(usize, usize),
    /// SWAP gate
    SWAP(usize, usize),
    /// Toffoli (CCNOT) gate
    Toffoli(usize, usize, usize),
    /// Rotation around X axis
    RX(usize, f32),
    /// Rotation around Y axis
    RY(usize, f32),
    /// Rotation around Z axis
    RZ(usize, f32),
}

/// Qubit state information for visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QubitState {
    /// State index in computational basis
    pub state_index: usize,
    /// Amplitude magnitude
    pub amplitude: f32,
    /// Phase angle
    pub phase: f32,
    /// Probability |amplitude|²
    pub probability: f32,
    /// Binary representation of state
    pub binary: String,
}

/// Gate operation record for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateRecord {
    /// Gate name
    pub gate: String,
    /// Target qubits
    pub qubits: Vec<usize>,
    /// Timestamp (operation count)
    pub op_count: u64,
}

/// Mini QuASIM - 12-Qubit Quantum Simulator
pub struct MiniQuASIM {
    /// State vector (4096 complex amplitudes)
    amplitudes: Vec<Complex>,
    /// Deterministic seed
    seed: u32,
    /// Gate history for audit
    gate_history: Vec<GateRecord>,
    /// Operation counter
    op_count: u64,
}

impl MiniQuASIM {
    /// Create a new Mini QuASIM instance
    pub fn new(seed: u32) -> Self {
        let mut amplitudes = vec![Complex::ZERO; STATE_SIZE];
        amplitudes[0] = Complex::ONE; // Initialize to |0...0⟩
        
        MiniQuASIM {
            amplitudes,
            seed,
            gate_history: Vec::new(),
            op_count: 0,
        }
    }

    /// Reset to initial |0...0⟩ state
    pub fn reset(&mut self) {
        for amp in &mut self.amplitudes {
            *amp = Complex::ZERO;
        }
        self.amplitudes[0] = Complex::ONE;
        self.gate_history.clear();
        self.op_count = 0;
    }

    /// Apply a quantum gate
    pub fn apply_gate(&mut self, gate: &QuantumGate) {
        match gate {
            QuantumGate::Hadamard(q) => self.hadamard(*q),
            QuantumGate::PauliX(q) => self.pauli_x(*q),
            QuantumGate::PauliY(q) => self.pauli_y(*q),
            QuantumGate::PauliZ(q) => self.pauli_z(*q),
            QuantumGate::Phase(q) => self.phase_gate(*q),
            QuantumGate::T(q) => self.t_gate(*q),
            QuantumGate::TDagger(q) => self.t_dagger(*q),
            QuantumGate::CNOT(c, t) => self.cnot(*c, *t),
            QuantumGate::CZ(c, t) => self.cz(*c, *t),
            QuantumGate::SWAP(q1, q2) => self.swap(*q1, *q2),
            QuantumGate::Toffoli(c1, c2, t) => self.toffoli(*c1, *c2, *t),
            QuantumGate::RX(q, theta) => self.rx(*q, *theta),
            QuantumGate::RY(q, theta) => self.ry(*q, *theta),
            QuantumGate::RZ(q, theta) => self.rz(*q, *theta),
        }
        self.op_count += 1;
    }

    /// Apply Hadamard gate to qubit
    /// H = (1/√2) * [[1, 1], [1, -1]]
    pub fn hadamard(&mut self, qubit: usize) {
        if qubit >= QUBITS { return; }
        
        let step = 1 << qubit;
        let h_factor = 0.70710678_f32; // 1/√2
        
        for i in (0..STATE_SIZE).step_by(2 * step) {
            for j in 0..step {
                let idx0 = i + j;
                let idx1 = idx0 + step;
                
                let a0 = self.amplitudes[idx0];
                let a1 = self.amplitudes[idx1];
                
                self.amplitudes[idx0] = Complex::new(
                    h_factor * (a0.re + a1.re),
                    h_factor * (a0.im + a1.im)
                );
                self.amplitudes[idx1] = Complex::new(
                    h_factor * (a0.re - a1.re),
                    h_factor * (a0.im - a1.im)
                );
            }
        }
        
        self.record_gate("H", vec![qubit]);
    }

    /// Apply Pauli-X (NOT) gate
    pub fn pauli_x(&mut self, qubit: usize) {
        if qubit >= QUBITS { return; }
        
        let step = 1 << qubit;
        for i in (0..STATE_SIZE).step_by(2 * step) {
            for j in 0..step {
                let idx0 = i + j;
                let idx1 = idx0 + step;
                let temp = self.amplitudes[idx0];
                self.amplitudes[idx0] = self.amplitudes[idx1];
                self.amplitudes[idx1] = temp;
            }
        }
        
        self.record_gate("X", vec![qubit]);
    }

    /// Apply Pauli-Y gate
    /// Y = [[0, -i], [i, 0]]
    pub fn pauli_y(&mut self, qubit: usize) {
        if qubit >= QUBITS { return; }
        
        let step = 1 << qubit;
        for i in (0..STATE_SIZE).step_by(2 * step) {
            for j in 0..step {
                let idx0 = i + j;
                let idx1 = idx0 + step;
                
                let a0 = self.amplitudes[idx0];
                let a1 = self.amplitudes[idx1];
                
                // |0⟩ -> i|1⟩, |1⟩ -> -i|0⟩
                self.amplitudes[idx0] = Complex::new(a1.im, -a1.re);
                self.amplitudes[idx1] = Complex::new(-a0.im, a0.re);
            }
        }
        
        self.record_gate("Y", vec![qubit]);
    }

    /// Apply Pauli-Z gate
    /// Z = [[1, 0], [0, -1]]
    pub fn pauli_z(&mut self, qubit: usize) {
        if qubit >= QUBITS { return; }
        
        for i in 0..STATE_SIZE {
            if (i >> qubit) & 1 == 1 {
                self.amplitudes[i] = self.amplitudes[i].scale(-1.0);
            }
        }
        
        self.record_gate("Z", vec![qubit]);
    }

    /// Apply Phase gate (S)
    /// S = [[1, 0], [0, i]]
    pub fn phase_gate(&mut self, qubit: usize) {
        if qubit >= QUBITS { return; }
        
        for i in 0..STATE_SIZE {
            if (i >> qubit) & 1 == 1 {
                let amp = self.amplitudes[i];
                self.amplitudes[i] = Complex::new(-amp.im, amp.re);
            }
        }
        
        self.record_gate("S", vec![qubit]);
    }

    /// Apply T gate (π/8 gate)
    /// T = [[1, 0], [0, e^(iπ/4)]]
    pub fn t_gate(&mut self, qubit: usize) {
        if qubit >= QUBITS { return; }
        
        let t_factor = Complex::new(0.70710678, 0.70710678); // e^(iπ/4)
        
        for i in 0..STATE_SIZE {
            if (i >> qubit) & 1 == 1 {
                self.amplitudes[i] = self.amplitudes[i].mul(t_factor);
            }
        }
        
        self.record_gate("T", vec![qubit]);
    }

    /// Apply T-dagger gate
    /// T† = [[1, 0], [0, e^(-iπ/4)]]
    pub fn t_dagger(&mut self, qubit: usize) {
        if qubit >= QUBITS { return; }
        
        let t_dag_factor = Complex::new(0.70710678, -0.70710678);
        
        for i in 0..STATE_SIZE {
            if (i >> qubit) & 1 == 1 {
                self.amplitudes[i] = self.amplitudes[i].mul(t_dag_factor);
            }
        }
        
        self.record_gate("T†", vec![qubit]);
    }

    /// Apply CNOT gate
    pub fn cnot(&mut self, control: usize, target: usize) {
        if control >= QUBITS || target >= QUBITS { return; }
        
        let ctrl_mask = 1 << control;
        let targ_mask = 1 << target;
        
        for i in 0..STATE_SIZE {
            if (i & ctrl_mask) != 0 {
                let pair_idx = i ^ targ_mask;
                if i < pair_idx {
                    let temp = self.amplitudes[i];
                    self.amplitudes[i] = self.amplitudes[pair_idx];
                    self.amplitudes[pair_idx] = temp;
                }
            }
        }
        
        self.record_gate("CNOT", vec![control, target]);
    }

    /// Apply Controlled-Z gate
    pub fn cz(&mut self, control: usize, target: usize) {
        if control >= QUBITS || target >= QUBITS { return; }
        
        let ctrl_mask = 1 << control;
        let targ_mask = 1 << target;
        
        for i in 0..STATE_SIZE {
            if (i & ctrl_mask) != 0 && (i & targ_mask) != 0 {
                self.amplitudes[i] = self.amplitudes[i].scale(-1.0);
            }
        }
        
        self.record_gate("CZ", vec![control, target]);
    }

    /// Apply SWAP gate
    pub fn swap(&mut self, qubit1: usize, qubit2: usize) {
        if qubit1 >= QUBITS || qubit2 >= QUBITS { return; }
        
        let mask1 = 1 << qubit1;
        let mask2 = 1 << qubit2;
        
        for i in 0..STATE_SIZE {
            let bit1 = (i & mask1) >> qubit1;
            let bit2 = (i & mask2) >> qubit2;
            
            if bit1 != bit2 {
                let j = (i ^ mask1) ^ mask2;
                if i < j {
                    let temp = self.amplitudes[i];
                    self.amplitudes[i] = self.amplitudes[j];
                    self.amplitudes[j] = temp;
                }
            }
        }
        
        self.record_gate("SWAP", vec![qubit1, qubit2]);
    }

    /// Apply Toffoli (CCNOT) gate
    pub fn toffoli(&mut self, control1: usize, control2: usize, target: usize) {
        if control1 >= QUBITS || control2 >= QUBITS || target >= QUBITS { return; }
        
        let ctrl1_mask = 1 << control1;
        let ctrl2_mask = 1 << control2;
        let targ_mask = 1 << target;
        
        for i in 0..STATE_SIZE {
            if (i & ctrl1_mask) != 0 && (i & ctrl2_mask) != 0 {
                let pair_idx = i ^ targ_mask;
                if i < pair_idx {
                    let temp = self.amplitudes[i];
                    self.amplitudes[i] = self.amplitudes[pair_idx];
                    self.amplitudes[pair_idx] = temp;
                }
            }
        }
        
        self.record_gate("TOFFOLI", vec![control1, control2, target]);
    }

    /// Apply RX rotation
    pub fn rx(&mut self, qubit: usize, theta: f32) {
        if qubit >= QUBITS { return; }
        
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let step = 1 << qubit;
        
        for i in (0..STATE_SIZE).step_by(2 * step) {
            for j in 0..step {
                let idx0 = i + j;
                let idx1 = idx0 + step;
                
                let a0 = self.amplitudes[idx0];
                let a1 = self.amplitudes[idx1];
                
                self.amplitudes[idx0] = Complex::new(
                    cos_half * a0.re + sin_half * a1.im,
                    cos_half * a0.im - sin_half * a1.re,
                );
                self.amplitudes[idx1] = Complex::new(
                    cos_half * a1.re + sin_half * a0.im,
                    cos_half * a1.im - sin_half * a0.re,
                );
            }
        }
        
        self.record_gate("RX", vec![qubit]);
    }

    /// Apply RY rotation
    pub fn ry(&mut self, qubit: usize, theta: f32) {
        if qubit >= QUBITS { return; }
        
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        let step = 1 << qubit;
        
        for i in (0..STATE_SIZE).step_by(2 * step) {
            for j in 0..step {
                let idx0 = i + j;
                let idx1 = idx0 + step;
                
                let a0 = self.amplitudes[idx0];
                let a1 = self.amplitudes[idx1];
                
                self.amplitudes[idx0] = Complex::new(
                    cos_half * a0.re - sin_half * a1.re,
                    cos_half * a0.im - sin_half * a1.im,
                );
                self.amplitudes[idx1] = Complex::new(
                    sin_half * a0.re + cos_half * a1.re,
                    sin_half * a0.im + cos_half * a1.im,
                );
            }
        }
        
        self.record_gate("RY", vec![qubit]);
    }

    /// Apply RZ rotation
    pub fn rz(&mut self, qubit: usize, theta: f32) {
        if qubit >= QUBITS { return; }
        
        let cos_half = (theta / 2.0).cos();
        let sin_half = (theta / 2.0).sin();
        
        for i in 0..STATE_SIZE {
            if (i >> qubit) & 1 == 0 {
                let amp = self.amplitudes[i];
                self.amplitudes[i] = Complex::new(
                    cos_half * amp.re + sin_half * amp.im,
                    cos_half * amp.im - sin_half * amp.re,
                );
            } else {
                let amp = self.amplitudes[i];
                self.amplitudes[i] = Complex::new(
                    cos_half * amp.re - sin_half * amp.im,
                    cos_half * amp.im + sin_half * amp.re,
                );
            }
        }
        
        self.record_gate("RZ", vec![qubit]);
    }

    /// Get probability of a computational basis state
    #[inline]
    pub fn measure_prob(&self, state: usize) -> f32 {
        if state < STATE_SIZE {
            self.amplitudes[state].norm_sq()
        } else {
            0.0
        }
    }

    /// Get all probabilities as a vector
    pub fn get_probabilities(&self) -> Vec<f32> {
        self.amplitudes.iter().map(|a| a.norm_sq()).collect()
    }

    /// Get quantum state information for visualization
    pub fn get_state_info(&self, max_states: usize) -> Vec<QubitState> {
        let mut states: Vec<QubitState> = self.amplitudes
            .iter()
            .enumerate()
            .filter(|(_, amp)| amp.norm_sq() > 1e-10)
            .take(max_states)
            .map(|(idx, amp)| QubitState {
                state_index: idx,
                amplitude: amp.norm_sq().sqrt(),
                phase: amp.phase(),
                probability: amp.norm_sq(),
                binary: format!("{:012b}", idx),
            })
            .collect();
        
        // Sort by probability descending. NaN values from invalid amplitudes
        // are treated as Equal to maintain determinism and avoid panics.
        // In valid quantum states, probabilities are always finite.
        states.sort_by(|a, b| b.probability.partial_cmp(&a.probability).unwrap_or(core::cmp::Ordering::Equal));
        states
    }

    /// Calculate Shannon entropy
    pub fn entropy(&self) -> f32 {
        let mut entropy = 0.0_f32;
        for amp in &self.amplitudes {
            let p = amp.norm_sq();
            if p > 1e-10 {
                entropy -= p * p.ln();
            }
        }
        entropy
    }

    /// Get state hash for determinism verification
    pub fn get_state_hash(&self) -> u64 {
        let mut hash: u64 = 0;
        for (i, amp) in self.amplitudes.iter().enumerate() {
            hash ^= ((amp.re.to_bits() as u64) << 32) | (amp.im.to_bits() as u64);
            hash = hash.rotate_left(7) ^ (i as u64);
        }
        hash
    }

    /// Get gate history
    pub fn get_gate_history(&self) -> &[GateRecord] {
        &self.gate_history
    }

    /// Get operation count
    pub fn get_op_count(&self) -> u64 {
        self.op_count
    }

    /// Record a gate operation
    fn record_gate(&mut self, gate: &str, qubits: Vec<usize>) {
        self.gate_history.push(GateRecord {
            gate: gate.into(),
            qubits,
            op_count: self.op_count,
        });
    }

    /// Run Bell state circuit: (|00⟩ + |11⟩)/√2
    pub fn bell_state(&mut self) -> (f32, f32) {
        self.reset();
        self.hadamard(0);
        self.cnot(0, 1);
        (self.measure_prob(0), self.measure_prob(3))
    }

    /// Run GHZ state circuit: (|000⟩ + |111⟩)/√2
    pub fn ghz_state(&mut self) -> (f32, f32) {
        self.reset();
        self.hadamard(0);
        self.cnot(0, 1);
        self.cnot(1, 2);
        (self.measure_prob(0), self.measure_prob(7))
    }
}

impl Default for MiniQuASIM {
    fn default() -> Self {
        Self::new(42) // Default deterministic seed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_state() {
        let qs = MiniQuASIM::new(42);
        assert!((qs.measure_prob(0) - 1.0).abs() < 1e-6);
        assert!(qs.measure_prob(1).abs() < 1e-6);
    }

    #[test]
    fn test_hadamard() {
        let mut qs = MiniQuASIM::new(42);
        qs.hadamard(0);
        
        assert!((qs.measure_prob(0) - 0.5).abs() < 0.01);
        assert!((qs.measure_prob(1) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_bell_state() {
        let mut qs = MiniQuASIM::new(42);
        let (p00, p11) = qs.bell_state();
        
        assert!((p00 - 0.5).abs() < 0.01);
        assert!((p11 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_ghz_state() {
        let mut qs = MiniQuASIM::new(42);
        let (p000, p111) = qs.ghz_state();
        
        assert!((p000 - 0.5).abs() < 0.01);
        assert!((p111 - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_determinism() {
        let mut qs1 = MiniQuASIM::new(42);
        let mut qs2 = MiniQuASIM::new(42);
        
        qs1.bell_state();
        qs2.bell_state();
        
        assert_eq!(qs1.get_state_hash(), qs2.get_state_hash());
    }

    #[test]
    fn test_toffoli() {
        let mut qs = MiniQuASIM::new(42);
        qs.pauli_x(0);
        qs.pauli_x(1);
        qs.toffoli(0, 1, 2);
        
        // Should be in |111⟩ = 7
        assert!((qs.measure_prob(7) - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_gate_history() {
        let mut qs = MiniQuASIM::new(42);
        qs.hadamard(0);
        qs.cnot(0, 1);
        
        let history = qs.get_gate_history();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].gate, "H");
        assert_eq!(history[1].gate, "CNOT");
    }
}
