//! Q-Substrate: Unified Quantum-AI-CodeGen Interface
//!
//! Provides a convergent interface combining:
//! - Quantum circuit simulation
//! - AI intent classification
//! - Code generation
//!
//! # Example
//! ```rust
//! use qratum::q_substrate::{QSubstrate, QuantumGate};
//!
//! let mut qs = QSubstrate::new();
//!
//! // Quantum: Bell state
//! let probs = qs.run_quantum(&[QuantumGate::Hadamard(0), QuantumGate::CNOT(0, 1)]);
//!
//! // AI: Intent classification
//! let intent = qs.classify_intent("generate fibonacci code");
//!
//! // Code gen
//! let code = qs.generate_code("fibonacci function", "rust").unwrap();
//! ```

use alloc::string::{String, ToString};
use alloc::vec::Vec;
use alloc::vec;
use alloc::format;
use core::f64::consts::FRAC_1_SQRT_2;

/// Quantum gate operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QuantumGate {
    /// Hadamard gate on qubit
    Hadamard(usize),
    /// Pauli-X (NOT) gate on qubit
    PauliX(usize),
    /// Pauli-Y gate on qubit
    PauliY(usize),
    /// Pauli-Z gate on qubit
    PauliZ(usize),
    /// CNOT (controlled-NOT) gate: control, target
    CNOT(usize, usize),
    /// CZ (controlled-Z) gate: control, target
    CZ(usize, usize),
    /// T gate (π/8 rotation) on qubit
    T(usize),
    /// S gate (π/4 rotation) on qubit
    S(usize),
    /// Rotation around X axis: qubit, angle (radians)
    RX(usize, f64),
    /// Rotation around Y axis: qubit, angle (radians)
    RY(usize, f64),
    /// Rotation around Z axis: qubit, angle (radians)
    RZ(usize, f64),
    /// Toffoli (CCNOT) gate: control1, control2, target
    Toffoli(usize, usize, usize),
    /// SWAP gate between two qubits
    SWAP(usize, usize),
    /// Measurement on qubit
    Measure(usize),
}

/// Intent classification result
#[derive(Debug, Clone, PartialEq)]
pub struct IntentResult {
    /// Detected intent category
    pub intent: IntentType,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Extracted entities
    pub entities: Vec<(String, String)>,
}

/// Types of intents that can be classified
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IntentType {
    /// Generate code
    CodeGeneration,
    /// Run quantum circuit
    QuantumExecution,
    /// Data analysis
    DataAnalysis,
    /// File operation
    FileOperation,
    /// Search/query
    Search,
    /// Explanation/documentation
    Explain,
    /// Optimization task
    Optimize,
    /// Unknown intent
    Unknown,
}

/// Code generation result
#[derive(Debug, Clone)]
pub struct CodeGenResult {
    /// Generated code
    pub code: String,
    /// Programming language
    pub language: String,
    /// Explanation of the code
    pub explanation: String,
    /// Confidence score
    pub confidence: f64,
}

/// Complex number for quantum state representation
#[derive(Debug, Clone, Copy)]
pub struct Complex {
    pub re: f64,
    pub im: f64,
}

impl Complex {
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }

    pub fn zero() -> Self {
        Self { re: 0.0, im: 0.0 }
    }

    pub fn one() -> Self {
        Self { re: 1.0, im: 0.0 }
    }

    pub fn norm_sq(&self) -> f64 {
        self.re * self.re + self.im * self.im
    }

    pub fn mul(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re * other.re - self.im * other.im,
            im: self.re * other.im + self.im * other.re,
        }
    }

    pub fn add(&self, other: &Complex) -> Complex {
        Complex {
            re: self.re + other.re,
            im: self.im + other.im,
        }
    }

    pub fn scale(&self, s: f64) -> Complex {
        Complex {
            re: self.re * s,
            im: self.im * s,
        }
    }
}

/// Q-Substrate: Unified quantum-AI-codegen interface
pub struct QSubstrate {
    /// Number of qubits in the quantum register
    num_qubits: usize,
    /// Quantum state vector (2^n complex amplitudes)
    state: Vec<Complex>,
    /// Measurement results
    measurements: Vec<bool>,
    /// Execution history for provenance
    history: Vec<String>,
}

impl Default for QSubstrate {
    fn default() -> Self {
        Self::new()
    }
}

impl QSubstrate {
    /// Create a new Q-Substrate with default 4 qubits
    pub fn new() -> Self {
        Self::with_qubits(4)
    }

    /// Create a Q-Substrate with specified number of qubits
    pub fn with_qubits(num_qubits: usize) -> Self {
        let size = 1 << num_qubits; // 2^n
        let mut state = vec![Complex::zero(); size];
        state[0] = Complex::one(); // |000...0⟩ initial state

        Self {
            num_qubits,
            state,
            measurements: Vec::new(),
            history: Vec::new(),
        }
    }

    /// Reset quantum state to |0⟩^n
    pub fn reset(&mut self) {
        let size = 1 << self.num_qubits;
        self.state = vec![Complex::zero(); size];
        self.state[0] = Complex::one();
        self.measurements.clear();
        self.history.push("RESET".to_string());
    }

    /// Run a quantum circuit and return measurement probabilities
    pub fn run_quantum(&mut self, gates: &[QuantumGate]) -> Vec<f64> {
        for gate in gates {
            self.apply_gate(gate);
            self.history.push(format!("{:?}", gate));
        }
        self.get_probabilities()
    }

    /// Apply a single quantum gate
    fn apply_gate(&mut self, gate: &QuantumGate) {
        match gate {
            QuantumGate::Hadamard(q) => self.apply_hadamard(*q),
            QuantumGate::PauliX(q) => self.apply_pauli_x(*q),
            QuantumGate::PauliY(q) => self.apply_pauli_y(*q),
            QuantumGate::PauliZ(q) => self.apply_pauli_z(*q),
            QuantumGate::CNOT(c, t) => self.apply_cnot(*c, *t),
            QuantumGate::CZ(c, t) => self.apply_cz(*c, *t),
            QuantumGate::T(q) => self.apply_t(*q),
            QuantumGate::S(q) => self.apply_s(*q),
            QuantumGate::RX(q, theta) => self.apply_rx(*q, *theta),
            QuantumGate::RY(q, theta) => self.apply_ry(*q, *theta),
            QuantumGate::RZ(q, theta) => self.apply_rz(*q, *theta),
            QuantumGate::Toffoli(c1, c2, t) => self.apply_toffoli(*c1, *c2, *t),
            QuantumGate::SWAP(q1, q2) => self.apply_swap(*q1, *q2),
            QuantumGate::Measure(q) => { self.measure_qubit(*q); }
        }
    }

    /// Apply Hadamard gate to qubit q
    fn apply_hadamard(&mut self, q: usize) {
        let size = self.state.len();
        let mask = 1 << q;

        for i in 0..size {
            if i & mask == 0 {
                let j = i | mask;
                let a = self.state[i];
                let b = self.state[j];
                self.state[i] = a.add(&b).scale(FRAC_1_SQRT_2);
                self.state[j] = a.add(&b.scale(-1.0)).scale(FRAC_1_SQRT_2);
            }
        }
    }

    /// Apply Pauli-X (NOT) gate
    fn apply_pauli_x(&mut self, q: usize) {
        let size = self.state.len();
        let mask = 1 << q;

        for i in 0..size {
            if i & mask == 0 {
                let j = i | mask;
                self.state.swap(i, j);
            }
        }
    }

    /// Apply Pauli-Y gate
    fn apply_pauli_y(&mut self, q: usize) {
        let size = self.state.len();
        let mask = 1 << q;

        for i in 0..size {
            if i & mask == 0 {
                let j = i | mask;
                let a = self.state[i];
                let b = self.state[j];
                self.state[i] = Complex::new(b.im, -b.re);
                self.state[j] = Complex::new(-a.im, a.re);
            }
        }
    }

    /// Apply Pauli-Z gate
    fn apply_pauli_z(&mut self, q: usize) {
        let size = self.state.len();
        let mask = 1 << q;

        for i in 0..size {
            if i & mask != 0 {
                self.state[i] = self.state[i].scale(-1.0);
            }
        }
    }

    /// Apply CNOT gate
    fn apply_cnot(&mut self, control: usize, target: usize) {
        let size = self.state.len();
        let control_mask = 1 << control;
        let target_mask = 1 << target;

        for i in 0..size {
            if (i & control_mask != 0) && (i & target_mask == 0) {
                let j = i | target_mask;
                self.state.swap(i, j);
            }
        }
    }

    /// Apply CZ gate
    fn apply_cz(&mut self, control: usize, target: usize) {
        let size = self.state.len();
        let control_mask = 1 << control;
        let target_mask = 1 << target;

        for i in 0..size {
            if (i & control_mask != 0) && (i & target_mask != 0) {
                self.state[i] = self.state[i].scale(-1.0);
            }
        }
    }

    /// Apply T gate (π/8 rotation)
    fn apply_t(&mut self, q: usize) {
        let size = self.state.len();
        let mask = 1 << q;
        let phase = Complex::new(FRAC_1_SQRT_2, FRAC_1_SQRT_2);

        for i in 0..size {
            if i & mask != 0 {
                self.state[i] = self.state[i].mul(&phase);
            }
        }
    }

    /// Apply S gate (π/4 rotation)
    fn apply_s(&mut self, q: usize) {
        let size = self.state.len();
        let mask = 1 << q;
        let phase = Complex::new(0.0, 1.0);

        for i in 0..size {
            if i & mask != 0 {
                self.state[i] = self.state[i].mul(&phase);
            }
        }
    }

    /// Apply RX rotation
    fn apply_rx(&mut self, q: usize, theta: f64) {
        let size = self.state.len();
        let mask = 1 << q;
        let cos_half = libm::cos(theta / 2.0);
        let sin_half = libm::sin(theta / 2.0);

        for i in 0..size {
            if i & mask == 0 {
                let j = i | mask;
                let a = self.state[i];
                let b = self.state[j];
                self.state[i] = Complex::new(
                    a.re * cos_half + b.im * sin_half,
                    a.im * cos_half - b.re * sin_half,
                );
                self.state[j] = Complex::new(
                    b.re * cos_half + a.im * sin_half,
                    b.im * cos_half - a.re * sin_half,
                );
            }
        }
    }

    /// Apply RY rotation
    fn apply_ry(&mut self, q: usize, theta: f64) {
        let size = self.state.len();
        let mask = 1 << q;
        let cos_half = libm::cos(theta / 2.0);
        let sin_half = libm::sin(theta / 2.0);

        for i in 0..size {
            if i & mask == 0 {
                let j = i | mask;
                let a = self.state[i];
                let b = self.state[j];
                self.state[i] = a.scale(cos_half).add(&b.scale(-sin_half));
                self.state[j] = a.scale(sin_half).add(&b.scale(cos_half));
            }
        }
    }

    /// Apply RZ rotation
    fn apply_rz(&mut self, q: usize, theta: f64) {
        let size = self.state.len();
        let mask = 1 << q;
        let phase_neg = Complex::new(libm::cos(-theta / 2.0), libm::sin(-theta / 2.0));
        let phase_pos = Complex::new(libm::cos(theta / 2.0), libm::sin(theta / 2.0));

        for i in 0..size {
            if i & mask == 0 {
                self.state[i] = self.state[i].mul(&phase_neg);
            } else {
                self.state[i] = self.state[i].mul(&phase_pos);
            }
        }
    }

    /// Apply Toffoli (CCNOT) gate
    fn apply_toffoli(&mut self, c1: usize, c2: usize, target: usize) {
        let size = self.state.len();
        let c1_mask = 1 << c1;
        let c2_mask = 1 << c2;
        let target_mask = 1 << target;

        for i in 0..size {
            if (i & c1_mask != 0) && (i & c2_mask != 0) && (i & target_mask == 0) {
                let j = i | target_mask;
                self.state.swap(i, j);
            }
        }
    }

    /// Apply SWAP gate
    fn apply_swap(&mut self, q1: usize, q2: usize) {
        let size = self.state.len();
        let mask1 = 1 << q1;
        let mask2 = 1 << q2;

        for i in 0..size {
            let bit1 = (i & mask1) != 0;
            let bit2 = (i & mask2) != 0;
            if bit1 != bit2 {
                let j = (i ^ mask1) ^ mask2;
                if i < j {
                    self.state.swap(i, j);
                }
            }
        }
    }

    /// Measure a qubit (collapses state, returns 0 or 1)
    fn measure_qubit(&mut self, q: usize) -> bool {
        let mask = 1 << q;
        let mut prob_one = 0.0;

        for (i, amp) in self.state.iter().enumerate() {
            if i & mask != 0 {
                prob_one += amp.norm_sq();
            }
        }

        // Deterministic "measurement" for simulation (use prob > 0.5)
        let result = prob_one > 0.5;
        self.measurements.push(result);

        // Collapse state
        let normalization = if result {
            1.0 / libm::sqrt(prob_one)
        } else {
            1.0 / libm::sqrt(1.0 - prob_one)
        };

        for i in 0..self.state.len() {
            if (i & mask != 0) != result {
                self.state[i] = Complex::zero();
            } else {
                self.state[i] = self.state[i].scale(normalization);
            }
        }

        result
    }

    /// Get measurement probabilities for all basis states
    pub fn get_probabilities(&self) -> Vec<f64> {
        self.state.iter().map(|c| c.norm_sq()).collect()
    }

    /// Classify intent from natural language input
    pub fn classify_intent(&self, input: &str) -> IntentResult {
        let input_lower = input.to_lowercase();
        let mut entities: Vec<(String, String)> = Vec::new();

        // Keyword-based classification with confidence scoring
        let (intent, confidence) = if input_lower.contains("generate") 
            || input_lower.contains("create") 
            || input_lower.contains("write")
            || input_lower.contains("code") 
        {
            // Extract language entity
            for lang in &["rust", "python", "javascript", "typescript", "go", "java", "c++", "c"] {
                if input_lower.contains(lang) {
                    entities.push(("language".to_string(), lang.to_string()));
                    break;
                }
            }
            // Extract function name hints
            for keyword in &["fibonacci", "sort", "search", "api", "server", "function", "class"] {
                if input_lower.contains(keyword) {
                    entities.push(("target".to_string(), keyword.to_string()));
                }
            }
            (IntentType::CodeGeneration, 0.92)
        } else if input_lower.contains("quantum") 
            || input_lower.contains("qubit")
            || input_lower.contains("circuit")
            || input_lower.contains("entangle")
            || input_lower.contains("superposition")
        {
            (IntentType::QuantumExecution, 0.95)
        } else if input_lower.contains("analyze") 
            || input_lower.contains("data")
            || input_lower.contains("statistics")
        {
            (IntentType::DataAnalysis, 0.85)
        } else if input_lower.contains("file")
            || input_lower.contains("read")
            || input_lower.contains("save")
            || input_lower.contains("load")
        {
            (IntentType::FileOperation, 0.88)
        } else if input_lower.contains("search")
            || input_lower.contains("find")
            || input_lower.contains("query")
        {
            (IntentType::Search, 0.87)
        } else if input_lower.contains("explain")
            || input_lower.contains("what")
            || input_lower.contains("how")
            || input_lower.contains("document")
        {
            (IntentType::Explain, 0.80)
        } else if input_lower.contains("optimize")
            || input_lower.contains("improve")
            || input_lower.contains("faster")
        {
            (IntentType::Optimize, 0.83)
        } else {
            (IntentType::Unknown, 0.40)
        };

        IntentResult {
            intent,
            confidence,
            entities,
        }
    }

    /// Generate code based on description and target language
    pub fn generate_code(&self, description: &str, language: &str) -> Result<CodeGenResult, String> {
        let desc_lower = description.to_lowercase();

        let (code, explanation) = match language.to_lowercase().as_str() {
            "rust" => self.generate_rust_code(&desc_lower),
            "python" => self.generate_python_code(&desc_lower),
            "javascript" | "js" => self.generate_js_code(&desc_lower),
            "typescript" | "ts" => self.generate_ts_code(&desc_lower),
            _ => return Err(format!("Unsupported language: {}", language)),
        };

        Ok(CodeGenResult {
            code,
            language: language.to_string(),
            explanation,
            confidence: 0.88,
        })
    }

    fn generate_rust_code(&self, desc: &str) -> (String, String) {
        if desc.contains("fibonacci") {
            (
                r#"/// Calculate the nth Fibonacci number
pub fn fibonacci(n: u64) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let mut a = 0u64;
            let mut b = 1u64;
            for _ in 2..=n {
                let temp = a + b;
                a = b;
                b = temp;
            }
            b
        }
    }
}

/// Generate Fibonacci sequence up to n terms
pub fn fibonacci_sequence(n: usize) -> Vec<u64> {
    (0..n as u64).map(fibonacci).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci() {
        assert_eq!(fibonacci(0), 0);
        assert_eq!(fibonacci(1), 1);
        assert_eq!(fibonacci(10), 55);
        assert_eq!(fibonacci(20), 6765);
    }
}"#.to_string(),
                "Iterative Fibonacci implementation with O(n) time and O(1) space complexity.".to_string()
            )
        } else if desc.contains("sort") {
            (
                r#"/// Quicksort implementation
pub fn quicksort<T: Ord + Clone>(arr: &mut [T]) {
    if arr.len() <= 1 {
        return;
    }
    let pivot_idx = partition(arr);
    quicksort(&mut arr[0..pivot_idx]);
    quicksort(&mut arr[pivot_idx + 1..]);
}

fn partition<T: Ord + Clone>(arr: &mut [T]) -> usize {
    let len = arr.len();
    let pivot_idx = len / 2;
    arr.swap(pivot_idx, len - 1);
    
    let mut i = 0;
    for j in 0..len - 1 {
        if arr[j] <= arr[len - 1] {
            arr.swap(i, j);
            i += 1;
        }
    }
    arr.swap(i, len - 1);
    i
}"#.to_string(),
                "In-place quicksort with average O(n log n) time complexity.".to_string()
            )
        } else {
            (
                format!(r#"/// TODO: Implement {}
pub fn generated_function() -> Result<(), Box<dyn std::error::Error>> {{
    // Implementation for: {}
    unimplemented!("Generated stub for: {}")
}}"#, desc, desc, desc),
                format!("Generated stub for: {}", desc)
            )
        }
    }

    fn generate_python_code(&self, desc: &str) -> (String, String) {
        if desc.contains("fibonacci") {
            (
                r#"def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def fibonacci_sequence(n: int) -> list[int]:
    """Generate Fibonacci sequence up to n terms."""
    return [fibonacci(i) for i in range(n)]


if __name__ == "__main__":
    print(f"fib(10) = {fibonacci(10)}")
    print(f"sequence: {fibonacci_sequence(10)}")"#.to_string(),
                "Iterative Fibonacci with O(n) time complexity.".to_string()
            )
        } else {
            (
                format!(r#"def generated_function():
    """TODO: Implement {}"""
    raise NotImplementedError("Generated stub for: {}")"#, desc, desc),
                format!("Generated stub for: {}", desc)
            )
        }
    }

    fn generate_js_code(&self, desc: &str) -> (String, String) {
        if desc.contains("fibonacci") {
            (
                r#"/**
 * Calculate the nth Fibonacci number
 * @param {number} n - The position in the Fibonacci sequence
 * @returns {number} The nth Fibonacci number
 */
function fibonacci(n) {
    if (n <= 0) return 0;
    if (n === 1) return 1;
    
    let a = 0, b = 1;
    for (let i = 2; i <= n; i++) {
        [a, b] = [b, a + b];
    }
    return b;
}

/**
 * Generate Fibonacci sequence up to n terms
 * @param {number} n - Number of terms
 * @returns {number[]} Array of Fibonacci numbers
 */
function fibonacciSequence(n) {
    return Array.from({ length: n }, (_, i) => fibonacci(i));
}

module.exports = { fibonacci, fibonacciSequence };"#.to_string(),
                "Iterative Fibonacci implementation in JavaScript.".to_string()
            )
        } else {
            (
                format!(r#"/**
 * TODO: Implement {}
 */
function generatedFunction() {{
    throw new Error("Generated stub for: {}");
}}"#, desc, desc),
                format!("Generated stub for: {}", desc)
            )
        }
    }

    fn generate_ts_code(&self, desc: &str) -> (String, String) {
        if desc.contains("fibonacci") {
            (
                r#"/**
 * Calculate the nth Fibonacci number
 */
export function fibonacci(n: number): number {
    if (n <= 0) return 0;
    if (n === 1) return 1;
    
    let a = 0, b = 1;
    for (let i = 2; i <= n; i++) {
        [a, b] = [b, a + b];
    }
    return b;
}

/**
 * Generate Fibonacci sequence up to n terms
 */
export function fibonacciSequence(n: number): number[] {
    return Array.from({ length: n }, (_, i) => fibonacci(i));
}"#.to_string(),
                "TypeScript Fibonacci with type annotations.".to_string()
            )
        } else {
            (
                format!(r#"/**
 * TODO: Implement {}
 */
export function generatedFunction(): void {{
    throw new Error("Generated stub for: {}");
}}"#, desc, desc),
                format!("Generated stub for: {}", desc)
            )
        }
    }

    /// Get execution history
    pub fn get_history(&self) -> &[String] {
        &self.history
    }

    /// Get measurement results
    pub fn get_measurements(&self) -> &[bool] {
        &self.measurements
    }

    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bell_state() {
        let mut qs = QSubstrate::with_qubits(2);
        let probs = qs.run_quantum(&[
            QuantumGate::Hadamard(0),
            QuantumGate::CNOT(0, 1),
        ]);
        
        // Bell state |00⟩ + |11⟩ should have ~50% probability for 00 and 11
        assert!((probs[0] - 0.5).abs() < 0.001); // |00⟩
        assert!(probs[1].abs() < 0.001);          // |01⟩
        assert!(probs[2].abs() < 0.001);          // |10⟩
        assert!((probs[3] - 0.5).abs() < 0.001); // |11⟩
    }

    #[test]
    fn test_intent_classification() {
        let qs = QSubstrate::new();
        
        let result = qs.classify_intent("generate fibonacci code in rust");
        assert_eq!(result.intent, IntentType::CodeGeneration);
        assert!(result.confidence > 0.8);
        
        let result = qs.classify_intent("run quantum circuit with entanglement");
        assert_eq!(result.intent, IntentType::QuantumExecution);
    }

    #[test]
    fn test_code_generation() {
        let qs = QSubstrate::new();
        
        let result = qs.generate_code("fibonacci function", "rust").unwrap();
        assert!(result.code.contains("fibonacci"));
        assert_eq!(result.language, "rust");
        
        let result = qs.generate_code("fibonacci", "python").unwrap();
        assert!(result.code.contains("def fibonacci"));
    }
}
