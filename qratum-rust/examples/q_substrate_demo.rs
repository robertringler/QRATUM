//! Q-Substrate Demo: Quantum + AI + Code Generation
//!
//! This example demonstrates the unified Q-Substrate interface that combines:
//! - Quantum circuit simulation (Bell state, entanglement)
//! - AI intent classification
//! - Code generation in multiple languages

use qratum::q_substrate::{QSubstrate, QuantumGate};

fn main() {
    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    Q-SUBSTRATE DEMONSTRATION");
    println!("           Quantum + AI + Code Generation Interface");
    println!("═══════════════════════════════════════════════════════════════════\n");

    let mut qs = QSubstrate::with_qubits(2);

    // ═══════════════════════════════════════════════════════════════════════
    // QUANTUM: Bell State Creation
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ QUANTUM SIMULATION: Bell State                                 │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let probs = qs.run_quantum(&[
        QuantumGate::Hadamard(0),
        QuantumGate::CNOT(0, 1),
    ]);

    println!("Circuit: H(0) → CNOT(0,1)");
    println!("Creates maximally entangled Bell state: |Φ+⟩ = (|00⟩ + |11⟩)/√2\n");
    println!("Measurement Probabilities:");
    println!("  |00⟩ → {:.4} ({:.1}%)", probs[0], probs[0] * 100.0);
    println!("  |01⟩ → {:.4} ({:.1}%)", probs[1], probs[1] * 100.0);
    println!("  |10⟩ → {:.4} ({:.1}%)", probs[2], probs[2] * 100.0);
    println!("  |11⟩ → {:.4} ({:.1}%)", probs[3], probs[3] * 100.0);
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // AI: Intent Classification
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ AI INTENT CLASSIFICATION                                       │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let inputs = [
        "generate fibonacci code in rust",
        "run quantum circuit with entanglement",
        "analyze the data for patterns",
        "explain how quicksort works",
        "optimize this algorithm for speed",
    ];

    for input in &inputs {
        let intent = qs.classify_intent(input);
        println!("Input: \"{}\"", input);
        println!("  Intent: {:?}", intent.intent);
        println!("  Confidence: {:.1}%", intent.confidence * 100.0);
        if !intent.entities.is_empty() {
            println!("  Entities: {:?}", intent.entities);
        }
        println!();
    }

    // ═══════════════════════════════════════════════════════════════════════
    // CODE GENERATION
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ CODE GENERATION                                                │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    // Rust Fibonacci
    match qs.generate_code("fibonacci function", "rust") {
        Ok(result) => {
            println!("═══ Rust Fibonacci ═══");
            println!("Confidence: {:.1}%", result.confidence * 100.0);
            println!("Explanation: {}\n", result.explanation);
            println!("```rust");
            println!("{}", result.code);
            println!("```\n");
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    // Python Fibonacci
    match qs.generate_code("fibonacci function", "python") {
        Ok(result) => {
            println!("═══ Python Fibonacci ═══");
            println!("Confidence: {:.1}%", result.confidence * 100.0);
            println!("Explanation: {}\n", result.explanation);
            println!("```python");
            println!("{}", result.code);
            println!("```\n");
        }
        Err(e) => eprintln!("Error: {}", e),
    }

    // ═══════════════════════════════════════════════════════════════════════
    // ADVANCED QUANTUM: GHZ State
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ ADVANCED QUANTUM: GHZ State (3 qubits)                         │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    let mut qs3 = QSubstrate::with_qubits(3);
    let ghz_probs = qs3.run_quantum(&[
        QuantumGate::Hadamard(0),
        QuantumGate::CNOT(0, 1),
        QuantumGate::CNOT(1, 2),
    ]);

    println!("Circuit: H(0) → CNOT(0,1) → CNOT(1,2)");
    println!("Creates GHZ state: (|000⟩ + |111⟩)/√2\n");
    println!("Measurement Probabilities:");
    for (i, prob) in ghz_probs.iter().enumerate() {
        if *prob > 0.01 {
            println!("  |{:03b}⟩ → {:.4} ({:.1}%)", i, prob, prob * 100.0);
        }
    }
    println!();

    // ═══════════════════════════════════════════════════════════════════════
    // EXECUTION HISTORY
    // ═══════════════════════════════════════════════════════════════════════
    println!("┌─────────────────────────────────────────────────────────────────┐");
    println!("│ EXECUTION HISTORY (Provenance Trail)                           │");
    println!("└─────────────────────────────────────────────────────────────────┘");
    println!();

    for (i, op) in qs.get_history().iter().enumerate() {
        println!("  [{}] {}", i + 1, op);
    }
    println!();

    println!("═══════════════════════════════════════════════════════════════════");
    println!("                    DEMONSTRATION COMPLETE");
    println!("═══════════════════════════════════════════════════════════════════");
}
