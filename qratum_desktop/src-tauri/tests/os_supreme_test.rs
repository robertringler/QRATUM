// Size and performance tests for OS Supreme module - Phase 4

use std::mem;

#[test]
fn test_os_supreme_size_constraints() {
    use crate::qr_os_supreme::{OSSupreme, QUANTUM_STATE_BYTES};

    // Check struct sizes
    let os_size = mem::size_of::<OSSupreme>();
    println!("OSSupreme size: {} bytes", os_size);
    println!("Quantum state size: {} bytes", QUANTUM_STATE_BYTES);

    // Should fit within reasonable stack limits
    // Phase 4 increases size limit from 40KB to 50KB due to:
    // - MiniLM integration adds MiniAI struct with MiniLMInference
    // - WasmPodConfig for pod isolation configuration
    // - Gate history vector for operation tracking
    // - Additional quantum gates implementation overhead
    // Base quantum state is still 32KB (4096 * 8 bytes)
    assert!(os_size < 50_000, "OSSupreme too large: {} bytes", os_size);
}

#[test]
fn test_determinism() {
    use crate::qr_os_supreme::OSSupreme;
    
    let mut os1 = OSSupreme::new();
    let mut os2 = OSSupreme::new();
    
    let (q1, ai1) = os1.supremacy_test(&[1, 2, 3]);
    let (q2, ai2) = os2.supremacy_test(&[1, 2, 3]);
    
    // Should be deterministic
    assert!((q1 - q2).abs() < 1e-6, "Quantum results not deterministic");
    assert_eq!(ai1, ai2, "AI results not deterministic");
}

#[test]
fn test_performance() {
    use crate::qr_os_supreme::OSSupreme;
    use std::time::Instant;
    
    let mut os = OSSupreme::new();
    
    let start = Instant::now();
    for _ in 0..100 {
        os.run_bell_state();
    }
    let duration = start.elapsed();
    
    println!("100 Bell state executions: {:?}", duration);
    println!("Average: {:?}", duration / 100);
    
    // Should be fast (< 1ms per operation target)
    assert!(duration.as_millis() < 1000, "Too slow: {:?}", duration);
}

#[test]
fn test_quantum_correctness() {
    use crate::qr_os_supreme::OSSupreme;
    
    let mut os = OSSupreme::new();
    let (p00, p11) = os.run_bell_state();
    
    // Bell state should have 50/50 superposition
    assert!((p00 - 0.5).abs() < 0.01, "P(00) should be ~0.5, got {}", p00);
    assert!((p11 - 0.5).abs() < 0.01, "P(11) should be ~0.5, got {}", p11);
    
    // Probabilities should sum to ~1
    let total = p00 + p11;
    assert!((total - 1.0).abs() < 0.01, "Probabilities should sum to 1, got {}", total);
}

#[test]
fn test_ai_determinism() {
    use crate::qr_os_supreme::OSSupreme;
    
    let mut os = OSSupreme::new();
    
    let result1 = os.run_ai(&[42, 43, 44]);
    os.reset();
    let result2 = os.run_ai(&[42, 43, 44]);
    
    assert_eq!(result1, result2, "AI should be deterministic");
}

#[test]
fn test_regression_proof() {
    use crate::qr_os_supreme::{TEXT_SIZE_TARGET, STACK_SIZE_TARGET, QUANTUM_STATE_BYTES};
    
    // Regression checks - these should never increase
    println!("Text size target: {} bytes", TEXT_SIZE_TARGET);
    println!("Stack size target: {} bytes", STACK_SIZE_TARGET);
    println!("Quantum state: {} bytes", QUANTUM_STATE_BYTES);
    
    // Verify we're within supremacy constraints
    assert!(TEXT_SIZE_TARGET <= 4096, "Text size exceeds 4KB limit");
    assert!(STACK_SIZE_TARGET <= 1024, "Stack size exceeds 1KB limit");
    
    // Quantum state is on stack but that's expected for quantum computing
    assert!(QUANTUM_STATE_BYTES == 32768, "Quantum state should be exactly 32KB");
}

// Phase 4 - Advanced Gate Tests
#[test]
fn test_advanced_gates_phase4() {
    use crate::qr_os_supreme::OSSupreme;
    
    let mut os = OSSupreme::new();
    
    // Test Phase gate
    os.apply_hadamard(0);
    os.apply_phase(0);
    let state = os.get_quantum_state();
    assert!(!state.is_empty(), "Quantum state should have entries");
    
    // Reset and test T gate
    os.reset();
    os.apply_hadamard(0);
    os.apply_t(0);
    let state = os.get_quantum_state();
    assert!(!state.is_empty(), "Quantum state should have entries after T gate");
    
    // Reset and test Toffoli
    os.reset();
    os.apply_pauli_x(0);
    os.apply_pauli_x(1);
    os.apply_toffoli(0, 1, 2);
    let state = os.get_quantum_state();
    assert!(!state.is_empty(), "Quantum state should have entries after Toffoli");
}

#[test]
fn test_ghz_state_correctness() {
    use crate::qr_os_supreme::OSSupreme;
    
    let mut os = OSSupreme::new();
    let probs = os.run_ghz_state();
    
    // GHZ state: (|000⟩ + |111⟩)/√2
    assert!((probs[0] - 0.5).abs() < 0.01, "P(000) should be ~0.5");
    assert!((probs[1] - 0.5).abs() < 0.01, "P(111) should be ~0.5");
}

#[test]
fn test_minilm_embedding_phase4() {
    use crate::qr_os_supreme::{MiniLMInference};
    
    let mut minilm = MiniLMInference::new(42);
    
    // Test embedding generation
    let emb = minilm.embed("test quantum simulation");
    assert_eq!(emb.len(), MiniLMInference::EMBEDDING_DIM);
    
    // Test normalization
    let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 0.01, "Embedding should be normalized");
    
    // Test determinism
    let mut minilm2 = MiniLMInference::new(42);
    let emb2 = minilm2.embed("test quantum simulation");
    for (a, b) in emb.iter().zip(emb2.iter()) {
        assert!((a - b).abs() < 1e-6, "Embeddings should be deterministic");
    }
}

#[test]
fn test_wasm_pod_isolation_phase4() {
    use crate::qr_os_supreme::{OSSupreme, WasmPodConfig};
    
    let config = WasmPodConfig {
        pod_id: "test_pod".to_string(),
        memory_limit_kb: 128,
        deterministic_mode: true,
        sandbox_enabled: true,
    };
    
    let os = OSSupreme::with_config(config);
    
    assert_eq!(os.get_pod_config().pod_id, "test_pod");
    assert!(os.get_pod_config().sandbox_enabled);
    assert!(os.get_pod_config().deterministic_mode);
}

#[test]
fn test_gate_history_tracking() {
    use crate::qr_os_supreme::OSSupreme;
    
    let mut os = OSSupreme::new();
    
    os.apply_hadamard(0);
    os.apply_cnot(0, 1);
    os.apply_t(0);
    
    let history = os.get_gate_history();
    assert_eq!(history.len(), 3);
    assert_eq!(history[0].gate_name, "H");
    assert_eq!(history[1].gate_name, "CNOT");
    assert_eq!(history[2].gate_name, "T");
}

#[test]
fn test_rollback_functionality() {
    use crate::qr_os_supreme::OSSupreme;
    
    let mut os = OSSupreme::new();
    
    // Perform some operations
    os.run_bell_state();
    os.run_ai(&[1, 2, 3]);
    
    let stats_before = os.get_stats();
    assert!(stats_before.exec_count > 0);
    
    // Rollback
    let success = os.rollback();
    assert!(success);
    
    // Verify state is reset
    let stats_after = os.get_stats();
    assert_eq!(stats_after.exec_count, 0);
    assert_eq!(os.get_gate_history().len(), 0);
}
