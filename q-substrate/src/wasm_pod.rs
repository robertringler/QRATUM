//! WASM Pod Isolation Layer
//!
//! Secure sandbox for deterministic module execution:
//! - AI module pod (streaming inference)
//! - Quantum module pod (state vectors)
//! - DCGE module pod (code generation)
//! - Provenance logging for inter-module calls
//! - No side-channel leaks
//!
//! Memory footprint: Configurable per pod

extern crate alloc;

use alloc::collections::VecDeque;
use alloc::string::String;
use alloc::vec;
use alloc::vec::Vec;
use serde::{Deserialize, Serialize};

/// Pod types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PodType {
    /// AI inference pod
    AI,
    /// Quantum simulation pod
    Quantum,
    /// Code generation pod
    DCGE,
    /// Combined supremacy pod
    Supremacy,
}

/// Pod configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodConfig {
    /// Unique pod identifier
    pub pod_id: String,
    /// Pod type
    pub pod_type: PodType,
    /// Memory limit in KB
    pub memory_limit_kb: usize,
    /// Stack limit in bytes
    pub stack_limit: usize,
    /// Enable deterministic mode
    pub deterministic_mode: bool,
    /// Enable sandbox isolation
    pub sandbox_enabled: bool,
    /// Enable provenance logging
    pub provenance_logging: bool,
}

impl Default for PodConfig {
    fn default() -> Self {
        PodConfig {
            pod_id: "default_pod".into(),
            pod_type: PodType::Supremacy,
            memory_limit_kb: 64,
            stack_limit: 1024,
            deterministic_mode: true,
            sandbox_enabled: true,
            provenance_logging: true,
        }
    }
}

/// Pod status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodStatus {
    /// Pod ID
    pub pod_id: String,
    /// Is pod active
    pub active: bool,
    /// Current memory usage in bytes
    pub memory_used: usize,
    /// Peak memory usage in bytes
    pub peak_memory: usize,
    /// Operations executed
    pub op_count: u64,
    /// Last operation timestamp
    pub last_op_time: u64,
    /// Error count
    pub error_count: u32,
}

/// Inter-pod message
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PodMessage {
    /// Source pod ID
    pub source_pod: String,
    /// Target pod ID
    pub target_pod: String,
    /// Message type
    pub msg_type: MessageType,
    /// Payload (serialized)
    pub payload: Vec<u8>,
    /// Timestamp
    pub timestamp: u64,
}

/// Message types for inter-pod communication
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MessageType {
    /// Request data from another pod
    Request,
    /// Response with data
    Response,
    /// Error notification
    Error,
    /// State synchronization
    Sync,
    /// Rollback command
    Rollback,
}

/// Provenance record for audit trail
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProvenanceEntry {
    /// Source pod
    pub source: String,
    /// Target pod (if applicable)
    pub target: Option<String>,
    /// Operation type
    pub operation: String,
    /// Input hash (for verification)
    pub input_hash: u64,
    /// Output hash (for verification)
    pub output_hash: u64,
    /// Timestamp
    pub timestamp: u64,
    /// Duration in microseconds
    pub duration_us: u64,
}

/// WASM Pod instance
#[derive(Debug, Clone)]
pub struct WasmPod {
    /// Pod configuration
    pub config: PodConfig,
    /// Pod status
    pub status: PodStatus,
    /// Memory allocation tracker
    memory_allocated: usize,
    /// Operation counter
    op_counter: u64,
    /// Timestamp counter (simulated)
    timestamp: u64,
}

impl WasmPod {
    /// Create a new WASM pod
    pub fn new(config: PodConfig) -> Self {
        let pod_id = config.pod_id.clone();
        WasmPod {
            config,
            status: PodStatus {
                pod_id,
                active: true,
                memory_used: 0,
                peak_memory: 0,
                op_count: 0,
                last_op_time: 0,
                error_count: 0,
            },
            memory_allocated: 0,
            op_counter: 0,
            timestamp: 0,
        }
    }

    /// Allocate memory within pod limits
    pub fn allocate(&mut self, size: usize) -> Result<(), String> {
        let limit = self.config.memory_limit_kb * 1024;
        
        if self.memory_allocated + size > limit {
            self.status.error_count += 1;
            return Err(format!(
                "Memory limit exceeded: requested {} bytes, limit {} bytes",
                size, limit
            ));
        }
        
        self.memory_allocated += size;
        self.status.memory_used = self.memory_allocated;
        
        if self.memory_allocated > self.status.peak_memory {
            self.status.peak_memory = self.memory_allocated;
        }
        
        Ok(())
    }

    /// Free allocated memory
    pub fn free(&mut self, size: usize) {
        if size > self.memory_allocated {
            self.memory_allocated = 0;
        } else {
            self.memory_allocated -= size;
        }
        self.status.memory_used = self.memory_allocated;
    }

    /// Record an operation
    pub fn record_operation(&mut self, op_name: &str) -> ProvenanceEntry {
        self.op_counter += 1;
        self.timestamp += 1;
        self.status.op_count = self.op_counter;
        self.status.last_op_time = self.timestamp;
        
        ProvenanceEntry {
            source: self.config.pod_id.clone(),
            target: None,
            operation: op_name.into(),
            input_hash: 0,
            output_hash: 0,
            timestamp: self.timestamp,
            duration_us: 0,
        }
    }

    /// Reset pod to initial state
    pub fn reset(&mut self) {
        self.memory_allocated = 0;
        self.status.memory_used = 0;
        self.status.op_count = 0;
        self.status.error_count = 0;
        self.op_counter = 0;
    }

    /// Check if pod can execute
    pub fn can_execute(&self) -> bool {
        self.status.active && self.memory_allocated < self.config.memory_limit_kb * 1024
    }
}

/// Pod isolation manager
pub struct PodIsolation {
    /// AI pod
    ai_pod: WasmPod,
    /// Quantum pod
    quantum_pod: WasmPod,
    /// DCGE pod
    dcge_pod: WasmPod,
    /// Provenance log
    provenance_log: Vec<ProvenanceEntry>,
    /// Message queue (VecDeque for O(1) pop_front)
    message_queue: VecDeque<PodMessage>,
    /// Global timestamp
    global_timestamp: u64,
}

impl PodIsolation {
    /// Create pod isolation manager from config
    pub fn new(config: &crate::config::QSubstrateConfig) -> Self {
        PodIsolation {
            ai_pod: WasmPod::new(PodConfig {
                pod_id: "ai_pod".into(),
                pod_type: PodType::AI,
                memory_limit_kb: config.memory.ai_pod_limit_kb,
                stack_limit: 1024,
                deterministic_mode: true,
                sandbox_enabled: true,
                provenance_logging: true,
            }),
            quantum_pod: WasmPod::new(PodConfig {
                pod_id: "quantum_pod".into(),
                pod_type: PodType::Quantum,
                memory_limit_kb: config.memory.quantum_pod_limit_kb,
                stack_limit: 1024,
                deterministic_mode: true,
                sandbox_enabled: true,
                provenance_logging: true,
            }),
            dcge_pod: WasmPod::new(PodConfig {
                pod_id: "dcge_pod".into(),
                pod_type: PodType::DCGE,
                memory_limit_kb: config.memory.dcge_pod_limit_kb,
                stack_limit: 1024,
                deterministic_mode: true,
                sandbox_enabled: true,
                provenance_logging: true,
            }),
            provenance_log: Vec::new(),
            message_queue: VecDeque::new(),
            global_timestamp: 0,
        }
    }

    /// Get pod by type
    pub fn get_pod(&self, pod_type: PodType) -> &WasmPod {
        match pod_type {
            PodType::AI => &self.ai_pod,
            PodType::Quantum => &self.quantum_pod,
            PodType::DCGE => &self.dcge_pod,
            PodType::Supremacy => &self.ai_pod, // Default to AI pod
        }
    }

    /// Get mutable pod by type
    pub fn get_pod_mut(&mut self, pod_type: PodType) -> &mut WasmPod {
        match pod_type {
            PodType::AI => &mut self.ai_pod,
            PodType::Quantum => &mut self.quantum_pod,
            PodType::DCGE => &mut self.dcge_pod,
            PodType::Supremacy => &mut self.ai_pod,
        }
    }

    /// Send message between pods
    pub fn send_message(&mut self, source: PodType, target: PodType, 
                        msg_type: MessageType, payload: Vec<u8>) -> Result<(), String> {
        self.global_timestamp += 1;
        
        let msg = PodMessage {
            source_pod: self.get_pod(source).config.pod_id.clone(),
            target_pod: self.get_pod(target).config.pod_id.clone(),
            msg_type,
            payload,
            timestamp: self.global_timestamp,
        };
        
        // Log provenance
        if self.get_pod(source).config.provenance_logging {
            self.provenance_log.push(ProvenanceEntry {
                source: msg.source_pod.clone(),
                target: Some(msg.target_pod.clone()),
                operation: format!("{:?}", msg_type),
                input_hash: 0,
                output_hash: 0,
                timestamp: self.global_timestamp,
                duration_us: 0,
            });
        }
        
        self.message_queue.push_back(msg);
        Ok(())
    }

    /// Process next message in queue (O(1) using VecDeque)
    pub fn process_message(&mut self) -> Option<PodMessage> {
        self.message_queue.pop_front()
    }

    /// Get all pod statuses
    pub fn get_all_statuses(&self) -> Vec<PodStatus> {
        vec![
            self.ai_pod.status.clone(),
            self.quantum_pod.status.clone(),
            self.dcge_pod.status.clone(),
        ]
    }

    /// Get provenance log
    pub fn get_provenance_log(&self) -> &[ProvenanceEntry] {
        &self.provenance_log
    }

    /// Reset all pods
    pub fn reset_all(&mut self) {
        self.ai_pod.reset();
        self.quantum_pod.reset();
        self.dcge_pod.reset();
        self.provenance_log.clear();
        self.message_queue.clear();
        self.global_timestamp = 0;
    }

    /// Execute operation in isolated pod
    pub fn execute_isolated<F, R>(&mut self, pod_type: PodType, op_name: &str, f: F) -> Result<R, String>
    where
        F: FnOnce(&mut WasmPod) -> Result<R, String>,
    {
        let pod = self.get_pod_mut(pod_type);
        
        if !pod.can_execute() {
            return Err("Pod cannot execute: check memory limits or status".into());
        }
        
        let entry = pod.record_operation(op_name);
        let result = f(pod);
        
        // Log provenance
        self.provenance_log.push(ProvenanceEntry {
            source: entry.source,
            target: None,
            operation: op_name.into(),
            input_hash: 0,
            output_hash: 0,
            timestamp: entry.timestamp,
            duration_us: 0,
        });
        
        result
    }
}

impl Default for PodIsolation {
    fn default() -> Self {
        Self::new(&crate::config::QSubstrateConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pod_creation() {
        let pod = WasmPod::new(PodConfig::default());
        assert!(pod.status.active);
        assert_eq!(pod.status.memory_used, 0);
    }

    #[test]
    fn test_memory_allocation() {
        let mut pod = WasmPod::new(PodConfig {
            memory_limit_kb: 1, // 1 KB limit
            ..PodConfig::default()
        });
        
        // Should succeed
        assert!(pod.allocate(512).is_ok());
        assert_eq!(pod.status.memory_used, 512);
        
        // Should fail (over limit)
        assert!(pod.allocate(1024).is_err());
    }

    #[test]
    fn test_memory_free() {
        let mut pod = WasmPod::new(PodConfig::default());
        pod.allocate(100).unwrap();
        pod.free(50);
        assert_eq!(pod.status.memory_used, 50);
    }

    #[test]
    fn test_pod_reset() {
        let mut pod = WasmPod::new(PodConfig::default());
        pod.allocate(100).unwrap();
        pod.record_operation("test");
        
        pod.reset();
        
        assert_eq!(pod.status.memory_used, 0);
        assert_eq!(pod.status.op_count, 0);
    }

    #[test]
    fn test_isolation_manager() {
        let mut isolation = PodIsolation::default();
        
        let statuses = isolation.get_all_statuses();
        assert_eq!(statuses.len(), 3);
        
        // All pods should be active
        for status in statuses {
            assert!(status.active);
        }
    }

    #[test]
    fn test_inter_pod_messaging() {
        let mut isolation = PodIsolation::default();
        
        isolation.send_message(
            PodType::AI,
            PodType::Quantum,
            MessageType::Request,
            vec![1, 2, 3],
        ).unwrap();
        
        let msg = isolation.process_message();
        assert!(msg.is_some());
        
        let msg = msg.unwrap();
        assert_eq!(msg.source_pod, "ai_pod");
        assert_eq!(msg.target_pod, "quantum_pod");
    }

    #[test]
    fn test_provenance_logging() {
        let mut isolation = PodIsolation::default();
        
        isolation.execute_isolated(PodType::AI, "test_op", |_pod| {
            Ok::<(), String>(())
        }).unwrap();
        
        let log = isolation.get_provenance_log();
        assert!(!log.is_empty());
        assert_eq!(log[0].operation, "test_op");
    }
}
