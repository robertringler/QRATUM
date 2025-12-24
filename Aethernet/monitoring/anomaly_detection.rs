//! Anomaly Detection Module
//!
//! Real-time monitoring and alerting for TXO propagation, enclave integrity,
//! and operator behavior anomalies.
//!
//! Security Hardening (Aethernet Phase I-II):
//! - Threshold alerts for propagation delays
//! - Integrity violation detection (Merkle proof failures)
//! - Operator deviation monitoring (unusual patterns)
//! - Statistical anomaly detection (Z-score, IQR)

#![no_std]

extern crate alloc;

use alloc::vec::Vec;
use alloc::collections::VecDeque;
use sha3::{Digest, Sha3_256};

/// Anomaly type classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnomalyType {
    /// TXO propagation delay exceeds threshold
    PropagationDelay,
    /// Merkle proof verification failed
    IntegrityViolation,
    /// Operator behavior deviation
    OperatorDeviation,
    /// Enclave attestation failure
    AttestationFailure,
    /// Unusual transaction volume
    VolumeAnomaly,
    /// Consensus timeout
    ConsensusTimeout,
}

/// Anomaly severity level
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub enum Severity {
    /// Informational (no action required)
    Info,
    /// Warning (should investigate)
    Warning,
    /// Critical (immediate action required)
    Critical,
}

/// Anomaly detection event
#[derive(Debug, Clone)]
pub struct AnomalyEvent {
    /// Type of anomaly detected
    pub anomaly_type: AnomalyType,
    /// Severity level
    pub severity: Severity,
    /// Timestamp of detection
    pub timestamp: u64,
    /// Affected entity (TXO hash, operator ID, etc.)
    pub entity_id: [u8; 32],
    /// Metric value that triggered alert
    pub metric_value: f64,
    /// Expected/threshold value
    pub threshold_value: f64,
    /// Human-readable description
    pub description: &'static str,
}

/// Propagation monitoring statistics
#[derive(Debug, Clone, Copy)]
pub struct PropagationStats {
    /// Mean propagation time (microseconds)
    pub mean_time: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum observed time
    pub min_time: u64,
    /// Maximum observed time
    pub max_time: u64,
    /// Sample count
    pub sample_count: usize,
}

/// Operator behavior profile
#[derive(Debug, Clone)]
pub struct OperatorProfile {
    /// Operator identifier
    pub operator_id: [u8; 32],
    /// Typical TXO rate (per hour)
    pub typical_rate: f64,
    /// Typical operation classes
    pub typical_operations: Vec<u32>,
    /// Historical deviation score
    pub baseline_score: f64,
}

/// Anomaly detector configuration
#[derive(Debug, Clone, Copy)]
pub struct DetectorConfig {
    /// Propagation delay threshold (microseconds)
    pub propagation_threshold: u64,
    /// Z-score threshold for statistical anomalies
    pub z_score_threshold: f64,
    /// Window size for moving average
    pub window_size: usize,
    /// Enable operator profiling
    pub operator_profiling: bool,
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            propagation_threshold: 5_000_000, // 5 seconds
            z_score_threshold: 3.0, // 3 standard deviations
            window_size: 100,
            operator_profiling: true,
        }
    }
}

/// Anomaly detector context
///
/// Monitors TXO propagation, enclave integrity, and operator behavior.
pub struct AnomalyDetector {
    /// Detector configuration
    config: DetectorConfig,
    /// Recent propagation times (rolling window)
    propagation_window: VecDeque<u64>,
    /// Detected anomaly events
    events: Vec<AnomalyEvent>,
    /// Operator behavior profiles
    operator_profiles: Vec<OperatorProfile>,
    /// Integrity violation counter
    integrity_violations: u64,
}

impl AnomalyDetector {
    /// Create new anomaly detector
    pub fn new(config: DetectorConfig) -> Self {
        Self {
            config,
            propagation_window: VecDeque::with_capacity(config.window_size),
            events: Vec::new(),
            operator_profiles: Vec::new(),
            integrity_violations: 0,
        }
    }
    
    /// Monitor TXO propagation delay
    ///
    /// # Arguments
    /// * `txo_hash` - TXO identifier
    /// * `propagation_time` - Time taken to propagate (microseconds)
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * Some(AnomalyEvent) if anomaly detected, None otherwise
    pub fn monitor_propagation(
        &mut self,
        txo_hash: [u8; 32],
        propagation_time: u64,
        current_time: u64,
    ) -> Option<AnomalyEvent> {
        // Add to rolling window
        if self.propagation_window.len() >= self.config.window_size {
            self.propagation_window.pop_front();
        }
        self.propagation_window.push_back(propagation_time);
        
        // Check threshold
        if propagation_time > self.config.propagation_threshold {
            let event = AnomalyEvent {
                anomaly_type: AnomalyType::PropagationDelay,
                severity: if propagation_time > self.config.propagation_threshold * 2 {
                    Severity::Critical
                } else {
                    Severity::Warning
                },
                timestamp: current_time,
                entity_id: txo_hash,
                metric_value: propagation_time as f64,
                threshold_value: self.config.propagation_threshold as f64,
                description: "TXO propagation delay exceeds threshold",
            };
            
            self.events.push(event.clone());
            return Some(event);
        }
        
        // Statistical anomaly detection (Z-score)
        if self.propagation_window.len() >= 30 {
            let stats = self.compute_propagation_stats();
            let z_score = ((propagation_time as f64) - stats.mean_time) / stats.std_dev;
            
            if z_score.abs() > self.config.z_score_threshold {
                let event = AnomalyEvent {
                    anomaly_type: AnomalyType::PropagationDelay,
                    severity: Severity::Warning,
                    timestamp: current_time,
                    entity_id: txo_hash,
                    metric_value: z_score,
                    threshold_value: self.config.z_score_threshold,
                    description: "Statistical propagation anomaly detected",
                };
                
                self.events.push(event.clone());
                return Some(event);
            }
        }
        
        None
    }
    
    /// Monitor integrity violations
    ///
    /// # Arguments
    /// * `txo_hash` - TXO with integrity violation
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * AnomalyEvent describing the violation
    pub fn monitor_integrity_violation(
        &mut self,
        txo_hash: [u8; 32],
        current_time: u64,
    ) -> AnomalyEvent {
        self.integrity_violations += 1;
        
        let event = AnomalyEvent {
            anomaly_type: AnomalyType::IntegrityViolation,
            severity: Severity::Critical,
            timestamp: current_time,
            entity_id: txo_hash,
            metric_value: self.integrity_violations as f64,
            threshold_value: 0.0,
            description: "Merkle proof verification failed",
        };
        
        self.events.push(event.clone());
        event
    }
    
    /// Monitor operator behavior
    ///
    /// # Arguments
    /// * `operator_id` - Operator identifier
    /// * `txo_rate` - Current TXO rate (per hour)
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * Some(AnomalyEvent) if deviation detected, None otherwise
    pub fn monitor_operator_behavior(
        &mut self,
        operator_id: [u8; 32],
        txo_rate: f64,
        current_time: u64,
    ) -> Option<AnomalyEvent> {
        if !self.config.operator_profiling {
            return None;
        }
        
        // Find operator profile
        let profile = self.operator_profiles.iter()
            .find(|p| p.operator_id == operator_id);
        
        if let Some(profile) = profile {
            // Compute deviation from baseline
            let deviation = ((txo_rate - profile.typical_rate) / profile.typical_rate).abs();
            
            // Check if deviation is significant
            if deviation > 0.5 { // 50% deviation
                let event = AnomalyEvent {
                    anomaly_type: AnomalyType::OperatorDeviation,
                    severity: if deviation > 1.0 {
                        Severity::Critical
                    } else {
                        Severity::Warning
                    },
                    timestamp: current_time,
                    entity_id: operator_id,
                    metric_value: txo_rate,
                    threshold_value: profile.typical_rate,
                    description: "Operator behavior deviation detected",
                };
                
                self.events.push(event.clone());
                return Some(event);
            }
        }
        
        None
    }
    
    /// Monitor enclave attestation
    ///
    /// # Arguments
    /// * `enclave_id` - Enclave identifier
    /// * `attestation_valid` - Whether attestation succeeded
    /// * `current_time` - Current timestamp
    ///
    /// # Returns
    /// * Some(AnomalyEvent) if attestation failed, None otherwise
    pub fn monitor_attestation(
        &mut self,
        enclave_id: [u8; 32],
        attestation_valid: bool,
        current_time: u64,
    ) -> Option<AnomalyEvent> {
        if !attestation_valid {
            let event = AnomalyEvent {
                anomaly_type: AnomalyType::AttestationFailure,
                severity: Severity::Critical,
                timestamp: current_time,
                entity_id: enclave_id,
                metric_value: 0.0,
                threshold_value: 1.0,
                description: "Enclave attestation verification failed",
            };
            
            self.events.push(event.clone());
            return Some(event);
        }
        
        None
    }
    
    /// Add operator profile for baseline comparison
    pub fn add_operator_profile(&mut self, profile: OperatorProfile) {
        self.operator_profiles.push(profile);
    }
    
    /// Compute propagation statistics
    fn compute_propagation_stats(&self) -> PropagationStats {
        if self.propagation_window.is_empty() {
            return PropagationStats {
                mean_time: 0.0,
                std_dev: 0.0,
                min_time: 0,
                max_time: 0,
                sample_count: 0,
            };
        }
        
        let sum: u64 = self.propagation_window.iter().sum();
        let mean = (sum as f64) / (self.propagation_window.len() as f64);
        
        let variance: f64 = self.propagation_window.iter()
            .map(|&x| {
                let diff = (x as f64) - mean;
                diff * diff
            })
            .sum::<f64>() / (self.propagation_window.len() as f64);
        
        let std_dev = variance.sqrt();
        
        let min_time = *self.propagation_window.iter().min().unwrap();
        let max_time = *self.propagation_window.iter().max().unwrap();
        
        PropagationStats {
            mean_time: mean,
            std_dev,
            min_time,
            max_time,
            sample_count: self.propagation_window.len(),
        }
    }
    
    /// Get all detected anomaly events
    pub fn events(&self) -> &[AnomalyEvent] {
        &self.events
    }
    
    /// Get events by severity
    pub fn events_by_severity(&self, severity: Severity) -> Vec<&AnomalyEvent> {
        self.events.iter()
            .filter(|e| e.severity == severity)
            .collect()
    }
    
    /// Get events by type
    pub fn events_by_type(&self, anomaly_type: AnomalyType) -> Vec<&AnomalyEvent> {
        self.events.iter()
            .filter(|e| e.anomaly_type == anomaly_type)
            .collect()
    }
    
    /// Clear event history (for maintenance)
    pub fn clear_events(&mut self) {
        self.events.clear();
    }
    
    /// Get current statistics
    pub fn statistics(&self) -> DetectorStatistics {
        DetectorStatistics {
            total_events: self.events.len(),
            critical_events: self.events_by_severity(Severity::Critical).len(),
            warning_events: self.events_by_severity(Severity::Warning).len(),
            integrity_violations: self.integrity_violations,
            propagation_stats: self.compute_propagation_stats(),
        }
    }
}

/// Detector runtime statistics
#[derive(Debug, Clone)]
pub struct DetectorStatistics {
    /// Total number of anomaly events
    pub total_events: usize,
    /// Number of critical events
    pub critical_events: usize,
    /// Number of warning events
    pub warning_events: usize,
    /// Total integrity violations
    pub integrity_violations: u64,
    /// Current propagation statistics
    pub propagation_stats: PropagationStats,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_detector_creation() {
        let config = DetectorConfig::default();
        let detector = AnomalyDetector::new(config);
        
        assert_eq!(detector.events.len(), 0);
        assert_eq!(detector.integrity_violations, 0);
    }
    
    #[test]
    fn test_propagation_threshold() {
        let config = DetectorConfig {
            propagation_threshold: 1000,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        
        let txo_hash = [0x42u8; 32];
        
        // Normal propagation time
        let event1 = detector.monitor_propagation(txo_hash, 500, 10000);
        assert!(event1.is_none());
        
        // Exceeds threshold
        let event2 = detector.monitor_propagation(txo_hash, 1500, 10001);
        assert!(event2.is_some());
        assert_eq!(event2.unwrap().anomaly_type, AnomalyType::PropagationDelay);
    }
    
    #[test]
    fn test_integrity_violation() {
        let config = DetectorConfig::default();
        let mut detector = AnomalyDetector::new(config);
        
        let txo_hash = [0x42u8; 32];
        let event = detector.monitor_integrity_violation(txo_hash, 10000);
        
        assert_eq!(event.anomaly_type, AnomalyType::IntegrityViolation);
        assert_eq!(event.severity, Severity::Critical);
        assert_eq!(detector.integrity_violations, 1);
    }
    
    #[test]
    fn test_operator_deviation() {
        let config = DetectorConfig::default();
        let mut detector = AnomalyDetector::new(config);
        
        let operator_id = [0x01u8; 32];
        
        // Add baseline profile
        let profile = OperatorProfile {
            operator_id,
            typical_rate: 100.0,
            typical_operations: vec![1, 2, 3],
            baseline_score: 0.5,
        };
        detector.add_operator_profile(profile);
        
        // Normal rate
        let event1 = detector.monitor_operator_behavior(operator_id, 105.0, 10000);
        assert!(event1.is_none());
        
        // 60% deviation - should trigger warning
        let event2 = detector.monitor_operator_behavior(operator_id, 160.0, 10001);
        assert!(event2.is_some());
        assert_eq!(event2.unwrap().anomaly_type, AnomalyType::OperatorDeviation);
    }
    
    #[test]
    fn test_attestation_failure() {
        let config = DetectorConfig::default();
        let mut detector = AnomalyDetector::new(config);
        
        let enclave_id = [0x42u8; 32];
        
        // Valid attestation
        let event1 = detector.monitor_attestation(enclave_id, true, 10000);
        assert!(event1.is_none());
        
        // Failed attestation
        let event2 = detector.monitor_attestation(enclave_id, false, 10001);
        assert!(event2.is_some());
        assert_eq!(event2.unwrap().severity, Severity::Critical);
    }
    
    #[test]
    fn test_propagation_stats() {
        let config = DetectorConfig::default();
        let mut detector = AnomalyDetector::new(config);
        
        let txo_hash = [0x42u8; 32];
        
        // Add some samples
        for i in 1..=10 {
            detector.monitor_propagation(txo_hash, i * 100, 10000 + i);
        }
        
        let stats = detector.compute_propagation_stats();
        
        assert_eq!(stats.sample_count, 10);
        assert_eq!(stats.min_time, 100);
        assert_eq!(stats.max_time, 1000);
        assert!(stats.mean_time > 0.0);
    }
    
    #[test]
    fn test_event_filtering() {
        let config = DetectorConfig::default();
        let mut detector = AnomalyDetector::new(config);
        
        let txo_hash = [0x42u8; 32];
        
        // Generate different types of events
        detector.monitor_integrity_violation(txo_hash, 10000);
        detector.monitor_attestation(txo_hash, false, 10001);
        
        // Filter by severity
        let critical = detector.events_by_severity(Severity::Critical);
        assert_eq!(critical.len(), 2);
        
        // Filter by type
        let integrity = detector.events_by_type(AnomalyType::IntegrityViolation);
        assert_eq!(integrity.len(), 1);
    }
    
    #[test]
    fn test_statistics() {
        let config = DetectorConfig::default();
        let mut detector = AnomalyDetector::new(config);
        
        let txo_hash = [0x42u8; 32];
        
        // Generate some events
        detector.monitor_integrity_violation(txo_hash, 10000);
        detector.monitor_propagation(txo_hash, 100, 10001);
        
        let stats = detector.statistics();
        
        assert!(stats.total_events >= 1);
        assert_eq!(stats.integrity_violations, 1);
    }
    
    #[test]
    fn test_rolling_window() {
        let config = DetectorConfig {
            window_size: 5,
            ..Default::default()
        };
        let mut detector = AnomalyDetector::new(config);
        
        let txo_hash = [0x42u8; 32];
        
        // Add more samples than window size
        for i in 1..=10 {
            detector.monitor_propagation(txo_hash, i * 100, 10000 + i);
        }
        
        // Window should be limited to 5
        assert_eq!(detector.propagation_window.len(), 5);
    }
}
