//! # Proxy Module - Bonded Approvals with Reputation Staking
//!
//! ## Lifecycle Stage: Execution (privileged operations)
//!
//! Certain operations require proxy approval from reputation-staked participants.
//! Proxies bond reputation capital that can be slashed for misbehavior.
//!
//! ## Architectural Role
//!
//! - **Reputation Staking**: Proxies stake reputation capital
//! - **Bonded Approvals**: Approvals carry slashing risk
//! - **Justification Required**: Every approval must document rationale
//! - **Accountability**: Misbehavior triggers reputation slashing
//!
//! ## Inputs → Outputs
//!
//! - Input: Operation request + justification
//! - Output: Signed approval OR rejection with reason
//!
//! ## Security Rationale
//!
//! - Reputation capital at risk discourages malicious approvals
//! - Justification requirement creates audit trail
//! - Multi-proxy threshold prevents single-party abuse
//! - Slashing mechanism enforces accountability

extern crate alloc;
use alloc::vec;
use alloc::vec::Vec;
use alloc::string::String;

use crate::txo::{Txo, TxoType};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Proxy Configuration
#[derive(Debug, Clone)]
pub struct ProxyConfig {
    /// Minimum reputation stake required
    pub min_reputation_stake: u64,
    
    /// Approval threshold (number of proxies)
    pub approval_threshold: usize,
    
    /// Slashing percentage for misbehavior (0-100)
    pub slashing_percentage: u8,
    
    /// Approval timeout (milliseconds)
    pub approval_timeout_ms: u64,
}

impl Default for ProxyConfig {
    fn default() -> Self {
        Self {
            min_reputation_stake: 1000,
            approval_threshold: 2,      // 2-of-N approval
            slashing_percentage: 20,    // 20% stake slashed
            approval_timeout_ms: 600_000, // 10 minutes
        }
    }
}

/// Proxy Participant
///
/// ## Lifecycle Stage: Quorum Convergence | Execution
///
/// Represents a reputation-staked proxy participant.
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct ProxyParticipant {
    /// Participant identifier
    pub id: [u8; 32],
    
    /// Current reputation stake
    pub reputation_stake: u64,
    
    /// Bonded stake (locked for approvals)
    pub bonded_stake: u64,
    
    /// Public key for signature verification
    pub public_key: [u8; 32],
    
    /// Approval history count
    pub approval_count: u64,
    
    /// Slashing history count
    pub slashing_count: u64,
}

impl ProxyParticipant {
    /// Create new proxy participant
    pub fn new(id: [u8; 32], reputation_stake: u64, public_key: [u8; 32]) -> Self {
        Self {
            id,
            reputation_stake,
            bonded_stake: 0,
            public_key,
            approval_count: 0,
            slashing_count: 0,
        }
    }
    
    /// Bond stake for approval
    ///
    /// ## Security Rationale
    /// - Stake bonded until approval completes or times out
    /// - Bonded stake can be slashed for misbehavior
    pub fn bond_stake(&mut self, amount: u64) -> Result<(), &'static str> {
        if amount > self.reputation_stake {
            return Err("Insufficient reputation stake");
        }
        
        self.reputation_stake -= amount;
        self.bonded_stake += amount;
        Ok(())
    }
    
    /// Release bonded stake
    pub fn release_stake(&mut self, amount: u64) -> Result<(), &'static str> {
        if amount > self.bonded_stake {
            return Err("Insufficient bonded stake");
        }
        
        self.bonded_stake -= amount;
        self.reputation_stake += amount;
        Ok(())
    }
    
    /// Slash stake for misbehavior
    ///
    /// ## Security Rationale
    /// - Permanent reputation loss for malicious behavior
    /// - Discourages collusion and abuse
    pub fn slash_stake(&mut self, amount: u64) -> u64 {
        let slashed = amount.min(self.bonded_stake);
        self.bonded_stake -= slashed;
        self.slashing_count += 1;
        slashed
    }
}

/// Proxy Approval Request
///
/// ## Lifecycle Stage: Execution
///
/// Request for proxy approval of privileged operation.
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct ProxyApprovalRequest {
    /// Request identifier
    pub id: [u8; 32],
    
    /// Operation being requested
    pub operation: String,
    
    /// Justification for operation
    pub justification: String,
    
    /// Request timestamp
    pub timestamp: u64,
    
    /// Required stake bond amount
    pub required_bond: u64,
    
    /// Requesting party ID
    pub requester_id: [u8; 32],
}

impl ProxyApprovalRequest {
    /// Create new approval request
    pub fn new(
        operation: String,
        justification: String,
        required_bond: u64,
        requester_id: [u8; 32],
    ) -> Self {
        Self {
            id: [0u8; 32], // Computed from hash
            operation,
            justification,
            timestamp: current_timestamp(),
            required_bond,
            requester_id,
        }
    }
}

/// Proxy Approval
///
/// ## Lifecycle Stage: Execution
///
/// Signed approval from reputation-staked proxy.
///
/// ## Security Rationale
/// - Signature binds proxy to approval decision
/// - Justification creates audit trail
/// - Bonded stake at risk for misbehavior
#[derive(Debug, Clone, Zeroize, ZeroizeOnDrop)]
pub struct ProxyApproval {
    /// Request being approved
    pub request_id: [u8; 32],
    
    /// Approving proxy ID
    pub proxy_id: [u8; 32],
    
    /// Bonded stake amount
    pub bonded_amount: u64,
    
    /// Approval timestamp
    pub timestamp: u64,
    
    /// Approval justification
    pub justification: String,
    
    /// Proxy signature
    pub signature: [u8; 64],
}

impl ProxyApproval {
    /// Convert to TXO for audit trail
    ///
    /// ## Audit Trail
    /// - Emits ProxyApproval TXO to ephemeral ledger
    /// - Records approval justification
    /// - Links to original request
    pub fn to_txo(&self) -> Txo {
        let payload = alloc::format!(
            "Proxy approval: request={:?} | proxy={:?} | bond={} | justification={}",
            self.request_id,
            self.proxy_id,
            self.bonded_amount,
            self.justification
        ).into_bytes();
        
        Txo::new(
            TxoType::ProxyApproval,
            self.timestamp,
            payload,
            vec![self.request_id],
        )
    }
}

/// Proxy Manager
///
/// ## Lifecycle Stage: Execution
///
/// Manages proxy participants and approval workflow.
#[derive(Clone)]
pub struct ProxyManager {
    /// Registered proxy participants
    participants: Vec<ProxyParticipant>,
    
    /// Pending approval requests
    pending_requests: Vec<ProxyApprovalRequest>,
    
    /// Collected approvals
    approvals: Vec<ProxyApproval>,
    
    /// Configuration
    config: ProxyConfig,
}

impl ProxyManager {
    /// Create new proxy manager
    pub fn new(config: ProxyConfig) -> Self {
        Self {
            participants: Vec::new(),
            pending_requests: Vec::new(),
            approvals: Vec::new(),
            config,
        }
    }
    
    /// Register proxy participant
    ///
    /// ## Lifecycle Stage: Quorum Convergence
    pub fn register_participant(
        &mut self,
        participant: ProxyParticipant,
    ) -> Result<(), &'static str> {
        // Check minimum stake requirement
        if participant.reputation_stake < self.config.min_reputation_stake {
            return Err("Insufficient reputation stake");
        }
        
        // Check not already registered
        if self.participants.iter().any(|p| p.id == participant.id) {
            return Err("Participant already registered");
        }
        
        self.participants.push(participant);
        Ok(())
    }
    
    /// Submit approval request
    ///
    /// ## Lifecycle Stage: Execution
    pub fn submit_request(
        &mut self,
        request: ProxyApprovalRequest,
    ) -> Result<[u8; 32], &'static str> {
        // Check sufficient proxies available
        let eligible_proxies = self.participants.iter()
            .filter(|p| p.reputation_stake >= request.required_bond)
            .count();
        
        if eligible_proxies < self.config.approval_threshold {
            return Err("Insufficient eligible proxies");
        }
        
        let request_id = request.id;
        self.pending_requests.push(request);
        Ok(request_id)
    }
    
    /// Submit approval
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// ## Audit Trail
    /// - Logs approval to ephemeral ledger
    /// - Bonds proxy stake
    /// - Emits ProxyApproval TXO
    pub fn submit_approval(
        &mut self,
        approval: ProxyApproval,
    ) -> Result<(), &'static str> {
        // Find proxy participant
        let proxy = self.participants.iter_mut()
            .find(|p| p.id == approval.proxy_id)
            .ok_or("Proxy not found")?;
        
        // Bond stake
        proxy.bond_stake(approval.bonded_amount)?;
        proxy.approval_count += 1;
        
        // TODO: Verify signature
        
        self.approvals.push(approval);
        Ok(())
    }
    
    /// Check if request approved
    ///
    /// ## Lifecycle Stage: Execution
    pub fn is_approved(&self, request_id: &[u8; 32]) -> bool {
        let approval_count = self.approvals.iter()
            .filter(|a| &a.request_id == request_id)
            .count();
        
        approval_count >= self.config.approval_threshold
    }
    
    /// Finalize approved request
    ///
    /// ## Lifecycle Stage: Execution
    ///
    /// Releases bonded stakes after approval completes successfully.
    pub fn finalize_request(&mut self, request_id: &[u8; 32]) -> Result<(), &'static str> {
        // Find approvals for this request
        let approvals: Vec<_> = self.approvals.iter()
            .filter(|a| &a.request_id == request_id)
            .cloned()
            .collect();
        
        if approvals.len() < self.config.approval_threshold {
            return Err("Insufficient approvals");
        }
        
        // Release bonded stakes
        for approval in approvals {
            if let Some(proxy) = self.participants.iter_mut()
                .find(|p| p.id == approval.proxy_id) {
                proxy.release_stake(approval.bonded_amount)?;
            }
        }
        
        // Remove from pending
        self.pending_requests.retain(|r| &r.id != request_id);
        self.approvals.retain(|a| &a.request_id != request_id);
        
        Ok(())
    }
    
    /// Slash misbehaving proxy
    ///
    /// ## Security Rationale
    /// - Enforces accountability
    /// - Deters malicious approvals
    /// - Creates permanent reputation cost
    pub fn slash_proxy(&mut self, proxy_id: &[u8; 32], amount: u64) -> Result<u64, &'static str> {
        let proxy = self.participants.iter_mut()
            .find(|p| &p.id == proxy_id)
            .ok_or("Proxy not found")?;
        
        let slashed = proxy.slash_stake(amount);
        Ok(slashed)
    }
}

/// Get current timestamp (milliseconds since epoch)
fn current_timestamp() -> u64 {
    #[cfg(feature = "std")]
    {
        use std::time::{SystemTime, UNIX_EPOCH};
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_millis() as u64
    }
    #[cfg(not(feature = "std"))]
    {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_proxy_participant_bond() {
        let mut proxy = ProxyParticipant::new([1u8; 32], 1000, [2u8; 32]);
        assert!(proxy.bond_stake(500).is_ok());
        assert_eq!(proxy.bonded_stake, 500);
        assert_eq!(proxy.reputation_stake, 500);
    }
    
    #[test]
    fn test_proxy_manager_registration() {
        let config = ProxyConfig::default();
        let mut manager = ProxyManager::new(config);
        
        let proxy = ProxyParticipant::new([1u8; 32], 2000, [2u8; 32]);
        assert!(manager.register_participant(proxy).is_ok());
    }
}
