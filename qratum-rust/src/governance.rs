//! # Governance Module - Rust Protocol Layer
//!
//! ## Lifecycle Stage: Governance Execution
//!
//! This module provides Rust-side governance structures that integrate with
//! the Python governance protocol for decentralized protocol changes.
//!
//! ## Architectural Role
//!
//! - **Proposal Types**: Define governance proposal structures
//! - **Voting Mechanisms**: Track votes and calculate outcomes
//! - **Execution Hooks**: Execute approved governance proposals
//! - **Stake-Weighted Voting**: Weight votes by validator stake
//!
//! ## Security Rationale
//!
//! - All proposals require threshold approval
//! - Votes are weighted by stake (prevents Sybil attacks)
//! - Time-locked execution prevents rushed changes
//! - Veto mechanism for emergency stops
//!
//! ## Audit Trail
//!
//! - All proposals logged with proposer and details
//! - All votes recorded with voter and weight
//! - Execution events logged with timestamp
//! - Veto events recorded with authority


extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use alloc::string::String;

/// Proposal identifier
pub type ProposalID = [u8; 32];

/// Voter identifier (typically validator ID)
pub type VoterID = [u8; 32];

/// Authority identifier (for veto power)
pub type AuthorityID = [u8; 32];

/// Governance proposal type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProposalType {
    /// Change protocol parameter
    ParameterChange,
    /// Protocol upgrade
    ProtocolUpgrade,
    /// Validator set change
    ValidatorSetChange,
    /// Treasury spending
    TreasurySpending,
    /// Emergency action
    Emergency,
}

/// Vote decision
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VoteDecision {
    /// Vote in favor
    Approve,
    /// Vote against
    Reject,
    /// Abstain from voting
    Abstain,
}

/// Governance vote
#[derive(Debug, Clone)]
pub struct GovernanceVote {
    /// Voter identifier
    pub voter: VoterID,
    
    /// Vote decision
    pub decision: VoteDecision,
    
    /// Voting weight (typically stake amount)
    pub weight: u64,
    
    /// Vote signature
    pub signature: [u8; 64],
    
    /// Epoch when vote was cast
    pub epoch: u64,
}

/// Governance proposal
#[derive(Debug, Clone)]
pub struct GovernanceProposal {
    /// Unique proposal identifier
    pub id: ProposalID,
    
    /// Proposal type
    pub proposal_type: ProposalType,
    
    /// Proposer identifier
    pub proposer: VoterID,
    
    /// Proposal description
    pub description: String,
    
    /// Proposal payload (implementation-specific)
    pub payload: Vec<u8>,
    
    /// Required approval threshold (percentage, 0-100)
    pub threshold: u8,
    
    /// Voting period (number of epochs)
    pub voting_period: u64,
    
    /// Timelock period after approval (epochs before execution)
    pub timelock: u64,
    
    /// Epoch when proposal was created
    pub creation_epoch: u64,
}

impl GovernanceProposal {
    /// Check if voting period is active
    pub fn is_voting_active(&self, current_epoch: u64) -> bool {
        current_epoch < self.creation_epoch + self.voting_period
    }
    
    /// Check if proposal is approved (based on vote tallies)
    pub fn is_approved(&self, approve_weight: u64, total_weight: u64) -> bool {
        if total_weight == 0 {
            return false;
        }
        
        let approval_percentage = (approve_weight * 100) / total_weight;
        approval_percentage >= self.threshold as u64
    }
    
    /// Check if proposal can be executed
    pub fn can_execute(&self, current_epoch: u64, is_approved: bool) -> bool {
        // Must be approved
        if !is_approved {
            return false;
        }
        
        // Voting period must be over
        if current_epoch < self.creation_epoch + self.voting_period {
            return false;
        }
        
        // Timelock must have elapsed
        if current_epoch < self.creation_epoch + self.voting_period + self.timelock {
            return false;
        }
        
        true
    }
}

/// Governance state
///
/// ## Security Invariants
/// - Only approved proposals can be executed
/// - Proposals respect voting period and timelock
/// - Vetoed proposals cannot be executed
/// - All votes are stake-weighted
pub struct GovernanceState {
    /// Active proposals
    pub proposals: BTreeMap<ProposalID, GovernanceProposal>,
    
    /// Votes for each proposal
    pub votes: BTreeMap<ProposalID, Vec<GovernanceVote>>,
    
    /// Executed proposals
    pub executed: Vec<ProposalID>,
    
    /// Vetoed proposals
    pub vetoed: Vec<ProposalID>,
    
    /// Current epoch
    pub current_epoch: u64,
    
    /// Total voting weight (typically total stake)
    pub total_voting_weight: u64,
}

impl GovernanceState {
    /// Create new governance state
    pub fn new() -> Self {
        Self {
            proposals: BTreeMap::new(),
            votes: BTreeMap::new(),
            executed: Vec::new(),
            vetoed: Vec::new(),
            current_epoch: 0,
            total_voting_weight: 0,
        }
    }
    
    /// Submit a new governance proposal
    pub fn submit_proposal(&mut self, proposal: GovernanceProposal) {
        let id = proposal.id;
        self.proposals.insert(id, proposal);
        self.votes.insert(id, Vec::new());
        
        // TODO: Emit audit TXO for proposal submission
    }
    
    /// Cast a vote on a proposal
    pub fn vote(&mut self, proposal_id: ProposalID, vote: GovernanceVote) -> bool {
        // Check if proposal exists
        let proposal = match self.proposals.get(&proposal_id) {
            Some(p) => p,
            None => return false,
        };
        
        // Check if voting is still active
        if !proposal.is_voting_active(self.current_epoch) {
            return false;
        }
        
        // Check for duplicate votes
        if let Some(votes) = self.votes.get(&proposal_id) {
            if votes.iter().any(|v| v.voter == vote.voter) {
                return false; // Already voted
            }
        }
        
        // Add vote
        self.votes.get_mut(&proposal_id).unwrap().push(vote);
        
        // TODO: Emit audit TXO for vote
        
        true
    }
    
    /// Calculate vote tally for a proposal
    pub fn tally_votes(&self, proposal_id: &ProposalID) -> (u64, u64, u64) {
        let votes = match self.votes.get(proposal_id) {
            Some(v) => v,
            None => return (0, 0, 0),
        };
        
        let mut approve = 0u64;
        let mut reject = 0u64;
        let mut abstain = 0u64;
        
        for vote in votes {
            match vote.decision {
                VoteDecision::Approve => approve += vote.weight,
                VoteDecision::Reject => reject += vote.weight,
                VoteDecision::Abstain => abstain += vote.weight,
            }
        }
        
        (approve, reject, abstain)
    }
    
    /// Execute an approved proposal
    pub fn execute_proposal(&mut self, proposal_id: ProposalID) -> bool {
        // Check if proposal exists
        let proposal = match self.proposals.get(&proposal_id) {
            Some(p) => p.clone(),
            None => return false,
        };
        
        // Check if already executed
        if self.executed.contains(&proposal_id) {
            return false;
        }
        
        // Check if vetoed
        if self.vetoed.contains(&proposal_id) {
            return false;
        }
        
        // Calculate approval
        let (approve, _, _) = self.tally_votes(&proposal_id);
        let is_approved = proposal.is_approved(approve, self.total_voting_weight);
        
        // Check if can execute
        if !proposal.can_execute(self.current_epoch, is_approved) {
            return false;
        }
        
        // Execute proposal (implementation-specific)
        // TODO: Dispatch to appropriate handler based on proposal_type
        
        // Mark as executed
        self.executed.push(proposal_id);
        
        // TODO: Emit audit TXO for execution
        
        true
    }
    
    /// Veto a proposal
    pub fn veto(&mut self, proposal_id: ProposalID, _authority: AuthorityID) -> bool {
        // Check if proposal exists
        if !self.proposals.contains_key(&proposal_id) {
            return false;
        }
        
        // Check if already executed
        if self.executed.contains(&proposal_id) {
            return false; // Cannot veto executed proposals
        }
        
        // Check if already vetoed
        if self.vetoed.contains(&proposal_id) {
            return false;
        }
        
        // TODO: Verify authority has veto power
        
        // Mark as vetoed
        self.vetoed.push(proposal_id);
        
        // TODO: Emit audit TXO for veto
        
        true
    }
    
    /// Advance to next epoch
    pub fn advance_epoch(&mut self) {
        self.current_epoch += 1;
    }
    
    /// Get proposal by ID
    pub fn get_proposal(&self, id: &ProposalID) -> Option<&GovernanceProposal> {
        self.proposals.get(id)
    }
}

impl Default for GovernanceState {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    
    #[test]
    fn test_governance_proposal() {
        let proposal = GovernanceProposal {
            id: [1u8; 32],
            proposal_type: ProposalType::ParameterChange,
            proposer: [2u8; 32],
            description: "Test proposal".into(),
            payload: vec![],
            threshold: 67, // 67% approval required
            voting_period: 10,
            timelock: 5,
            creation_epoch: 0,
        };
        
        assert!(proposal.is_voting_active(5));
        assert!(!proposal.is_voting_active(11));
    }
    
    #[test]
    fn test_governance_state() {
        let mut state = GovernanceState::new();
        state.total_voting_weight = 1000;
        
        let proposal = GovernanceProposal {
            id: [1u8; 32],
            proposal_type: ProposalType::ParameterChange,
            proposer: [2u8; 32],
            description: "Test proposal".into(),
            payload: vec![],
            threshold: 67,
            voting_period: 10,
            timelock: 5,
            creation_epoch: 0,
        };
        
        state.submit_proposal(proposal);
        
        // Vote
        let vote = GovernanceVote {
            voter: [3u8; 32],
            decision: VoteDecision::Approve,
            weight: 700,
            signature: [0u8; 64],
            epoch: 0,
        };
        
        let voted = state.vote([1u8; 32], vote);
        assert!(voted);
        
        // Check tally
        let (approve, reject, abstain) = state.tally_votes(&[1u8; 32]);
        assert_eq!(approve, 700);
        assert_eq!(reject, 0);
        assert_eq!(abstain, 0);
    }
}
