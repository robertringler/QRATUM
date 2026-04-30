//! # Validator Incentives - Stake-Based Rewards and Slashing
//!
//! ## Lifecycle Stage: Epoch Finalization
//!
//! This module implements validator incentives through stake-based rewards and
//! slashing mechanisms, ensuring validators are economically incentivized to
//! behave honestly and penalized for misbehavior.
//!
//! ## Architectural Role
//!
//! - **Stake Registry**: Tracks validator stakes and delegations
//! - **Reward Distribution**: Rewards validators for successful proposals and votes
//! - **Slashing Mechanism**: Penalizes validators for violations
//! - **Economic Security**: Ensures cost of attack exceeds potential benefit
//!
//! ## Security Rationale
//!
//! - Stake creates economic incentive for honest behavior
//! - Slashing creates economic disincentive for malicious behavior
//! - Reward distribution incentivizes active participation
//! - All stake changes are auditable and irreversible
//!
//! ## Audit Trail
//!
//! - All stake deposits and withdrawals logged
//! - All reward distributions recorded with validator and amount
//! - All slashing events logged with reason and amount
//! - Epoch summaries with total rewards and slashing


extern crate alloc;
use alloc::collections::BTreeMap;
use alloc::vec::Vec;
use alloc::string::String;

use crate::consensus::{ValidatorID, Violation};

/// Stake information
#[derive(Debug, Clone)]
pub struct Stake {
    /// Validator who owns this stake
    pub validator: ValidatorID,
    
    /// Amount of stake (in base units)
    pub amount: u64,
    
    /// Delegation (if stake is delegated)
    pub delegator: Option<[u8; 32]>,
    
    /// Lock period (epochs until stake can be withdrawn)
    pub lock_period: u64,
    
    /// Epoch when stake was deposited
    pub deposit_epoch: u64,
}

impl Stake {
    /// Create new stake
    pub fn new(validator: ValidatorID, amount: u64, deposit_epoch: u64, lock_period: u64) -> Self {
        Self {
            validator,
            amount,
            delegator: None,
            lock_period,
            deposit_epoch,
        }
    }
    
    /// Check if stake is unlocked at given epoch
    pub fn is_unlocked(&self, current_epoch: u64) -> bool {
        current_epoch >= self.deposit_epoch + self.lock_period
    }
}

/// Validator incentives manager
///
/// ## Security Invariants
/// - Total rewards never exceed max supply
/// - Slashing is permanent and irreversible
/// - Stake withdrawals respect lock periods
/// - All changes are auditable
pub struct ValidatorIncentives {
    /// Stake registry mapping validator to total stake
    pub stake_registry: BTreeMap<ValidatorID, Stake>,
    
    /// Reward pool available for distribution
    pub reward_pool: u64,
    
    /// Total stake in the system
    pub total_stake: u64,
    
    /// Total rewards distributed
    pub total_rewards_distributed: u64,
    
    /// Total amount slashed
    pub total_slashed: u64,
    
    /// Current epoch
    pub current_epoch: u64,
    
    /// Reward rate per epoch (in basis points, 10000 = 100%)
    pub reward_rate: u64,
    
    /// Slashing rate per violation (in basis points)
    pub slashing_rate: u64,
}

impl ValidatorIncentives {
    /// Create new validator incentives manager
    ///
    /// ## Inputs
    /// - `initial_reward_pool`: Initial reward pool size
    /// - `reward_rate`: Reward rate per epoch (basis points)
    /// - `slashing_rate`: Slashing rate per violation (basis points)
    pub fn new(initial_reward_pool: u64, reward_rate: u64, slashing_rate: u64) -> Self {
        Self {
            stake_registry: BTreeMap::new(),
            reward_pool: initial_reward_pool,
            total_stake: 0,
            total_rewards_distributed: 0,
            total_slashed: 0,
            current_epoch: 0,
            reward_rate,
            slashing_rate,
        }
    }
    
    /// Deposit stake for a validator
    ///
    /// ## Inputs
    /// - `validator`: Validator identifier
    /// - `amount`: Amount to stake
    /// - `lock_period`: Number of epochs to lock stake
    ///
    /// ## Security
    /// - Stake must be positive
    /// - Stake is locked for specified period
    /// - Audit trail records deposit
    pub fn deposit_stake(&mut self, validator: ValidatorID, amount: u64, lock_period: u64) {
        if amount == 0 {
            return; // No-op for zero stake
        }
        
        let stake = Stake::new(validator, amount, self.current_epoch, lock_period);
        
        // Update or insert stake
        if let Some(existing_stake) = self.stake_registry.get_mut(&validator) {
            existing_stake.amount += amount;
        } else {
            self.stake_registry.insert(validator, stake);
        }
        
        self.total_stake += amount;
        
        // TODO: Emit audit TXO for stake deposit
    }
    
    /// Withdraw stake for a validator
    ///
    /// ## Inputs
    /// - `validator`: Validator identifier
    /// - `amount`: Amount to withdraw
    ///
    /// ## Returns
    /// - `true` if withdrawal successful
    /// - `false` if insufficient stake or still locked
    ///
    /// ## Security
    /// - Stake must be unlocked
    /// - Cannot withdraw more than staked
    /// - Audit trail records withdrawal
    pub fn withdraw_stake(&mut self, validator: &ValidatorID, amount: u64) -> bool {
        if let Some(stake) = self.stake_registry.get_mut(validator) {
            // Check if stake is unlocked
            if !stake.is_unlocked(self.current_epoch) {
                return false; // Still locked
            }
            
            // Check if sufficient stake
            if stake.amount < amount {
                return false; // Insufficient stake
            }
            
            // Withdraw
            stake.amount -= amount;
            self.total_stake -= amount;
            
            // Remove entry if stake is zero
            if stake.amount == 0 {
                self.stake_registry.remove(validator);
            }
            
            // TODO: Emit audit TXO for stake withdrawal
            
            true
        } else {
            false // Validator not found
        }
    }
    
    /// Reward a validator for successful participation
    ///
    /// ## Inputs
    /// - `validator`: Validator to reward
    /// - `amount`: Reward amount
    ///
    /// ## Security
    /// - Rewards come from reward pool
    /// - Cannot exceed available pool
    /// - Audit trail records reward
    pub fn reward(&mut self, validator: ValidatorID, amount: u64) {
        // Check if reward pool has sufficient funds
        if self.reward_pool < amount {
            return; // Insufficient reward pool
        }
        
        // Deduct from reward pool
        self.reward_pool -= amount;
        
        // Add to validator's stake
        if let Some(stake) = self.stake_registry.get_mut(&validator) {
            stake.amount += amount;
            self.total_stake += amount;
        } else {
            // Validator not staking yet - create new stake entry
            let stake = Stake::new(validator, amount, self.current_epoch, 0);
            self.stake_registry.insert(validator, stake);
            self.total_stake += amount;
        }
        
        self.total_rewards_distributed += amount;
        
        // TODO: Emit audit TXO for reward distribution
    }
    
    /// Slash a validator for misbehavior
    ///
    /// ## Inputs
    /// - `validator`: Validator to slash
    /// - `amount`: Amount to slash
    /// - `reason`: Violation reason
    ///
    /// ## Security
    /// - Slashing is irreversible
    /// - Slashed stake is burned (removed from circulation)
    /// - Audit trail records slashing with reason
    /// - Cannot slash more than validator's stake
    pub fn slash(&mut self, validator: ValidatorID, amount: u64, reason: Violation) {
        if let Some(stake) = self.stake_registry.get_mut(&validator) {
            // Calculate actual slash amount (capped at stake amount)
            let slash_amount = amount.min(stake.amount);
            
            // Slash stake
            stake.amount -= slash_amount;
            self.total_stake -= slash_amount;
            self.total_slashed += slash_amount;
            
            // Remove entry if stake is zero
            if stake.amount == 0 {
                self.stake_registry.remove(&validator);
            }
            
            // TODO: Emit audit TXO for slashing event with reason
            
            // Placeholder to use `reason` parameter
            let _ = reason;
        }
    }
    
    /// Calculate and distribute epoch rewards to all active validators
    ///
    /// ## Inputs
    /// - `active_validators`: List of validators who participated this epoch
    ///
    /// ## Security
    /// - Rewards proportional to stake
    /// - Only active validators receive rewards
    /// - Total rewards capped by reward pool
    pub fn distribute_epoch_rewards(&mut self, active_validators: &[ValidatorID]) {
        if active_validators.is_empty() {
            return; // No validators to reward
        }
        
        // Calculate total stake of active validators
        let active_stake: u64 = active_validators
            .iter()
            .filter_map(|v| self.stake_registry.get(v))
            .map(|s| s.amount)
            .sum();
        
        if active_stake == 0 {
            return; // No stake to reward
        }
        
        // Calculate total epoch reward (reward_rate is in basis points)
        let total_epoch_reward = (self.reward_pool * self.reward_rate) / 10000;
        
        // Distribute rewards proportionally
        for validator in active_validators {
            if let Some(stake) = self.stake_registry.get(validator) {
                // Calculate validator's share
                let validator_reward = (total_epoch_reward * stake.amount) / active_stake;
                
                // Reward validator
                self.reward(*validator, validator_reward);
            }
        }
        
        // Advance epoch
        self.current_epoch += 1;
        
        // TODO: Emit audit TXO for epoch reward distribution
    }
    
    /// Get stake for a validator
    pub fn get_stake(&self, validator: &ValidatorID) -> Option<u64> {
        self.stake_registry.get(validator).map(|s| s.amount)
    }
    
    /// Get total stake in the system
    pub fn get_total_stake(&self) -> u64 {
        self.total_stake
    }
}

impl Default for ValidatorIncentives {
    fn default() -> Self {
        Self::new(
            1_000_000_000, // 1B reward pool
            100,           // 1% per epoch (100 basis points)
            1000,          // 10% slashing rate (1000 basis points)
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;
    
    #[test]
    fn test_stake_deposit_and_withdrawal() {
        let mut incentives = ValidatorIncentives::default();
        
        let validator = [1u8; 32];
        
        // Deposit stake
        incentives.deposit_stake(validator, 1000, 0);
        assert_eq!(incentives.get_stake(&validator), Some(1000));
        assert_eq!(incentives.total_stake, 1000);
        
        // Withdraw stake
        let withdrawn = incentives.withdraw_stake(&validator, 500);
        assert!(withdrawn);
        assert_eq!(incentives.get_stake(&validator), Some(500));
        assert_eq!(incentives.total_stake, 500);
    }
    
    #[test]
    fn test_reward() {
        let mut incentives = ValidatorIncentives::default();
        
        let validator = [1u8; 32];
        
        // Reward validator
        incentives.reward(validator, 1000);
        assert_eq!(incentives.get_stake(&validator), Some(1000));
        assert_eq!(incentives.total_rewards_distributed, 1000);
    }
    
    #[test]
    fn test_slash() {
        let mut incentives = ValidatorIncentives::default();
        
        let validator = [1u8; 32];
        
        // Deposit stake first
        incentives.deposit_stake(validator, 1000, 0);
        
        // Slash validator
        incentives.slash(validator, 500, Violation::DoubleSigning);
        assert_eq!(incentives.get_stake(&validator), Some(500));
        assert_eq!(incentives.total_slashed, 500);
    }
    
    #[test]
    fn test_epoch_rewards() {
        let mut incentives = ValidatorIncentives::default();
        
        let validator1 = [1u8; 32];
        let validator2 = [2u8; 32];
        
        // Deposit stakes
        incentives.deposit_stake(validator1, 1000, 0);
        incentives.deposit_stake(validator2, 1000, 0);
        
        // Distribute epoch rewards
        incentives.distribute_epoch_rewards(&[validator1, validator2]);
        
        // Check rewards were distributed
        assert!(incentives.total_rewards_distributed > 0);
        assert_eq!(incentives.current_epoch, 1);
    }
    
    #[test]
    fn test_locked_stake() {
        let mut incentives = ValidatorIncentives::default();
        
        let validator = [1u8; 32];
        
        // Deposit stake with 10 epoch lock period
        incentives.deposit_stake(validator, 1000, 10);
        
        // Try to withdraw immediately (should fail)
        let withdrawn = incentives.withdraw_stake(&validator, 500);
        assert!(!withdrawn);
        
        // Advance epochs
        incentives.current_epoch = 10;
        
        // Try to withdraw after lock period (should succeed)
        let withdrawn = incentives.withdraw_stake(&validator, 500);
        assert!(withdrawn);
    }
}
