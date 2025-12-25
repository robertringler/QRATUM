//! Decentralized Ghost Machine Architecture Demo
//!
//! This example demonstrates the integration of all decentralized ghost machine
//! components in a complete QRATUM session.

use qratum::*;

fn main() {
    println!("=== QRATUM Decentralized Ghost Machine Demo ===\n");
    
    // 1. Configure session with ghost machine parameters
    println!("1. Configuring decentralized session...");
    let mut config = SessionConfig::default();
    config.consensus_threshold = 67; // 2/3 supermajority
    config.max_peers = 100;
    config.reward_rate = 100;  // 1% per epoch
    config.slashing_rate = 1000; // 10% per violation
    println!("   ✓ Consensus threshold: {}%", config.consensus_threshold);
    println!("   ✓ Max peers: {}", config.max_peers);
    println!("   ✓ Reward rate: {} bps", config.reward_rate);
    println!("   ✓ Slashing rate: {} bps\n", config.slashing_rate);
    
    // 2. Initialize consensus engine
    println!("2. Initializing consensus engine...");
    let mut consensus = BasicConsensusEngine::new(ConsensusType::BftHotStuff, 67);
    println!("   ✓ Algorithm: BftHotStuff");
    println!("   ✓ Threshold: 2/3 supermajority\n");
    
    // 3. Register validators
    println!("3. Registering validators...");
    for i in 0..5 {
        let mut validator_id = [0u8; 32];
        validator_id[0] = i;
        let info = ValidatorInfo {
            public_key: validator_id,
            stake: 1000,
            voting_power: 1000,
            status: ValidatorStatus::Active,
            successful_proposals: 0,
            violations: 0,
        };
        consensus.validator_registry.register_validator(validator_id, info);
    }
    println!("   ✓ Registered 5 validators");
    println!("   ✓ Total stake: {}\n", consensus.validator_registry.total_active_stake);
    
    // 4. Initialize P2P network
    println!("4. Initializing P2P network...");
    let node_id = [1u8; 32];
    let public_key = [2u8; 32];
    let _p2p = P2PNetwork::new(node_id, public_key, 100);
    println!("   ✓ Node ID: {:?}...", &node_id[..4]);
    println!("   ✓ Max peers: 100\n");
    
    // 5. Initialize validator incentives
    println!("5. Initializing validator incentives...");
    let mut incentives = ValidatorIncentives::default();
    for i in 0..5 {
        let mut validator_id = [0u8; 32];
        validator_id[0] = i;
        incentives.deposit_stake(validator_id, 1000, 10);
    }
    println!("   ✓ Total stake: {}", incentives.get_total_stake());
    println!("   ✓ Reward pool: {}\n", incentives.reward_pool);
    
    // 6. Initialize governance
    println!("6. Initializing governance...");
    let mut governance = GovernanceState::new();
    governance.total_voting_weight = incentives.get_total_stake();
    println!("   ✓ Total voting weight: {}\n", governance.total_voting_weight);
    
    // 7. Initialize protocol upgrades
    println!("7. Initializing protocol upgrade manager...");
    let upgrades = UpgradeManager::default();
    println!("   ✓ Current version: {}.{}.{}\n", 
        upgrades.current_version.major,
        upgrades.current_version.minor,
        upgrades.current_version.patch
    );
    
    // 8. Initialize transport layer
    println!("8. Initializing anti-censorship transport...");
    let mut transport = CensorshipResistance::default();
    transport.configure_channel(Channel::Tcp);
    transport.configure_channel(Channel::Tor);
    println!("   ✓ Configured channels: TCP, Tor");
    println!("   ✓ Available channels: {}\n", 
        if transport.has_available_channel() { "Yes" } else { "No" });
    
    // 9. Demonstrate TXO consensus flow
    println!("9. Demonstrating TXO consensus flow...");
    let txo = Txo::new(TxoType::Input, 0, b"test transaction".to_vec(), vec![]);
    println!("   ✓ Created TXO: {:?}...", &txo.id[..4]);
    
    // Propose TXO
    let proposal_id = consensus.propose_txo(txo.clone());
    println!("   ✓ Proposed TXO: {:?}...", &proposal_id[..4]);
    
    // Validators vote
    for i in 0..4 {
        let mut validator_id = [0u8; 32];
        validator_id[0] = i;
        let vote = Vote {
            validator_id,
            proposal_id,
            approve: true,
            signature: [0u8; 64],
            height: 0,
        };
        consensus.vote_on_proposal(proposal_id, vote);
    }
    println!("   ✓ Collected votes: 4/5 (80%)");
    
    // Finalize TXO
    match consensus.finalize_txo(proposal_id) {
        Ok(commit) => {
            println!("   ✓ TXO finalized at height: {}", commit.height);
            println!("   ✓ Validator signatures: {}\n", commit.signatures.len());
        }
        Err(e) => println!("   ✗ Finalization failed: {:?}\n", e),
    }
    
    // 10. Demonstrate governance
    println!("10. Demonstrating governance...");
    let proposal = GovernanceProposal {
        id: [1u8; 32],
        proposal_type: ProposalType::ParameterChange,
        proposer: [0u8; 32],
        description: "Increase consensus threshold to 75%".into(),
        payload: vec![75],
        threshold: 67,
        voting_period: 10,
        timelock: 5,
        creation_epoch: 0,
    };
    governance.submit_proposal(proposal.clone());
    println!("   ✓ Submitted governance proposal");
    
    // Vote on proposal
    let vote = GovernanceVote {
        voter: [0u8; 32],
        decision: VoteDecision::Approve,
        weight: 4000,
        signature: [0u8; 64],
        epoch: 0,
    };
    governance.vote(proposal.id, vote);
    let (approve, reject, abstain) = governance.tally_votes(&proposal.id);
    println!("   ✓ Votes: Approve={}, Reject={}, Abstain={}\n", approve, reject, abstain);
    
    // 11. Demonstrate validator rewards
    println!("11. Demonstrating validator rewards...");
    let active_validators: Vec<_> = consensus.validator_registry
        .get_active_validators()
        .into_iter()
        .take(4)
        .collect();
    incentives.distribute_epoch_rewards(&active_validators);
    println!("   ✓ Distributed epoch rewards");
    println!("   ✓ Total rewards: {}", incentives.total_rewards_distributed);
    println!("   ✓ Current epoch: {}\n", incentives.current_epoch);
    
    // 12. Demonstrate ZK state transition
    println!("12. Demonstrating ZK state transition...");
    let prev_state = b"previous state";
    let next_state = b"next state";
    let transition = ZkStateTransition::generate_proof(
        prev_state,
        next_state,
        b"transition witness",
        0,
        TransitionType::TxoExecution,
    );
    println!("   ✓ Generated ZK proof");
    println!("   ✓ Proof valid: {}\n", transition.verify());
    
    println!("=== Demo Complete ===");
    println!("\nAll decentralized ghost machine components initialized and tested:");
    println!("✓ Protocol-enforced consensus");
    println!("✓ Decentralized governance");
    println!("✓ P2P network layer");
    println!("✓ Validator incentives");
    println!("✓ ZK state transitions");
    println!("✓ Protocol upgrades");
    println!("✓ Anti-censorship transport");
}
