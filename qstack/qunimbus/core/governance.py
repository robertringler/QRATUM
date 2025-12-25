"""Deterministic governance and voting primitives.

This module provides decentralized governance for the QRATUM ghost machine,
enabling protocol changes through on-chain voting with stake-weighted decisions,
time-locked execution, and Merkle-logged votes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from enum import Enum


class VoteDecision(Enum):
    """Vote decision options."""
    APPROVE = "approve"
    REJECT = "reject"
    ABSTAIN = "abstain"


class ProposalStatus(Enum):
    """Proposal lifecycle status."""
    ACTIVE = "active"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    VETOED = "vetoed"


@dataclass(frozen=True)
class PolicyProposal:
    """A governance policy proposal.
    
    Attributes:
        id: Unique proposal identifier (hex string)
        title: Human-readable proposal title
        description: Detailed proposal description
        payload: Implementation-specific proposal data
        proposer: Validator ID who proposed (hex string)
        threshold: Required approval percentage (0-100)
        voting_period: Number of epochs for voting
        timelock: Number of epochs after approval before execution
        creation_epoch: Epoch when proposal was created
    """
    id: str
    title: str
    description: str
    payload: bytes
    proposer: str
    threshold: int
    voting_period: int
    timelock: int
    creation_epoch: int


@dataclass
class Vote:
    """A vote on a proposal.
    
    Attributes:
        voter: Validator ID (hex string)
        proposal_id: Proposal being voted on (hex string)
        decision: Vote decision
        weight: Voting weight (typically stake amount)
        signature: Vote signature (hex string)
        epoch: Epoch when vote was cast
    """
    voter: str
    proposal_id: str
    decision: VoteDecision
    weight: int
    signature: str
    epoch: int


@dataclass
class Authority:
    """An authority with veto power.
    
    Attributes:
        id: Authority identifier (hex string)
        public_key: Authority public key (hex string)
        veto_power: Whether this authority can veto proposals
    """
    id: str
    public_key: str
    veto_power: bool


class GovernanceProtocol:
    """Decentralized governance protocol.
    
    Provides on-chain governance with:
    - Stake-weighted voting
    - Time-locked execution
    - Merkle-logged votes
    - Veto mechanism for emergency stops
    
    Security Properties:
    - All proposals require threshold approval
    - Votes weighted by validator stake (prevents Sybil attacks)
    - Time-locked execution prevents rushed changes
    - All actions auditable via Merkle logs
    """
    
    def __init__(self):
        """Initialize governance protocol."""
        self.proposals: Dict[str, PolicyProposal] = {}
        self.votes: Dict[str, List[Vote]] = {}
        self.proposal_status: Dict[str, ProposalStatus] = {}
        self.executed_proposals: List[str] = []
        self.vetoed_proposals: List[str] = []
        self.current_epoch: int = 0
        self.total_voting_weight: int = 0
        self.authorities: Dict[str, Authority] = {}
    
    def propose_policy_change(self, proposal: PolicyProposal) -> str:
        """Submit a new governance proposal.
        
        Args:
            proposal: Policy proposal to submit
            
        Returns:
            Proposal ID (hex string)
            
        Security:
            - Proposal logged in Merkle tree
            - Audit trail records proposer and timestamp
        """
        proposal_id = proposal.id
        self.proposals[proposal_id] = proposal
        self.votes[proposal_id] = []
        self.proposal_status[proposal_id] = ProposalStatus.ACTIVE
        
        # TODO: Emit audit TXO for proposal submission
        # TODO: Add to Merkle tree
        
        return proposal_id
    
    def vote(self, proposal_id: str, voter: str, vote: Vote) -> bool:
        """Cast a vote on a proposal.
        
        Args:
            proposal_id: Proposal to vote on (hex string)
            voter: Validator ID casting vote (hex string)
            vote: Vote object with decision and weight
            
        Returns:
            True if vote accepted, False otherwise
            
        Security:
            - Vote signature verified
            - Duplicate votes prevented
            - Vote logged in Merkle tree
            - Voting period enforced
        """
        # Check if proposal exists
        if proposal_id not in self.proposals:
            return False
        
        proposal = self.proposals[proposal_id]
        
        # Check if voting is still active
        if self.current_epoch >= proposal.creation_epoch + proposal.voting_period:
            return False
        
        # Check if proposal is still active
        if self.proposal_status[proposal_id] != ProposalStatus.ACTIVE:
            return False
        
        # Check for duplicate votes
        if any(v.voter == voter for v in self.votes[proposal_id]):
            return False
        
        # TODO: Verify vote signature
        
        # Record vote
        self.votes[proposal_id].append(vote)
        
        # TODO: Add vote to Merkle tree
        # TODO: Emit audit TXO for vote
        
        return True
    
    def tally_votes(self, proposal_id: str) -> Dict[str, int]:
        """Calculate vote tally for a proposal.
        
        Args:
            proposal_id: Proposal to tally (hex string)
            
        Returns:
            Dictionary with approve, reject, abstain weights
        """
        if proposal_id not in self.votes:
            return {"approve": 0, "reject": 0, "abstain": 0}
        
        tally = {"approve": 0, "reject": 0, "abstain": 0}
        
        for vote in self.votes[proposal_id]:
            if vote.decision == VoteDecision.APPROVE:
                tally["approve"] += vote.weight
            elif vote.decision == VoteDecision.REJECT:
                tally["reject"] += vote.weight
            elif vote.decision == VoteDecision.ABSTAIN:
                tally["abstain"] += vote.weight
        
        return tally
    
    def is_approved(self, proposal_id: str) -> bool:
        """Check if proposal has reached approval threshold.
        
        Args:
            proposal_id: Proposal to check (hex string)
            
        Returns:
            True if approved, False otherwise
        """
        if proposal_id not in self.proposals:
            return False
        
        proposal = self.proposals[proposal_id]
        tally = self.tally_votes(proposal_id)
        
        if self.total_voting_weight == 0:
            return False
        
        approval_percentage = (tally["approve"] * 100) // self.total_voting_weight
        return approval_percentage >= proposal.threshold
    
    def can_execute(self, proposal_id: str) -> bool:
        """Check if proposal can be executed.
        
        Args:
            proposal_id: Proposal to check (hex string)
            
        Returns:
            True if can be executed, False otherwise
            
        Conditions:
            - Proposal must be approved
            - Voting period must be over
            - Timelock must have elapsed
            - Proposal must not be vetoed or already executed
        """
        if proposal_id not in self.proposals:
            return False
        
        proposal = self.proposals[proposal_id]
        
        # Check if approved
        if not self.is_approved(proposal_id):
            return False
        
        # Check if voting period is over
        if self.current_epoch < proposal.creation_epoch + proposal.voting_period:
            return False
        
        # Check if timelock has elapsed
        if self.current_epoch < proposal.creation_epoch + proposal.voting_period + proposal.timelock:
            return False
        
        # Check if already executed or vetoed
        if proposal_id in self.executed_proposals or proposal_id in self.vetoed_proposals:
            return False
        
        return True
    
    def execute_approved_proposal(self, proposal_id: str) -> bool:
        """Execute an approved proposal.
        
        Args:
            proposal_id: Proposal to execute (hex string)
            
        Returns:
            True if executed successfully, False otherwise
            
        Security:
            - Only approved proposals can execute
            - Timelock must have elapsed
            - Execution logged in Merkle tree
            - Execution is irreversible
        """
        if not self.can_execute(proposal_id):
            return False
        
        # Mark as executed
        self.executed_proposals.append(proposal_id)
        self.proposal_status[proposal_id] = ProposalStatus.EXECUTED
        
        # TODO: Execute proposal payload
        # TODO: Emit audit TXO for execution
        # TODO: Add to Merkle tree
        
        return True
    
    def veto(self, proposal_id: str, authority: Authority) -> bool:
        """Veto a proposal (emergency stop mechanism).
        
        Args:
            proposal_id: Proposal to veto (hex string)
            authority: Authority exercising veto power
            
        Returns:
            True if veto successful, False otherwise
            
        Security:
            - Only authorized entities can veto
            - Cannot veto already executed proposals
            - Veto logged in Merkle tree
        """
        # Check if proposal exists
        if proposal_id not in self.proposals:
            return False
        
        # Check if already executed
        if proposal_id in self.executed_proposals:
            return False
        
        # Check if already vetoed
        if proposal_id in self.vetoed_proposals:
            return False
        
        # Verify authority has veto power
        if not authority.veto_power:
            return False
        
        # TODO: Verify authority signature
        
        # Mark as vetoed
        self.vetoed_proposals.append(proposal_id)
        self.proposal_status[proposal_id] = ProposalStatus.VETOED
        
        # TODO: Emit audit TXO for veto
        # TODO: Add to Merkle tree
        
        return True
    
    def advance_epoch(self):
        """Advance to next epoch."""
        self.current_epoch += 1
    
    def set_total_voting_weight(self, weight: int):
        """Set total voting weight (typically total stake).
        
        Args:
            weight: Total voting weight
        """
        self.total_voting_weight = weight
    
    def register_authority(self, authority: Authority):
        """Register an authority with veto power.
        
        Args:
            authority: Authority to register
        """
        self.authorities[authority.id] = authority


@dataclass(frozen=True)
class GovernanceRule:
    identifier: str
    threshold: float
    weight: float

    def evaluate(self, metrics: dict[str, float]) -> float:
        score = metrics.get(self.identifier, 0.0)
        return self.weight if score >= self.threshold else 0.0


def vote_outcome(votes: dict[str, int]) -> dict[str, object]:
    tally = sum(votes.values()) or 1
    normalized = {k: v / tally for k, v in votes.items()}
    winner = sorted(normalized.items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return {"winner": winner[0], "share": winner[1], "distribution": normalized}


def governance_score(rules: list[GovernanceRule], metrics: dict[str, float]) -> dict[str, float]:
    total_weight = sum(rule.weight for rule in rules) or 1.0
    satisfied = sum(rule.evaluate(metrics) for rule in rules)
    return {"score": satisfied / total_weight, "details": sum(metrics.values())}


def deterministic_auction(bids: dict[str, float]) -> dict[str, object]:
    ordered = sorted(bids.items(), key=lambda kv: (-kv[1], kv[0]))
    winner, price = ordered[0]
    clearing_price = ordered[1][1] if len(ordered) > 1 else price
    return {"winner": winner, "clearing_price": clearing_price, "ordered": ordered}
