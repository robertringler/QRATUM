"""Tests for Aethernet Consensus Module."""

from Aethernet.core.consensus import (
    BFTConsensus,
    BlockHeader,
    ConsensusPhase,
    ConsensusVote,
    QuorumState,
    TrajectoryAwareConsensus,
    VoteType,
)


class TestBlockHeader:
    """Tests for BlockHeader."""

    def test_header_creation(self):
        """Test creating a block header."""
        header = BlockHeader(
            height=1,
            round=0,
            timestamp="2025-01-01T00:00:00Z",
            proposer_id="val_001",
            parent_hash="parent_hash",
            state_root="state_root",
            txs_root="txs_root",
            consensus_hash="consensus_hash",
        )
        assert header.height == 1
        assert header.round == 0

    def test_header_hash(self):
        """Test header hash is deterministic."""
        header1 = BlockHeader(
            height=1,
            round=0,
            timestamp="2025-01-01T00:00:00Z",
            proposer_id="val_001",
            parent_hash="p",
            state_root="s",
            txs_root="t",
            consensus_hash="c",
        )
        header2 = BlockHeader(
            height=1,
            round=0,
            timestamp="2025-01-01T00:00:00Z",
            proposer_id="val_001",
            parent_hash="p",
            state_root="s",
            txs_root="t",
            consensus_hash="c",
        )
        assert header1.compute_hash() == header2.compute_hash()

    def test_header_serialization(self):
        """Test header serialization."""
        header = BlockHeader(
            height=1,
            round=0,
            timestamp="2025-01-01T00:00:00Z",
            proposer_id="val_001",
            parent_hash="p",
            state_root="s",
            txs_root="t",
            consensus_hash="c",
        )
        serialized = header.serialize()
        assert "block_hash" in serialized
        assert serialized["height"] == 1


class TestConsensusVote:
    """Tests for ConsensusVote."""

    def test_vote_creation(self):
        """Test creating a vote."""
        vote = ConsensusVote(
            vote_type=VoteType.PREVOTE,
            height=1,
            round=0,
            block_hash="block_hash",
            validator_id="val_001",
            timestamp="2025-01-01T00:00:00Z",
            signature="sig",
        )
        assert vote.vote_type == VoteType.PREVOTE
        assert vote.validator_id == "val_001"

    def test_vote_hash(self):
        """Test vote hash computation."""
        vote = ConsensusVote(
            vote_type=VoteType.PREVOTE,
            height=1,
            round=0,
            block_hash="bh",
            validator_id="v1",
            timestamp="2025-01-01T00:00:00Z",
            signature="s",
        )
        hash1 = vote.compute_hash()
        assert len(hash1) == 64  # SHA256 hex


class TestQuorumState:
    """Tests for QuorumState."""

    def test_quorum_creation(self):
        """Test creating quorum state."""
        quorum = QuorumState(
            height=1,
            round=0,
            block_hash="bh",
            total_voting_power=100,
        )
        assert quorum.prevote_power == 0
        assert quorum.precommit_power == 0

    def test_add_prevote(self):
        """Test adding prevote."""
        quorum = QuorumState(height=1, round=0, block_hash="bh", total_voting_power=100)
        vote = ConsensusVote(
            VoteType.PREVOTE,
            1,
            0,
            "bh",
            "v1",
            "ts",
            "sig",
        )
        result = quorum.add_prevote(vote, 30)
        assert result is True
        assert quorum.prevote_power == 30

    def test_add_duplicate_prevote(self):
        """Test duplicate prevote is rejected."""
        quorum = QuorumState(height=1, round=0, block_hash="bh", total_voting_power=100)
        vote = ConsensusVote(VoteType.PREVOTE, 1, 0, "bh", "v1", "ts", "sig")
        quorum.add_prevote(vote, 30)
        result = quorum.add_prevote(vote, 30)
        assert result is False
        assert quorum.prevote_power == 30

    def test_prevote_quorum(self):
        """Test prevote quorum detection."""
        quorum = QuorumState(height=1, round=0, block_hash="bh", total_voting_power=100)
        # Need >2/3 = >66.67
        quorum.add_prevote(ConsensusVote(VoteType.PREVOTE, 1, 0, "bh", "v1", "ts", "s"), 30)
        assert quorum.has_prevote_quorum() is False
        quorum.add_prevote(ConsensusVote(VoteType.PREVOTE, 1, 0, "bh", "v2", "ts", "s"), 37)
        assert quorum.has_prevote_quorum() is True  # 67 > 66.67

    def test_precommit_quorum(self):
        """Test precommit quorum detection."""
        quorum = QuorumState(height=1, round=0, block_hash="bh", total_voting_power=100)
        quorum.add_precommit(ConsensusVote(VoteType.PRECOMMIT, 1, 0, "bh", "v1", "ts", "s"), 67)
        assert quorum.has_precommit_quorum() is True


class TestBFTConsensus:
    """Tests for BFTConsensus."""

    def setup_method(self):
        """Set up test fixtures."""
        self.voting_power = {"v1": 40, "v2": 30, "v3": 30}

        def get_voting_power(vid):
            return self.voting_power.get(vid, 0)

        def get_total():
            return sum(self.voting_power.values())

        def get_proposer(slot):
            # Simple round-robin
            validators = list(self.voting_power.keys())
            return validators[slot % len(validators)]

        self.consensus = BFTConsensus(
            get_voting_power=get_voting_power,
            get_total_voting_power=get_total,
            get_proposer_for_slot=get_proposer,
        )

    def test_consensus_creation(self):
        """Test creating consensus engine."""
        assert self.consensus.current_height == 1
        assert self.consensus.current_round == 0

    def test_start_round(self):
        """Test starting a round."""
        round_state = self.consensus.start_round(1, 0)
        assert round_state.height == 1
        assert round_state.round == 0
        assert round_state.phase == ConsensusPhase.PROPOSE

    def test_propose_block(self):
        """Test proposing a block."""
        self.consensus.start_round(1, 0)
        # Get expected proposer
        proposer = self.consensus.get_proposer_for_slot(100)  # 1*100 + 0
        block = self.consensus.propose_block(
            height=1,
            round=0,
            proposer_id=proposer,
            parent_hash="parent",
            state_root="state",
            txs_root="txs",
        )
        assert block is not None
        assert block.height == 1

    def test_propose_wrong_proposer(self):
        """Test proposal from wrong proposer fails."""
        self.consensus.start_round(1, 0)
        block = self.consensus.propose_block(
            height=1,
            round=0,
            proposer_id="wrong_validator",
            parent_hash="p",
            state_root="s",
            txs_root="t",
        )
        assert block is None

    def test_prevote(self):
        """Test submitting a prevote."""
        self.consensus.start_round(1, 0)
        proposer = self.consensus.get_proposer_for_slot(100)
        block = self.consensus.propose_block(
            1,
            0,
            proposer,
            "p",
            "s",
            "t",
        )
        result = self.consensus.prevote(
            1,
            0,
            block.compute_hash(),
            "v1",
            "sig",
        )
        assert result is True

    def test_prevote_advances_phase(self):
        """Test prevote quorum advances to precommit."""
        self.consensus.start_round(1, 0)
        proposer = self.consensus.get_proposer_for_slot(100)
        block = self.consensus.propose_block(1, 0, proposer, "p", "s", "t")
        block_hash = block.compute_hash()

        # Submit enough prevotes for quorum
        self.consensus.prevote(1, 0, block_hash, "v1", "s1")
        self.consensus.prevote(1, 0, block_hash, "v2", "s2")
        self.consensus.prevote(1, 0, block_hash, "v3", "s3")

        round_state = self.consensus.rounds[(1, 0)]
        assert round_state.phase == ConsensusPhase.PRECOMMIT

    def test_precommit(self):
        """Test submitting a precommit."""
        self.consensus.start_round(1, 0)
        proposer = self.consensus.get_proposer_for_slot(100)
        block = self.consensus.propose_block(1, 0, proposer, "p", "s", "t")
        block_hash = block.compute_hash()

        # Get to precommit phase
        self.consensus.prevote(1, 0, block_hash, "v1", "s1")
        self.consensus.prevote(1, 0, block_hash, "v2", "s2")
        self.consensus.prevote(1, 0, block_hash, "v3", "s3")

        result = self.consensus.precommit(1, 0, block_hash, "v1", "s1")
        assert result is True

    def test_block_finalization(self):
        """Test block finalization after precommit quorum."""
        self.consensus.start_round(1, 0)
        proposer = self.consensus.get_proposer_for_slot(100)
        block = self.consensus.propose_block(1, 0, proposer, "p", "s", "t")
        block_hash = block.compute_hash()

        # Prevotes
        self.consensus.prevote(1, 0, block_hash, "v1", "s1")
        self.consensus.prevote(1, 0, block_hash, "v2", "s2")
        self.consensus.prevote(1, 0, block_hash, "v3", "s3")

        # Precommits
        self.consensus.precommit(1, 0, block_hash, "v1", "s1")
        self.consensus.precommit(1, 0, block_hash, "v2", "s2")
        self.consensus.precommit(1, 0, block_hash, "v3", "s3")

        finalized = self.consensus.get_finalized_block(1)
        assert finalized is not None
        assert finalized.compute_hash() == block_hash

    def test_get_latest_finalized_height(self):
        """Test getting latest finalized height."""
        assert self.consensus.get_latest_finalized_height() == 0

        # Finalize a block
        self.consensus.start_round(1, 0)
        proposer = self.consensus.get_proposer_for_slot(100)
        block = self.consensus.propose_block(1, 0, proposer, "p", "s", "t")
        bh = block.compute_hash()
        for v in ["v1", "v2", "v3"]:
            self.consensus.prevote(1, 0, bh, v, "s")
        for v in ["v1", "v2", "v3"]:
            self.consensus.precommit(1, 0, bh, v, "s")

        assert self.consensus.get_latest_finalized_height() == 1

    def test_consensus_stats(self):
        """Test consensus statistics."""
        stats = self.consensus.get_stats()
        assert stats["current_height"] == 1
        assert stats["finalized_blocks"] == 0


class TestTrajectoryAwareConsensus:
    """Tests for TrajectoryAwareConsensus."""

    def setup_method(self):
        """Set up test fixtures."""
        self.voting_power = {"v1": 50, "v2": 50}

        def get_vp(vid):
            return self.voting_power.get(vid, 0)

        def get_total():
            return sum(self.voting_power.values())

        def get_proposer(slot):
            return list(self.voting_power.keys())[slot % 2]

        self.consensus = TrajectoryAwareConsensus(
            get_voting_power=get_vp,
            get_total_voting_power=get_total,
            get_proposer_for_slot=get_proposer,
            collapse_threshold=0.3,
        )

    def test_trajectory_consensus_creation(self):
        """Test creating trajectory-aware consensus."""
        assert self.consensus.health_score == 1.0
        assert self.consensus.is_suspended is False

    def test_update_trajectory_metrics(self):
        """Test updating trajectory metrics."""
        self.consensus.update_trajectory_metrics(
            system_health=0.8,
            collapse_probability=0.1,
            precursor_signals=[],
        )
        assert self.consensus.health_score == 0.8
        assert self.consensus.collapse_probability == 0.1

    def test_auto_suspension(self):
        """Test automatic suspension on high collapse probability."""
        self.consensus.update_trajectory_metrics(
            system_health=0.5,
            collapse_probability=0.5,  # Above 0.3 threshold
            precursor_signals=["entropy_spike"],
        )
        assert self.consensus.is_suspended is True
        assert self.consensus.suspension_reason is not None

    def test_proposal_blocked_when_suspended(self):
        """Test proposals blocked when suspended."""
        self.consensus.start_round(1, 0)
        # Suspend
        self.consensus.update_trajectory_metrics(0.4, 0.5, [])

        proposer = self.consensus.get_proposer_for_slot(100)
        block = self.consensus.propose_block(1, 0, proposer, "p", "s", "t")
        assert block is None

    def test_resume_consensus(self):
        """Test resuming consensus."""
        self.consensus.update_trajectory_metrics(0.4, 0.5, [])
        assert self.consensus.is_suspended is True

        # Lower collapse probability
        self.consensus.collapse_probability = 0.1
        result = self.consensus.resume_consensus()
        assert result is True
        assert self.consensus.is_suspended is False

    def test_cannot_resume_high_risk(self):
        """Test cannot resume with high collapse probability."""
        self.consensus.update_trajectory_metrics(0.4, 0.5, [])
        # Try to resume without lowering risk
        result = self.consensus.resume_consensus()
        assert result is False

    def test_health_status(self):
        """Test health status reporting."""
        self.consensus.update_trajectory_metrics(0.7, 0.2, ["warning"])
        status = self.consensus.get_health_status()
        assert status["health_score"] == 0.7
        assert status["collapse_probability"] == 0.2
        assert status["is_suspended"] is False
        assert "warning" in status["precursor_signals"]
