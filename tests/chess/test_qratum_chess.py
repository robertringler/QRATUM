"""Tests for QRATUM-Chess core functionality."""

from __future__ import annotations

import pytest


class TestBitBoard:
    """Tests for bitboard representation."""
    
    def test_starting_position(self):
        """Test starting position bitboard creation."""
        from qratum_chess.core import BitBoard, Color, PieceType
        
        board = BitBoard.starting_position()
        
        # Check white pieces
        assert board.piece_at(0) == (Color.WHITE, PieceType.ROOK)  # a1
        assert board.piece_at(4) == (Color.WHITE, PieceType.KING)  # e1
        assert board.piece_at(8) == (Color.WHITE, PieceType.PAWN)  # a2
        
        # Check black pieces
        assert board.piece_at(63) == (Color.BLACK, PieceType.ROOK)  # h8
        assert board.piece_at(60) == (Color.BLACK, PieceType.KING)  # e8
        assert board.piece_at(48) == (Color.BLACK, PieceType.PAWN)  # a7
        
        # Check empty squares
        assert board.piece_at(32) is None  # a5
    
    def test_piece_placement(self):
        """Test piece placement and removal."""
        from qratum_chess.core import BitBoard, Color, PieceType
        
        board = BitBoard.empty()
        
        # Place a piece
        board.set_piece(28, Color.WHITE, PieceType.KNIGHT)  # e4
        assert board.piece_at(28) == (Color.WHITE, PieceType.KNIGHT)
        
        # Remove the piece
        board.remove_piece(28, Color.WHITE, PieceType.KNIGHT)
        assert board.piece_at(28) is None
    
    def test_attack_map(self):
        """Test attack map generation."""
        from qratum_chess.core import BitBoard, Color, PieceType
        
        board = BitBoard.starting_position()
        
        # White should have attacks from pawns
        white_attacks = board.get_attack_map(Color.WHITE)
        
        # Pawns on rank 2 attack rank 3
        assert white_attacks & (1 << 16)  # a3 attacked by a2 pawn
        assert white_attacks & (1 << 18)  # c3 attacked by b2/d2 pawns


class TestPosition:
    """Tests for position management."""
    
    def test_starting_position(self):
        """Test starting position creation."""
        from qratum_chess.core.position import Position
        from qratum_chess.core import Color, CastlingRights
        
        pos = Position.starting()
        
        assert pos.side_to_move == Color.WHITE
        assert pos.castling == CastlingRights.ALL
        assert pos.ep_square == -1
        assert pos.halfmove_clock == 0
        assert pos.fullmove_number == 1
    
    def test_fen_parsing(self):
        """Test FEN string parsing."""
        from qratum_chess.core.position import Position
        from qratum_chess.core import Color
        
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        pos = Position.from_fen(fen)
        
        assert pos.side_to_move == Color.BLACK
        assert pos.ep_square == 20  # e3
        assert pos.to_fen() == fen
    
    def test_legal_move_generation(self):
        """Test legal move generation."""
        from qratum_chess.core.position import Position
        
        pos = Position.starting()
        moves = pos.generate_legal_moves()
        
        # Starting position has 20 legal moves
        assert len(moves) == 20
    
    def test_make_move(self):
        """Test making a move."""
        from qratum_chess.core.position import Position, Move
        from qratum_chess.core import Color
        
        pos = Position.starting()
        
        # e2-e4
        move = Move(from_sq=12, to_sq=28)
        new_pos = pos.make_move(move)
        
        assert new_pos.side_to_move == Color.BLACK
        assert new_pos.ep_square == 20  # e3
        assert new_pos.board.piece_at(28) is not None  # e4 occupied
        assert new_pos.board.piece_at(12) is None  # e2 empty
    
    def test_check_detection(self):
        """Test check detection."""
        from qratum_chess.core.position import Position
        
        # Position with white king in check
        fen = "rnbqkbnr/ppp1pppp/8/3p4/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1"
        pos = Position.from_fen(fen)
        
        # White is not in check in this position
        assert not pos.is_in_check()
    
    def test_checkmate_detection(self):
        """Test checkmate detection."""
        from qratum_chess.core.position import Position
        
        # Fool's mate position
        fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
        pos = Position.from_fen(fen)
        
        assert pos.is_in_check()
        assert pos.is_checkmate()


class TestMove:
    """Tests for move handling."""
    
    def test_uci_conversion(self):
        """Test UCI notation conversion."""
        from qratum_chess.core.position import Move
        from qratum_chess.core import PieceType
        
        # Normal move
        move = Move(from_sq=12, to_sq=28)  # e2e4
        assert move.to_uci() == "e2e4"
        
        # Promotion
        move = Move(from_sq=52, to_sq=60, promotion=PieceType.QUEEN)  # e7e8q
        assert move.to_uci() == "e7e8q"
    
    def test_uci_parsing(self):
        """Test UCI notation parsing."""
        from qratum_chess.core.position import Move
        from qratum_chess.core import PieceType
        
        move = Move.from_uci("e2e4")
        assert move.from_sq == 12
        assert move.to_sq == 28
        
        move = Move.from_uci("e7e8q")
        assert move.promotion == PieceType.QUEEN


class TestSearch:
    """Tests for search algorithms."""
    
    def test_alphabeta_basic(self):
        """Test basic alpha-beta search."""
        from qratum_chess.core.position import Position
        from qratum_chess.search.alphabeta import AlphaBetaSearch
        
        pos = Position.starting()
        search = AlphaBetaSearch()
        
        best_move, value, stats = search.search(pos, depth=3)
        
        assert best_move is not None
        assert stats.nodes_searched > 0
        assert stats.depth_reached <= 3
    
    def test_mcts_basic(self):
        """Test basic MCTS search."""
        from qratum_chess.core.position import Position
        from qratum_chess.search.mcts import MCTSSearch
        
        pos = Position.starting()
        mcts = MCTSSearch()
        
        best_move, visit_counts, value = mcts.search(
            pos,
            num_simulations=100
        )
        
        assert best_move is not None
        assert len(visit_counts) > 0


class TestNeuralNetwork:
    """Tests for neural network components."""
    
    def test_position_encoding(self):
        """Test position encoding."""
        from qratum_chess.core.position import Position
        from qratum_chess.neural.encoding import PositionEncoder
        
        encoder = PositionEncoder()
        pos = Position.starting()
        
        tensor = encoder.encode(pos)
        
        assert tensor.shape == (28, 8, 8)
        assert tensor.dtype.name == 'float32'
    
    def test_network_forward(self):
        """Test network forward pass."""
        from qratum_chess.core.position import Position
        from qratum_chess.neural.network import NeuralEvaluator
        
        evaluator = NeuralEvaluator()
        pos = Position.starting()
        
        policy, value = evaluator.evaluate(pos)
        
        assert policy.shape[0] > 0
        assert -1 <= value <= 1


class TestTriModalCore:
    """Tests for Tri-Modal Cognitive Core."""
    
    def test_trimodal_evaluation(self):
        """Test tri-modal evaluation."""
        from qratum_chess.core.position import Position
        from qratum_chess.neural.trimodal import TriModalCore
        
        core = TriModalCore()
        pos = Position.starting()
        legal_moves = pos.generate_legal_moves()
        
        best_move, value, diagnostics = core.evaluate(pos, legal_moves)
        
        assert best_move is not None
        assert 'tactical_weight' in diagnostics
        assert 'strategic_weight' in diagnostics
        assert 'conceptual_weight' in diagnostics


class TestBenchmarks:
    """Tests for benchmark framework."""
    
    def test_performance_metrics(self):
        """Test performance metrics calculation."""
        from qratum_chess.benchmarks.metrics import PerformanceMetrics, PerformanceTargets
        
        metrics = PerformanceMetrics()
        
        # Just verify initialization
        assert metrics.targets.nodes_per_sec_single == 70_000_000
    
    def test_torture_suite_positions(self):
        """Test torture suite has valid positions."""
        from qratum_chess.benchmarks.torture import StrategicTortureSuite
        from qratum_chess.core.position import Position
        
        suite = StrategicTortureSuite()
        
        # Verify all positions are valid
        for name, fen, expected, category in suite.positions:
            pos = Position.from_fen(fen)
            assert pos is not None
    
    def test_benchmark_config_record_motifs(self):
        """Test BenchmarkConfig has record_motifs field."""
        from qratum_chess.benchmarks.runner import BenchmarkConfig
        
        config = BenchmarkConfig(record_motifs=True)
        assert config.record_motifs is True
        
        config = BenchmarkConfig(record_motifs=False)
        assert config.record_motifs is False
    
    def test_benchmark_runner_single_game(self):
        """Test BenchmarkRunner has run_single_game method."""
        from qratum_chess.benchmarks.runner import BenchmarkRunner, BenchmarkConfig
        
        config = BenchmarkConfig(
            run_performance=False,
            run_torture=False,
            run_elo=False,
            run_resilience=False,
        )
        runner = BenchmarkRunner(config)
        
        assert hasattr(runner, 'run_single_game')
        assert hasattr(runner, 'log_game_summary')
        assert hasattr(runner, 'compile_full_summary')


class TestSelfModifyingEngine:
    """Tests for SelfModifyingEngine."""
    
    def test_engine_initialization(self):
        """Test SelfModifyingEngine initialization with all parameters."""
        from qratum_chess.self_modifying import SelfModifyingEngine
        
        engine = SelfModifyingEngine(
            tactical_weight=0.4,
            strategic_weight=0.4,
            conceptual_weight=0.2,
            novelty_pressure=0.5,
            memory_decay=0.01,
            ontology_evolution=True,
            recursive_depth_limit=10
        )
        
        assert engine.engine_config.tactical_weight == 0.4
        assert engine.engine_config.strategic_weight == 0.4
        assert engine.engine_config.conceptual_weight == 0.2
        assert engine.engine_config.novelty_pressure == 0.5
        assert engine.engine_config.memory_decay == 0.01
        assert engine.engine_config.ontology_evolution is True
        assert engine.engine_config.recursive_depth_limit == 10
    
    def test_engine_checkpoint_save_load(self):
        """Test checkpoint save and load functionality."""
        import os
        import tempfile
        from qratum_chess.self_modifying import SelfModifyingEngine
        
        engine = SelfModifyingEngine(
            tactical_weight=0.5,
            strategic_weight=0.3,
            conceptual_weight=0.2,
        )
        
        # Modify some state
        engine.games_played = 10
        engine.win_count = 5
        engine.elo_history = [2500.0, 2520.0, 2540.0]
        
        # Save checkpoint
        with tempfile.NamedTemporaryFile(suffix='.ckpt', delete=False) as f:
            checkpoint_path = f.name
        
        try:
            engine.save_checkpoint(checkpoint_path)
            assert os.path.exists(checkpoint_path)
            
            # Create new engine and load checkpoint
            new_engine = SelfModifyingEngine()
            new_engine.load_checkpoint(checkpoint_path)
            
            assert new_engine.games_played == 10
            assert new_engine.win_count == 5
            assert len(new_engine.elo_history) == 3
        finally:
            if os.path.exists(checkpoint_path):
                os.remove(checkpoint_path)
    
    def test_engine_record_game_result(self):
        """Test game result recording and ELO updates."""
        from qratum_chess.self_modifying import SelfModifyingEngine
        
        engine = SelfModifyingEngine()
        
        # Record a win
        engine.record_game_result("win", opponent_elo=3500.0)
        assert engine.games_played == 1
        assert engine.win_count == 1
        assert len(engine.elo_history) == 1
        
        # Record a loss
        engine.record_game_result("loss", opponent_elo=3500.0)
        assert engine.games_played == 2
        assert engine.loss_count == 1
        
        # Record a draw
        engine.record_game_result("draw", opponent_elo=3500.0)
        assert engine.games_played == 3
        assert engine.draw_count == 1
    
    def test_engine_record_motif(self):
        """Test motif recording functionality."""
        from qratum_chess.self_modifying import SelfModifyingEngine
        
        engine = SelfModifyingEngine()
        
        engine.record_motif(
            position_fen="rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
            move_sequence=["e2e4"],
            motif_type="tactical",
            novelty_score=0.8
        )
        
        assert len(engine.discovered_motifs) == 1
        assert engine.discovered_motifs[0]["motif_type"] == "tactical"
        assert engine.discovered_motifs[0]["novelty_score"] == 0.8
    
    def test_engine_get_summary(self):
        """Test engine summary generation."""
        from qratum_chess.self_modifying import SelfModifyingEngine
        
        engine = SelfModifyingEngine(
            tactical_weight=0.4,
            strategic_weight=0.4,
            conceptual_weight=0.2,
        )
        
        summary = engine.get_engine_summary()
        
        assert "config" in summary
        assert "current_state" in summary
        assert "statistics" in summary
        assert "meta_dynamics" in summary
        assert "motifs" in summary


class TestAgents:
    """Tests for agent orchestration."""
    
    def test_orchestrator_new_game(self):
        """Test starting a new game via orchestrator."""
        from qratum_chess.agents.orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator()
        result = orchestrator.new_game()
        
        assert result.get("success") is True
    
    def test_orchestrator_make_move(self):
        """Test making a move via orchestrator."""
        from qratum_chess.agents.orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator()
        orchestrator.new_game()
        
        result = orchestrator.make_move("e2e4")
        
        assert result.get("success") is True


class TestUCI:
    """Tests for UCI protocol."""
    
    def test_uci_position_parsing(self):
        """Test UCI position command parsing."""
        from io import StringIO
        from qratum_chess.protocols.uci import UCIEngine
        
        # Create engine with mock streams
        input_stream = StringIO("position startpos moves e2e4\nquit\n")
        output_stream = StringIO()
        
        engine = UCIEngine(input_stream=input_stream, output_stream=output_stream)
        
        # Process position command
        engine._cmd_position(["startpos", "moves", "e2e4"])
        
        # Verify position was updated
        assert engine.position is not None
        assert engine.position.board.piece_at(28) is not None  # e4 occupied
