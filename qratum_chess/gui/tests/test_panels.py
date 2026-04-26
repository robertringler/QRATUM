"""Tests for QRATUM-Chess GUI system."""

import unittest

from qratum_chess.gui.panels.anti_holographic import AntiHolographicPanel
from qratum_chess.gui.panels.board import BoardPanel, BoardTheme, RenderMode
from qratum_chess.gui.panels.control import ControlPanel, GameMode
from qratum_chess.gui.panels.motif_tracker import MotifTracker, MotifType
from qratum_chess.gui.panels.quantum import QuantumPanel
from qratum_chess.gui.panels.search_tree import SearchTreePanel, TreeLayout
from qratum_chess.gui.panels.telemetry import TelemetryPanel
from qratum_chess.gui.panels.tricortex import TriCortexPanel


class TestBoardPanel(unittest.TestCase):
    """Tests for BoardPanel."""

    def test_initialization(self):
        """Test board panel initialization."""
        panel = BoardPanel(width=800, height=800)
        self.assertEqual(panel.width, 800)
        self.assertEqual(panel.height, 800)
        self.assertIsNotNone(panel.state)

    def test_set_position(self):
        """Test setting position from FEN."""
        panel = BoardPanel()
        fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
        panel.set_position(fen)
        self.assertEqual(panel.state.position_fen, fen)

    def test_toggle_render_mode(self):
        """Test toggling between 2D and 3D."""
        panel = BoardPanel()
        self.assertEqual(panel.state.render_mode, RenderMode.MODE_2D)
        panel.toggle_render_mode()
        self.assertEqual(panel.state.render_mode, RenderMode.MODE_3D)
        panel.toggle_render_mode()
        self.assertEqual(panel.state.render_mode, RenderMode.MODE_2D)

    def test_set_theme(self):
        """Test setting board theme."""
        panel = BoardPanel()
        panel.set_theme(BoardTheme.QUANTUM)
        self.assertEqual(panel.state.theme, BoardTheme.QUANTUM)

    def test_highlight_move(self):
        """Test move highlighting."""
        panel = BoardPanel()
        panel.highlight_move(12, 28)  # e2-e4
        self.assertEqual(len(panel.state.highlights), 2)

    def test_render_data(self):
        """Test getting render data."""
        panel = BoardPanel()
        data = panel.get_render_data()
        self.assertIn("squares", data)
        self.assertIn("evaluation", data)
        self.assertEqual(len(data["squares"]), 64)


class TestTriCortexPanel(unittest.TestCase):
    """Tests for TriCortexPanel."""

    def test_initialization(self):
        """Test tri-cortex panel initialization."""
        panel = TriCortexPanel()
        self.assertIsNotNone(panel.state.tactical_heatmap)
        self.assertIsNotNone(panel.state.strategic_heatmap)
        self.assertIsNotNone(panel.state.conceptual_heatmap)

    def test_update_from_trimodal(self):
        """Test updating from TriModalCore diagnostics."""
        panel = TriCortexPanel()
        diagnostics = {
            "tactical_weight": 0.4,
            "strategic_weight": 0.35,
            "conceptual_weight": 0.25,
            "entropy": 0.5,
            "confidence": 0.9,
        }
        panel.update_from_trimodal(diagnostics)
        self.assertEqual(panel.state.contributions.tactical, 0.4)
        self.assertEqual(panel.state.contributions.strategic, 0.35)

    def test_render_data(self):
        """Test getting render data."""
        panel = TriCortexPanel()
        data = panel.get_render_data()
        self.assertIn("heatmaps", data)
        self.assertIn("contributions", data)
        self.assertIn("entropy", data)


class TestSearchTreePanel(unittest.TestCase):
    """Tests for SearchTreePanel."""

    def test_initialization(self):
        """Test search tree panel initialization."""
        panel = SearchTreePanel()
        self.assertEqual(len(panel.state.nodes), 0)

    def test_set_layout(self):
        """Test setting layout mode."""
        panel = SearchTreePanel()
        panel.set_layout(TreeLayout.RADIAL)
        self.assertEqual(panel.state.layout, TreeLayout.RADIAL)

    def test_zoom_pan(self):
        """Test zoom and pan controls."""
        panel = SearchTreePanel()
        initial_zoom = panel.state.zoom
        panel.zoom_in(1.5)
        self.assertGreater(panel.state.zoom, initial_zoom)
        panel.pan(10, 20)
        self.assertEqual(panel.state.pan_x, 10 / panel.state.zoom)


class TestMotifTracker(unittest.TestCase):
    """Tests for MotifTracker."""

    def test_initialization(self):
        """Test motif tracker initialization."""
        tracker = MotifTracker()
        self.assertEqual(len(tracker.state.active_motifs), 0)

    def test_motif_categories(self):
        """Test motif category assignment."""
        tracker = MotifTracker()
        self.assertEqual(tracker.get_motif_category(MotifType.FORK), "tactical")
        self.assertEqual(tracker.get_motif_category(MotifType.PASSED_PAWN), "strategic")
        self.assertEqual(tracker.get_motif_category(MotifType.KINGSIDE_ATTACK), "conceptual")


class TestQuantumPanel(unittest.TestCase):
    """Tests for QuantumPanel."""

    def test_initialization(self):
        """Test quantum panel initialization."""
        panel = QuantumPanel()
        self.assertFalse(panel.state.enabled)

    def test_enable_disable(self):
        """Test enabling and disabling."""
        panel = QuantumPanel()
        panel.set_enabled(True)
        self.assertTrue(panel.state.enabled)
        panel.set_enabled(False)
        self.assertFalse(panel.state.enabled)

    def test_update_circuit_state(self):
        """Test updating circuit state."""
        panel = QuantumPanel()
        panel.set_enabled(True)
        amplitudes = [complex(0.5, 0.5), complex(0.5, -0.5)]
        panel.update_circuit_state(2, amplitudes, [(0, 1)])
        self.assertIsNotNone(panel.state.circuit_snapshot)


class TestAntiHolographicPanel(unittest.TestCase):
    """Tests for AntiHolographicPanel."""

    def test_initialization(self):
        """Test anti-holographic panel initialization."""
        panel = AntiHolographicPanel()
        self.assertEqual(panel.state.stochasticity.current_stochasticity, 0.0)

    def test_update_stochasticity(self):
        """Test updating stochasticity."""
        panel = AntiHolographicPanel()
        probs = [0.5, 0.3, 0.15, 0.05]
        panel.update_stochasticity(probs, 1, 4)  # Selected 2nd best
        self.assertGreater(panel.state.stochasticity.current_stochasticity, 0)

    def test_compute_overall_score(self):
        """Test computing overall AH score."""
        panel = AntiHolographicPanel()
        score = panel.compute_overall_score()
        self.assertGreaterEqual(score, 0)
        self.assertLessEqual(score, 1)


class TestTelemetryPanel(unittest.TestCase):
    """Tests for TelemetryPanel."""

    def test_initialization(self):
        """Test telemetry panel initialization."""
        panel = TelemetryPanel()
        self.assertGreater(panel.state.session_start_time, 0)

    def test_update_performance(self):
        """Test updating performance metrics."""
        panel = TelemetryPanel()
        panel.update_performance(1000000, 500000, 1.0)
        self.assertEqual(panel.state.performance.nodes_per_second, 1000000)
        self.assertEqual(panel.state.performance.evaluations_per_second, 500000)

    def test_update_cache(self):
        """Test updating cache metrics."""
        panel = TelemetryPanel()
        panel.update_cache(93, 100, 256, 500000, 1000000)
        self.assertEqual(panel.state.cache.hash_hit_rate, 0.93)


class TestControlPanel(unittest.TestCase):
    """Tests for ControlPanel."""

    def test_initialization(self):
        """Test control panel initialization."""
        panel = ControlPanel()
        self.assertFalse(panel.state.game_in_progress)

    def test_new_game(self):
        """Test starting new game."""
        panel = ControlPanel()
        result = panel.new_game()
        self.assertTrue(result)
        self.assertTrue(panel.state.game_in_progress)

    def test_set_game_mode(self):
        """Test setting game mode."""
        panel = ControlPanel()
        panel.set_game_mode(GameMode.AI_VS_AI)
        self.assertEqual(panel.state.game.mode, GameMode.AI_VS_AI)

    def test_toggle_quantum(self):
        """Test toggling quantum panel."""
        panel = ControlPanel()
        panel.toggle_quantum(True)
        self.assertTrue(panel.state.quantum.enabled)
        panel.toggle_quantum()
        self.assertFalse(panel.state.quantum.enabled)


class TestGUIIntegration(unittest.TestCase):
    """Integration tests for GUI system."""

    def test_full_gui_flow(self):
        """Test complete GUI workflow."""
        # Create panels
        board = BoardPanel()
        tricortex = TriCortexPanel()
        telemetry = TelemetryPanel()
        control = ControlPanel()

        # Simulate game start
        control.new_game()
        board.set_position("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")

        # Update diagnostics
        tricortex.update_from_trimodal(
            {
                "tactical_weight": 0.4,
                "strategic_weight": 0.35,
                "conceptual_weight": 0.25,
            }
        )

        # Update telemetry
        telemetry.update_performance(1000000, 500000, 1.0)

        # Get render data
        board_data = board.get_render_data()
        cortex_data = tricortex.get_render_data()

        self.assertEqual(len(board_data["squares"]), 64)
        self.assertIn("contributions", cortex_data)


if __name__ == "__main__":
    unittest.main()
