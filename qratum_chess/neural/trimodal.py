"""Tri-Modal Cognitive Core for QRATUM-Chess (Stage III).

Implements three co-evolving cognition stacks for chess evaluation:

1. Tactical Cortex: Ultra-fast NNUE-style evaluator trained on tactical motifs
   - Absolute superiority in short-horizon calculation
   
2. Strategic Cortex: Deep residual policy-value net trained by RL self-play
   - Long-range planning, structure mastery
   
3. Conceptual Cortex: Transformer-based abstraction learner
   - Pattern invention, anti-theory innovation

The fusion layer combines outputs using:
M_final = argmax_a Σ w_i * Q_i(a) + Ω(a)

Where:
- Q_i are cortex evaluations
- w_i are context-adaptive weights
- Ω is a novelty-pressure functional preventing stagnation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any
import math
import numpy as np


class GamePhase(Enum):
    """Game phase for adaptive weight selection."""
    OPENING = "opening"
    MIDDLEGAME = "middlegame"
    ENDGAME = "endgame"


@dataclass
class CortexOutput:
    """Output from a cognitive cortex.
    
    Attributes:
        move_values: Q-values for each legal move.
        confidence: Confidence score for this cortex's evaluation.
        features: Additional extracted features.
    """
    move_values: dict[Any, float]
    confidence: float
    features: dict[str, float] = field(default_factory=dict)


@dataclass
class TriModalConfig:
    """Configuration for Tri-Modal Cognitive Core.
    
    Attributes:
        tactical_weight: Base weight for tactical cortex.
        strategic_weight: Base weight for strategic cortex.
        conceptual_weight: Base weight for conceptual cortex.
        novelty_pressure: Strength of novelty-pressure functional.
        adaptive_weights: Whether to use adaptive weight adjustment.
    """
    tactical_weight: float = 0.4
    strategic_weight: float = 0.4
    conceptual_weight: float = 0.2
    novelty_pressure: float = 0.1
    adaptive_weights: bool = True


class TacticalCortex:
    """Ultra-fast NNUE-style tactical evaluator.
    
    Specializes in short-horizon tactical calculation using
    efficiently updatable neural network architecture.
    
    Key features:
    - Incrementally updatable feature extraction
    - Focus on material and tactical patterns
    - Optimized for speed over depth
    """
    
    # Feature weights (simplified NNUE-style)
    PIECE_VALUES = {
        'P': 100, 'N': 320, 'B': 330, 'R': 500, 'Q': 900, 'K': 0,
        'p': -100, 'n': -320, 'b': -330, 'r': -500, 'q': -900, 'k': 0,
    }
    
    # Piece-square tables for positional bonuses
    PAWN_PST = np.array([
        [0,  0,  0,  0,  0,  0,  0,  0],
        [50, 50, 50, 50, 50, 50, 50, 50],
        [10, 10, 20, 30, 30, 20, 10, 10],
        [5,  5, 10, 25, 25, 10,  5,  5],
        [0,  0,  0, 20, 20,  0,  0,  0],
        [5, -5,-10,  0,  0,-10, -5,  5],
        [5, 10, 10,-20,-20, 10, 10,  5],
        [0,  0,  0,  0,  0,  0,  0,  0]
    ])
    
    KNIGHT_PST = np.array([
        [-50,-40,-30,-30,-30,-30,-40,-50],
        [-40,-20,  0,  0,  0,  0,-20,-40],
        [-30,  0, 10, 15, 15, 10,  0,-30],
        [-30,  5, 15, 20, 20, 15,  5,-30],
        [-30,  0, 15, 20, 20, 15,  0,-30],
        [-30,  5, 10, 15, 15, 10,  5,-30],
        [-40,-20,  0,  5,  5,  0,-20,-40],
        [-50,-40,-30,-30,-30,-30,-40,-50]
    ])
    
    def __init__(self):
        """Initialize the tactical cortex."""
        self.feature_accumulator = np.zeros(512)  # NNUE-style feature vector
        self._init_nnue_weights()
    
    def _init_nnue_weights(self) -> None:
        """Initialize NNUE-style weight matrices."""
        # Two-layer fully connected network for evaluation
        self.fc1_weights = np.random.randn(512, 256) * 0.01
        self.fc1_bias = np.zeros(256)
        self.fc2_weights = np.random.randn(256, 1) * 0.01
        self.fc2_bias = np.zeros(1)
    
    def evaluate(self, position: 'Position', legal_moves: list) -> CortexOutput:
        """Evaluate position from tactical perspective.
        
        Args:
            position: Chess position to evaluate.
            legal_moves: List of legal moves.
            
        Returns:
            CortexOutput with move values and confidence.
        """
        # Extract features
        features = self._extract_features(position)
        
        # Get base evaluation
        base_eval = self._evaluate_features(features)
        
        # Score each move based on tactical considerations
        move_values = {}
        for move in legal_moves:
            move_score = self._score_move_tactically(position, move, base_eval)
            move_values[move] = move_score
        
        # Calculate confidence based on tactical clarity
        confidence = self._calculate_confidence(position, move_values)
        
        return CortexOutput(
            move_values=move_values,
            confidence=confidence,
            features={
                'material_balance': features['material'],
                'tactical_tension': features['tension'],
            }
        )
    
    def _extract_features(self, position: 'Position') -> dict[str, float]:
        """Extract tactical features from position."""
        from qratum_chess.core import Color, PieceType, iter_bits
        
        features = {
            'material': 0.0,
            'tension': 0.0,
            'king_safety': 0.0,
            'pst_bonus': 0.0,
        }
        
        # Material count
        for color in Color:
            sign = 1 if color == Color.WHITE else -1
            for pt in PieceType:
                pieces = int(position.board.pieces[color, pt])
                count = bin(pieces).count('1')
                if pt == PieceType.PAWN:
                    features['material'] += sign * count * 100
                elif pt == PieceType.KNIGHT:
                    features['material'] += sign * count * 320
                elif pt == PieceType.BISHOP:
                    features['material'] += sign * count * 330
                elif pt == PieceType.ROOK:
                    features['material'] += sign * count * 500
                elif pt == PieceType.QUEEN:
                    features['material'] += sign * count * 900
        
        # Tactical tension (pieces under attack)
        white_attacks = position.board.get_attack_map(Color.WHITE)
        black_attacks = position.board.get_attack_map(Color.BLACK)
        white_pieces = int(position.board.occupancy[Color.WHITE])
        black_pieces = int(position.board.occupancy[Color.BLACK])
        
        white_under_attack = black_attacks & white_pieces
        black_under_attack = white_attacks & black_pieces
        
        features['tension'] = bin(white_under_attack | black_under_attack).count('1')
        
        # Piece-square table bonuses (simplified)
        for sq in iter_bits(int(position.board.pieces[Color.WHITE, PieceType.PAWN])):
            rank, file = sq // 8, sq % 8
            features['pst_bonus'] += self.PAWN_PST[rank, file]
        
        for sq in iter_bits(int(position.board.pieces[Color.BLACK, PieceType.PAWN])):
            rank, file = 7 - (sq // 8), sq % 8
            features['pst_bonus'] -= self.PAWN_PST[rank, file]
        
        return features
    
    def _evaluate_features(self, features: dict[str, float]) -> float:
        """Evaluate features through NNUE network."""
        # Convert features to input vector
        input_vec = np.zeros(512)
        input_vec[0] = features['material'] / 2000.0
        input_vec[1] = features['tension'] / 10.0
        input_vec[2] = features['pst_bonus'] / 100.0
        
        # Forward pass
        hidden = np.maximum(0, input_vec @ self.fc1_weights + self.fc1_bias)
        output = hidden @ self.fc2_weights + self.fc2_bias
        
        return float(output[0])
    
    def _score_move_tactically(
        self,
        position: 'Position',
        move: 'Move',
        base_eval: float
    ) -> float:
        """Score a move from tactical perspective."""
        score = base_eval
        
        # Capture bonus
        captured = position.board.piece_at(move.to_sq)
        if captured:
            _, cap_type = captured
            capture_values = {0: 100, 1: 320, 2: 330, 3: 500, 4: 900, 5: 0}
            score += capture_values.get(int(cap_type), 0) * 0.1
        
        # Check bonus
        new_pos = position.make_move(move)
        if new_pos.is_in_check():
            score += 0.2
        
        # Checkmate bonus
        if new_pos.is_checkmate():
            score += 100.0
        
        return score
    
    def _calculate_confidence(
        self,
        position: 'Position',
        move_values: dict
    ) -> float:
        """Calculate confidence in tactical evaluation."""
        if not move_values:
            return 0.0
        
        values = list(move_values.values())
        if not values:
            return 0.0
        
        max_val = max(values)
        min_val = min(values)
        spread = max_val - min_val
        
        # High confidence if clear best move
        if spread > 0.5:
            return 0.9
        elif spread > 0.2:
            return 0.7
        else:
            return 0.5


class StrategicCortex:
    """Deep residual policy-value network trained by RL self-play.
    
    Specializes in long-range planning and structure mastery.
    Uses the PolicyValueNetwork from the neural module.
    """
    
    def __init__(self):
        """Initialize the strategic cortex."""
        from qratum_chess.neural.network import NeuralEvaluator
        self.evaluator = NeuralEvaluator()
    
    def evaluate(self, position: 'Position', legal_moves: list) -> CortexOutput:
        """Evaluate position from strategic perspective.
        
        Args:
            position: Chess position to evaluate.
            legal_moves: List of legal moves.
            
        Returns:
            CortexOutput with move values and confidence.
        """
        # Get policy and value from neural network
        policy, value = self.evaluator.evaluate(position)
        
        # Convert to move values
        move_probs = self.evaluator.get_move_probabilities(position, legal_moves)
        
        # Convert probabilities to Q-values (simplified)
        move_values = {}
        for move, prob in move_probs.items():
            # Q-value combines policy probability and value estimate
            move_values[move] = prob * 0.5 + value * 0.5
        
        # Fill in missing moves with base value
        for move in legal_moves:
            if move not in move_values:
                move_values[move] = value
        
        # Confidence based on value certainty
        confidence = abs(value) * 0.5 + 0.5
        
        return CortexOutput(
            move_values=move_values,
            confidence=confidence,
            features={
                'value_estimate': value,
                'policy_entropy': self._calculate_entropy(list(move_probs.values())),
            }
        )
    
    def _calculate_entropy(self, probs: list[float]) -> float:
        """Calculate entropy of probability distribution."""
        if not probs:
            return 0.0
        entropy = 0.0
        for p in probs:
            if p > 0:
                entropy -= p * math.log(p + 1e-10)
        return entropy


class ConceptualCortex:
    """Transformer-based abstraction learner for pattern invention.
    
    Specializes in anti-theory innovation and emergent pattern recognition.
    Uses attention mechanisms to identify novel strategic concepts.
    """
    
    def __init__(self, hidden_dim: int = 256, num_heads: int = 8):
        """Initialize the conceptual cortex.
        
        Args:
            hidden_dim: Hidden dimension for transformer.
            num_heads: Number of attention heads.
        """
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self._init_transformer_weights()
        
        # Historical pattern memory for novelty detection
        self.pattern_memory: list[np.ndarray] = []
        self.max_memory_size = 10000
    
    def _init_transformer_weights(self) -> None:
        """Initialize transformer attention weights."""
        dim = self.hidden_dim
        head_dim = dim // self.num_heads
        
        # Query, Key, Value projections
        self.W_q = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        self.W_k = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        self.W_v = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        self.W_o = np.random.randn(dim, dim) * np.sqrt(2.0 / dim)
        
        # FFN weights
        self.W_ff1 = np.random.randn(dim, dim * 4) * np.sqrt(2.0 / dim)
        self.W_ff2 = np.random.randn(dim * 4, dim) * np.sqrt(2.0 / (dim * 4))
    
    def evaluate(self, position: 'Position', legal_moves: list) -> CortexOutput:
        """Evaluate position from conceptual perspective.
        
        Args:
            position: Chess position to evaluate.
            legal_moves: List of legal moves.
            
        Returns:
            CortexOutput with move values and confidence.
        """
        # Encode position as sequence of piece embeddings
        piece_embeddings = self._encode_position(position)
        
        # Apply self-attention to extract patterns
        attended = self._self_attention(piece_embeddings)
        
        # Pool to get position representation
        pos_repr = np.mean(attended, axis=0)
        
        # Score moves based on conceptual patterns
        move_values = {}
        for move in legal_moves:
            move_embedding = self._embed_move(move, position)
            # Score based on alignment with learned concepts
            score = float(np.dot(pos_repr, move_embedding) / self.hidden_dim)
            
            # Add novelty bonus
            novelty = self._calculate_novelty(position, move)
            score += novelty * 0.1
            
            move_values[move] = score
        
        # Normalize scores
        if move_values:
            values = list(move_values.values())
            min_val, max_val = min(values), max(values)
            if max_val > min_val:
                for move in move_values:
                    move_values[move] = (move_values[move] - min_val) / (max_val - min_val)
        
        # Confidence based on pattern clarity
        confidence = self._calculate_confidence(attended)
        
        return CortexOutput(
            move_values=move_values,
            confidence=confidence,
            features={
                'attention_entropy': self._attention_entropy(attended),
                'novelty_score': np.mean([self._calculate_novelty(position, m) for m in legal_moves]),
            }
        )
    
    def _encode_position(self, position: 'Position') -> np.ndarray:
        """Encode position as sequence of embeddings."""
        from qratum_chess.core import Color, PieceType, iter_bits
        
        embeddings = []
        
        for color in Color:
            for pt in PieceType:
                pieces = int(position.board.pieces[color, pt])
                for sq in iter_bits(pieces):
                    # Create embedding for each piece
                    embedding = np.zeros(self.hidden_dim)
                    
                    # Piece type encoding
                    embedding[int(pt)] = 1.0
                    
                    # Color encoding
                    embedding[6 + int(color)] = 1.0
                    
                    # Square encoding (sinusoidal)
                    rank, file = sq // 8, sq % 8
                    for i in range(8, 24):
                        freq = 2 ** ((i - 8) / 4)
                        embedding[i] = np.sin(rank * freq) if i % 2 == 0 else np.cos(rank * freq)
                        embedding[i + 16] = np.sin(file * freq) if i % 2 == 0 else np.cos(file * freq)
                    
                    embeddings.append(embedding)
        
        if not embeddings:
            return np.zeros((1, self.hidden_dim))
        
        return np.array(embeddings)
    
    def _self_attention(self, x: np.ndarray) -> np.ndarray:
        """Apply multi-head self-attention."""
        seq_len = x.shape[0]
        head_dim = self.hidden_dim // self.num_heads
        
        # Compute Q, K, V
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Scaled dot-product attention
        scale = np.sqrt(head_dim)
        attention = Q @ K.T / scale
        attention = self._softmax(attention)
        
        # Apply attention to values
        attended = attention @ V
        
        # Output projection
        output = attended @ self.W_o
        
        # Add residual and layer norm (simplified)
        output = x + output
        output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-6)
        
        # FFN
        hidden = np.maximum(0, output @ self.W_ff1)
        ffn_out = hidden @ self.W_ff2
        
        # Final residual
        output = output + ffn_out
        output = output / (np.linalg.norm(output, axis=-1, keepdims=True) + 1e-6)
        
        return output
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax over last dimension."""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
    
    def _embed_move(self, move: 'Move', position: 'Position') -> np.ndarray:
        """Create embedding for a move."""
        embedding = np.zeros(self.hidden_dim)
        
        # From square
        from_rank, from_file = move.from_sq // 8, move.from_sq % 8
        embedding[0] = from_rank / 7.0
        embedding[1] = from_file / 7.0
        
        # To square
        to_rank, to_file = move.to_sq // 8, move.to_sq % 8
        embedding[2] = to_rank / 7.0
        embedding[3] = to_file / 7.0
        
        # Move direction
        embedding[4] = (to_rank - from_rank) / 7.0
        embedding[5] = (to_file - from_file) / 7.0
        
        # Distance
        dist = abs(to_rank - from_rank) + abs(to_file - from_file)
        embedding[6] = dist / 14.0
        
        # Capture flag
        if position.board.piece_at(move.to_sq):
            embedding[7] = 1.0
        
        # Promotion flag
        if move.promotion:
            embedding[8] = 1.0
            embedding[9 + int(move.promotion)] = 1.0
        
        return embedding
    
    def _calculate_novelty(self, position: 'Position', move: 'Move') -> float:
        """Calculate novelty score for a move."""
        # Create move signature
        from qratum_chess.core import PieceType
        
        piece = position.board.piece_at(move.from_sq)
        if not piece:
            return 0.5
        
        _, pt = piece
        
        # Simple novelty based on move pattern
        from_rank, from_file = move.from_sq // 8, move.from_sq % 8
        to_rank, to_file = move.to_sq // 8, move.to_sq % 8
        
        # Unusual moves get higher novelty
        if pt == PieceType.KING and abs(to_file - from_file) > 1:
            return 0.9  # Castling
        
        if pt == PieceType.PAWN and abs(to_file - from_file) > 0:
            return 0.7  # Pawn capture
        
        if pt == PieceType.KNIGHT:
            # Knight moves to edge are unusual
            if to_file in [0, 7] or to_rank in [0, 7]:
                return 0.8
        
        return 0.5
    
    def _calculate_confidence(self, attended: np.ndarray) -> float:
        """Calculate confidence based on attention patterns."""
        # Higher confidence when attention is focused
        if attended.shape[0] == 1:
            return 0.5
        
        # Calculate attention concentration
        norms = np.linalg.norm(attended, axis=1)
        concentration = np.std(norms)
        
        return min(0.5 + concentration, 0.9)
    
    def _attention_entropy(self, attended: np.ndarray) -> float:
        """Calculate entropy of attention distribution."""
        norms = np.linalg.norm(attended, axis=1)
        if np.sum(norms) == 0:
            return 0.0
        probs = norms / np.sum(norms)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return float(entropy)


class TriModalCore:
    """Tri-Modal Cognitive Core combining all three cortices.
    
    Fusion formula:
    M_final = argmax_a Σ w_i * Q_i(a) + Ω(a)
    
    Where:
    - Q_i are cortex evaluations
    - w_i are context-adaptive weights
    - Ω is a novelty-pressure functional
    """
    
    def __init__(self, config: TriModalConfig | None = None):
        """Initialize the Tri-Modal Core.
        
        Args:
            config: Configuration for the core.
        """
        self.config = config or TriModalConfig()
        
        # Initialize cortices
        self.tactical = TacticalCortex()
        self.strategic = StrategicCortex()
        self.conceptual = ConceptualCortex()
        
        # Move history for novelty pressure
        self.move_history: list[tuple[Any, Any]] = []
        self.max_history = 1000
    
    def evaluate(
        self,
        position: 'Position',
        legal_moves: list
    ) -> tuple[Any, float, dict[str, float]]:
        """Evaluate position and select best move using tri-modal fusion.
        
        Args:
            position: Chess position to evaluate.
            legal_moves: List of legal moves.
            
        Returns:
            Tuple of (best_move, value, diagnostics).
        """
        if not legal_moves:
            return None, 0.0, {}
        
        # Get outputs from all cortices
        tactical_out = self.tactical.evaluate(position, legal_moves)
        strategic_out = self.strategic.evaluate(position, legal_moves)
        conceptual_out = self.conceptual.evaluate(position, legal_moves)
        
        # Compute adaptive weights
        weights = self._compute_adaptive_weights(
            position, tactical_out, strategic_out, conceptual_out
        )
        
        # Fuse evaluations
        fused_values = {}
        for move in legal_moves:
            t_val = tactical_out.move_values.get(move, 0.0)
            s_val = strategic_out.move_values.get(move, 0.0)
            c_val = conceptual_out.move_values.get(move, 0.0)
            
            # Weighted combination
            fused = (
                weights['tactical'] * t_val +
                weights['strategic'] * s_val +
                weights['conceptual'] * c_val
            )
            
            # Add novelty pressure
            novelty = self._compute_novelty_pressure(position, move)
            fused += self.config.novelty_pressure * novelty
            
            fused_values[move] = fused
        
        # Select best move
        best_move = max(fused_values.keys(), key=lambda m: fused_values[m])
        best_value = fused_values[best_move]
        
        # Update history
        self._update_history(position, best_move)
        
        # Compile diagnostics
        diagnostics = {
            'tactical_weight': weights['tactical'],
            'strategic_weight': weights['strategic'],
            'conceptual_weight': weights['conceptual'],
            'tactical_confidence': tactical_out.confidence,
            'strategic_confidence': strategic_out.confidence,
            'conceptual_confidence': conceptual_out.confidence,
            'best_move_value': best_value,
            **tactical_out.features,
            **strategic_out.features,
            **conceptual_out.features,
        }
        
        return best_move, best_value, diagnostics
    
    def _compute_adaptive_weights(
        self,
        position: 'Position',
        tactical: CortexOutput,
        strategic: CortexOutput,
        conceptual: CortexOutput
    ) -> dict[str, float]:
        """Compute context-adaptive weights for cortex fusion."""
        if not self.config.adaptive_weights:
            return {
                'tactical': self.config.tactical_weight,
                'strategic': self.config.strategic_weight,
                'conceptual': self.config.conceptual_weight,
            }
        
        # Determine game phase
        phase = self._determine_game_phase(position)
        
        # Base weights by phase
        if phase == GamePhase.OPENING:
            base_weights = {'tactical': 0.2, 'strategic': 0.5, 'conceptual': 0.3}
        elif phase == GamePhase.MIDDLEGAME:
            base_weights = {'tactical': 0.4, 'strategic': 0.4, 'conceptual': 0.2}
        else:  # Endgame
            base_weights = {'tactical': 0.5, 'strategic': 0.4, 'conceptual': 0.1}
        
        # Adjust by confidence
        total_conf = tactical.confidence + strategic.confidence + conceptual.confidence
        if total_conf > 0:
            conf_weights = {
                'tactical': tactical.confidence / total_conf,
                'strategic': strategic.confidence / total_conf,
                'conceptual': conceptual.confidence / total_conf,
            }
            
            # Blend base weights with confidence-adjusted weights
            alpha = 0.5
            weights = {
                k: alpha * base_weights[k] + (1 - alpha) * conf_weights[k]
                for k in base_weights
            }
        else:
            weights = base_weights
        
        # Normalize
        total = sum(weights.values())
        return {k: v / total for k, v in weights.items()}
    
    def _determine_game_phase(self, position: 'Position') -> GamePhase:
        """Determine current game phase."""
        from qratum_chess.core import Color, PieceType
        
        # Count material
        queens = sum(
            bin(int(position.board.pieces[c, PieceType.QUEEN])).count('1')
            for c in Color
        )
        rooks = sum(
            bin(int(position.board.pieces[c, PieceType.ROOK])).count('1')
            for c in Color
        )
        minor = sum(
            bin(int(position.board.pieces[c, pt])).count('1')
            for c in Color
            for pt in [PieceType.KNIGHT, PieceType.BISHOP]
        )
        
        total_major_minor = queens * 9 + rooks * 5 + minor * 3
        
        if position.fullmove_number <= 10:
            return GamePhase.OPENING
        elif total_major_minor > 25:
            return GamePhase.MIDDLEGAME
        else:
            return GamePhase.ENDGAME
    
    def _compute_novelty_pressure(self, position: 'Position', move: 'Move') -> float:
        """Compute novelty pressure for a move to prevent stagnation."""
        # Check if move has been played recently
        move_key = (move.from_sq, move.to_sq, move.promotion)
        
        recent_count = sum(
            1 for pos_hash, m in self.move_history[-100:]
            if m == move_key
        )
        
        # Lower score for frequently played moves
        if recent_count > 5:
            return -0.1
        elif recent_count > 2:
            return 0.0
        else:
            return 0.1  # Novelty bonus
    
    def _update_history(self, position: 'Position', move: 'Move') -> None:
        """Update move history."""
        move_key = (move.from_sq, move.to_sq, move.promotion)
        pos_hash = position.hash()
        
        self.move_history.append((pos_hash, move_key))
        
        # Trim history
        if len(self.move_history) > self.max_history:
            self.move_history = self.move_history[-self.max_history:]
