"""Position encoding for neural network input.

Converts chess positions to tensor representations suitable for neural network
evaluation. Implements the multi-channel input encoding described in Stage II.

Input Tensor (8×8×C):
- 12 planes: One-hot planes for {P,N,B,R,Q,K} × {white, black}
- 4 planes: Castling rights (WK,WQ,BK,BQ)
- 1 plane: Side to move
- 1 plane: En-passant file mask
- 6 planes: Repetition counter, half-move clock, phase-of-game encoding
- 4 planes: Attack maps (white, black, union, difference)

Total: 28 input channels
"""

from __future__ import annotations

import numpy as np

from qratum_chess.core import (
    BitBoard,
    CastlingRights,
    Color,
    PieceType,
    iter_bits,
)
from qratum_chess.core.position import Position


class PositionEncoder:
    """Encodes chess positions as neural network input tensors.
    
    The encoder produces an 8×8×C tensor where C is the number of channels.
    Default encoding uses 28 channels for comprehensive position representation.
    """
    
    # Number of input channels
    NUM_PIECE_PLANES = 12  # 6 piece types × 2 colors
    NUM_CASTLING_PLANES = 4
    NUM_SIDE_PLANES = 1
    NUM_EP_PLANES = 1
    NUM_GAME_STATE_PLANES = 6  # repetition, halfmove clock, game phase
    NUM_ATTACK_PLANES = 4  # white attacks, black attacks, union, difference
    
    TOTAL_CHANNELS = (
        NUM_PIECE_PLANES
        + NUM_CASTLING_PLANES
        + NUM_SIDE_PLANES
        + NUM_EP_PLANES
        + NUM_GAME_STATE_PLANES
        + NUM_ATTACK_PLANES
    )  # = 28
    
    def __init__(self, include_attack_maps: bool = True):
        """Initialize the position encoder.
        
        Args:
            include_attack_maps: Whether to include attack map channels.
        """
        self.include_attack_maps = include_attack_maps
        if not include_attack_maps:
            self.num_channels = self.TOTAL_CHANNELS - self.NUM_ATTACK_PLANES
        else:
            self.num_channels = self.TOTAL_CHANNELS
    
    def encode(self, position: Position) -> np.ndarray:
        """Encode a position as an input tensor.
        
        Args:
            position: Chess position to encode.
            
        Returns:
            NumPy array of shape (C, 8, 8) with dtype float32.
        """
        tensor = np.zeros((self.num_channels, 8, 8), dtype=np.float32)
        channel = 0
        
        # Piece planes (12 channels)
        for color in Color:
            for piece_type in PieceType:
                bb = int(position.board.pieces[color, piece_type])
                for sq in iter_bits(bb):
                    rank, file = sq // 8, sq % 8
                    tensor[channel, rank, file] = 1.0
                channel += 1
        
        # Castling rights (4 channels)
        castling_rights = [
            CastlingRights.WHITE_KINGSIDE,
            CastlingRights.WHITE_QUEENSIDE,
            CastlingRights.BLACK_KINGSIDE,
            CastlingRights.BLACK_QUEENSIDE,
        ]
        for right in castling_rights:
            if position.castling & right:
                tensor[channel, :, :] = 1.0
            channel += 1
        
        # Side to move (1 channel)
        if position.side_to_move == Color.WHITE:
            tensor[channel, :, :] = 1.0
        channel += 1
        
        # En passant file mask (1 channel)
        if position.ep_square >= 0:
            ep_file = position.ep_square % 8
            tensor[channel, :, ep_file] = 1.0
        channel += 1
        
        # Game state planes (6 channels)
        # Repetition counter (normalized)
        current_hash = position.hash()
        rep_count = position.position_history.count(current_hash)
        tensor[channel, :, :] = min(rep_count / 2.0, 1.0)
        channel += 1
        
        # Half-move clock (normalized to 0-1, capped at 100)
        tensor[channel, :, :] = min(position.halfmove_clock / 100.0, 1.0)
        channel += 1
        
        # Game phase encoding (4 channels for opening/middlegame/endgame/late endgame)
        phase = self._compute_game_phase(position)
        tensor[channel, :, :] = phase[0]  # Opening
        channel += 1
        tensor[channel, :, :] = phase[1]  # Middlegame
        channel += 1
        tensor[channel, :, :] = phase[2]  # Endgame
        channel += 1
        tensor[channel, :, :] = phase[3]  # Late endgame
        channel += 1
        
        # Attack maps (4 channels)
        if self.include_attack_maps:
            white_attacks = position.board.get_attack_map(Color.WHITE)
            black_attacks = position.board.get_attack_map(Color.BLACK)
            
            # White attack map
            for sq in iter_bits(white_attacks):
                rank, file = sq // 8, sq % 8
                tensor[channel, rank, file] = 1.0
            channel += 1
            
            # Black attack map
            for sq in iter_bits(black_attacks):
                rank, file = sq // 8, sq % 8
                tensor[channel, rank, file] = 1.0
            channel += 1
            
            # Union of attacks
            union = white_attacks | black_attacks
            for sq in iter_bits(union):
                rank, file = sq // 8, sq % 8
                tensor[channel, rank, file] = 1.0
            channel += 1
            
            # Difference (contested squares)
            intersection = white_attacks & black_attacks
            for sq in iter_bits(intersection):
                rank, file = sq // 8, sq % 8
                tensor[channel, rank, file] = 1.0
            channel += 1
        
        return tensor
    
    def _compute_game_phase(self, position: Position) -> tuple[float, float, float, float]:
        """Compute game phase as a 4-tuple (opening, middlegame, endgame, late_endgame).
        
        Uses material count to determine game phase.
        """
        # Material values for phase calculation
        piece_values = {
            PieceType.PAWN: 1,
            PieceType.KNIGHT: 3,
            PieceType.BISHOP: 3,
            PieceType.ROOK: 5,
            PieceType.QUEEN: 9,
            PieceType.KING: 0,
        }
        
        total_material = 0
        for color in Color:
            for pt in PieceType:
                count = bin(int(position.board.pieces[color, pt])).count("1")
                total_material += count * piece_values[pt]
        
        # Maximum material (excluding kings): 2*(8*1 + 2*3 + 2*3 + 2*5 + 1*9) = 78
        max_material = 78
        material_ratio = total_material / max_material
        
        # Phase boundaries
        if material_ratio > 0.8:
            return (1.0, 0.0, 0.0, 0.0)  # Opening
        elif material_ratio > 0.5:
            return (0.0, 1.0, 0.0, 0.0)  # Middlegame
        elif material_ratio > 0.2:
            return (0.0, 0.0, 1.0, 0.0)  # Endgame
        else:
            return (0.0, 0.0, 0.0, 1.0)  # Late endgame
    
    def encode_batch(self, positions: list[Position]) -> np.ndarray:
        """Encode multiple positions as a batch tensor.
        
        Args:
            positions: List of chess positions.
            
        Returns:
            NumPy array of shape (N, C, 8, 8) with dtype float32.
        """
        batch = np.zeros((len(positions), self.num_channels, 8, 8), dtype=np.float32)
        for i, pos in enumerate(positions):
            batch[i] = self.encode(pos)
        return batch


class MoveEncoder:
    """Encodes chess moves as neural network policy targets.
    
    Maps moves to a flat index space for policy head output.
    Total move space: 4672 possible moves (including promotions).
    """
    
    # Move encoding constants
    QUEEN_MOVES = 56  # 7 directions × 8 distances (though limited by board)
    KNIGHT_MOVES = 8
    UNDERPROMOTIONS = 9  # 3 underpromotion types × 3 directions
    
    TOTAL_MOVES = 73 * 64  # Maximum moves per square × 64 squares = 4672
    
    def __init__(self):
        """Initialize the move encoder."""
        self._move_to_idx: dict[tuple[int, int, int], int] = {}
        self._idx_to_move: dict[int, tuple[int, int, int]] = {}
        self._build_move_index()
    
    def _build_move_index(self) -> None:
        """Build the move index mappings."""
        idx = 0
        
        for from_sq in range(64):
            from_rank, from_file = from_sq // 8, from_sq % 8
            
            # Queen-type moves (all directions, all distances)
            directions = [
                (0, 1), (0, -1), (1, 0), (-1, 0),  # Orthogonal
                (1, 1), (1, -1), (-1, 1), (-1, -1),  # Diagonal
            ]
            
            for dr, df in directions:
                for dist in range(1, 8):
                    to_rank = from_rank + dr * dist
                    to_file = from_file + df * dist
                    if 0 <= to_rank < 8 and 0 <= to_file < 8:
                        to_sq = to_rank * 8 + to_file
                        key = (from_sq, to_sq, 0)  # 0 = no promotion
                        self._move_to_idx[key] = idx
                        self._idx_to_move[idx] = key
                        idx += 1
            
            # Knight moves
            knight_offsets = [
                (2, 1), (2, -1), (-2, 1), (-2, -1),
                (1, 2), (1, -2), (-1, 2), (-1, -2),
            ]
            for dr, df in knight_offsets:
                to_rank = from_rank + dr
                to_file = from_file + df
                if 0 <= to_rank < 8 and 0 <= to_file < 8:
                    to_sq = to_rank * 8 + to_file
                    key = (from_sq, to_sq, 0)
                    if key not in self._move_to_idx:
                        self._move_to_idx[key] = idx
                        self._idx_to_move[idx] = key
                        idx += 1
            
            # Underpromotions (only from 7th rank for white, 2nd for black)
            if from_rank in [1, 6]:  # Possible promotion ranks
                for df in [-1, 0, 1]:  # Capture left, push, capture right
                    dr = 1 if from_rank == 6 else -1
                    to_rank = from_rank + dr
                    to_file = from_file + df
                    if 0 <= to_file < 8 and 0 <= to_rank < 8:
                        to_sq = to_rank * 8 + to_file
                        for promo in [1, 2, 3]:  # Knight, Bishop, Rook
                            key = (from_sq, to_sq, promo)
                            if key not in self._move_to_idx:
                                self._move_to_idx[key] = idx
                                self._idx_to_move[idx] = key
                                idx += 1
        
        self.num_moves = idx
    
    def encode_move(self, from_sq: int, to_sq: int, promotion: int = 0) -> int:
        """Encode a move as an index.
        
        Args:
            from_sq: Source square (0-63).
            to_sq: Target square (0-63).
            promotion: Promotion type (0=none/queen, 1=knight, 2=bishop, 3=rook).
            
        Returns:
            Move index.
        """
        key = (from_sq, to_sq, promotion)
        return self._move_to_idx.get(key, -1)
    
    def decode_move(self, idx: int) -> tuple[int, int, int]:
        """Decode a move index.
        
        Args:
            idx: Move index.
            
        Returns:
            Tuple of (from_sq, to_sq, promotion).
        """
        return self._idx_to_move.get(idx, (-1, -1, 0))


# Global encoder instances
POSITION_ENCODER = PositionEncoder()
MOVE_ENCODER = MoveEncoder()
