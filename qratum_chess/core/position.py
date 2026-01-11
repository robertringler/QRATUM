"""Chess position representation and move handling.

Implements complete chess position state including:
- Piece placement (via BitBoard)
- Side to move
- Castling rights
- En passant target square
- Half-move clock (for 50-move rule)
- Full-move number
- Repetition tracking
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from qratum_chess.core import (
    BitBoard,
    CastlingRights,
    Color,
    PieceType,
    SQUARE_NAMES,
    lsb,
    iter_bits,
    shift_up,
    shift_down,
    RANK_1,
    RANK_8,
    FILE_A,
    FILE_H,
)

if TYPE_CHECKING:
    pass


@dataclass
class Move:
    """Represents a chess move.
    
    Attributes:
        from_sq: Source square (0-63)
        to_sq: Destination square (0-63)
        promotion: Promotion piece type, if any
        is_castling: Whether this is a castling move
        is_en_passant: Whether this is an en passant capture
    """

    from_sq: int
    to_sq: int
    promotion: PieceType | None = None
    is_castling: bool = False
    is_en_passant: bool = False
    
    def to_uci(self) -> str:
        """Convert to UCI notation (e.g., 'e2e4', 'e7e8q')."""
        uci = SQUARE_NAMES[self.from_sq] + SQUARE_NAMES[self.to_sq]
        if self.promotion:
            promo_chars = {
                PieceType.KNIGHT: "n",
                PieceType.BISHOP: "b",
                PieceType.ROOK: "r",
                PieceType.QUEEN: "q",
            }
            uci += promo_chars.get(self.promotion, "")
        return uci
    
    @classmethod
    def from_uci(cls, uci: str) -> Move:
        """Parse UCI notation to Move."""
        from_sq = SQUARE_NAMES.index(uci[0:2])
        to_sq = SQUARE_NAMES.index(uci[2:4])
        promotion = None
        if len(uci) > 4:
            promo_map = {"n": PieceType.KNIGHT, "b": PieceType.BISHOP,
                        "r": PieceType.ROOK, "q": PieceType.QUEEN}
            promotion = promo_map.get(uci[4].lower())
        return cls(from_sq=from_sq, to_sq=to_sq, promotion=promotion)
    
    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Move):
            return False
        return (
            self.from_sq == other.from_sq
            and self.to_sq == other.to_sq
            and self.promotion == other.promotion
        )
    
    def __hash__(self) -> int:
        return hash((self.from_sq, self.to_sq, self.promotion))


@dataclass
class Position:
    """Complete chess position state.
    
    Attributes:
        board: Bitboard representation of pieces
        side_to_move: Color to move
        castling: Castling rights
        ep_square: En passant target square (-1 if none)
        halfmove_clock: Moves since last pawn move or capture (for 50-move rule)
        fullmove_number: Full move number (starts at 1)
        position_history: List of position hashes for repetition detection
    """

    board: BitBoard
    side_to_move: Color = Color.WHITE
    castling: CastlingRights = CastlingRights.ALL
    ep_square: int = -1
    halfmove_clock: int = 0
    fullmove_number: int = 1
    position_history: list[int] = field(default_factory=list)
    
    @classmethod
    def starting(cls) -> Position:
        """Create the starting position."""
        return cls(board=BitBoard.starting_position())
    
    @classmethod
    def from_fen(cls, fen: str) -> Position:
        """Parse a FEN string to create a position."""
        parts = fen.split()
        
        # Piece placement
        board = BitBoard.empty()
        rank, file = 7, 0
        
        piece_map = {
            "P": (Color.WHITE, PieceType.PAWN),
            "N": (Color.WHITE, PieceType.KNIGHT),
            "B": (Color.WHITE, PieceType.BISHOP),
            "R": (Color.WHITE, PieceType.ROOK),
            "Q": (Color.WHITE, PieceType.QUEEN),
            "K": (Color.WHITE, PieceType.KING),
            "p": (Color.BLACK, PieceType.PAWN),
            "n": (Color.BLACK, PieceType.KNIGHT),
            "b": (Color.BLACK, PieceType.BISHOP),
            "r": (Color.BLACK, PieceType.ROOK),
            "q": (Color.BLACK, PieceType.QUEEN),
            "k": (Color.BLACK, PieceType.KING),
        }
        
        for char in parts[0]:
            if char == "/":
                rank -= 1
                file = 0
            elif char.isdigit():
                file += int(char)
            else:
                color, piece_type = piece_map[char]
                board.set_piece(rank * 8 + file, color, piece_type)
                file += 1
        
        # Side to move
        side_to_move = Color.WHITE if parts[1] == "w" else Color.BLACK
        
        # Castling rights
        castling = CastlingRights.NONE
        if len(parts) > 2:
            if "K" in parts[2]:
                castling |= CastlingRights.WHITE_KINGSIDE
            if "Q" in parts[2]:
                castling |= CastlingRights.WHITE_QUEENSIDE
            if "k" in parts[2]:
                castling |= CastlingRights.BLACK_KINGSIDE
            if "q" in parts[2]:
                castling |= CastlingRights.BLACK_QUEENSIDE
        
        # En passant square
        ep_square = -1
        if len(parts) > 3 and parts[3] != "-":
            ep_square = SQUARE_NAMES.index(parts[3])
        
        # Half-move clock
        halfmove_clock = int(parts[4]) if len(parts) > 4 else 0
        
        # Full-move number
        fullmove_number = int(parts[5]) if len(parts) > 5 else 1
        
        return cls(
            board=board,
            side_to_move=side_to_move,
            castling=castling,
            ep_square=ep_square,
            halfmove_clock=halfmove_clock,
            fullmove_number=fullmove_number,
        )
    
    def to_fen(self) -> str:
        """Convert position to FEN string."""
        # Piece placement
        piece_chars = {
            (Color.WHITE, PieceType.PAWN): "P",
            (Color.WHITE, PieceType.KNIGHT): "N",
            (Color.WHITE, PieceType.BISHOP): "B",
            (Color.WHITE, PieceType.ROOK): "R",
            (Color.WHITE, PieceType.QUEEN): "Q",
            (Color.WHITE, PieceType.KING): "K",
            (Color.BLACK, PieceType.PAWN): "p",
            (Color.BLACK, PieceType.KNIGHT): "n",
            (Color.BLACK, PieceType.BISHOP): "b",
            (Color.BLACK, PieceType.ROOK): "r",
            (Color.BLACK, PieceType.QUEEN): "q",
            (Color.BLACK, PieceType.KING): "k",
        }
        
        fen_parts = []
        for rank in range(7, -1, -1):
            empty = 0
            rank_str = ""
            for file in range(8):
                sq = rank * 8 + file
                piece = self.board.piece_at(sq)
                if piece:
                    if empty > 0:
                        rank_str += str(empty)
                        empty = 0
                    rank_str += piece_chars[piece]
                else:
                    empty += 1
            if empty > 0:
                rank_str += str(empty)
            fen_parts.append(rank_str)
        
        position_str = "/".join(fen_parts)
        
        # Side to move
        side = "w" if self.side_to_move == Color.WHITE else "b"
        
        # Castling rights
        castling_str = ""
        if self.castling & CastlingRights.WHITE_KINGSIDE:
            castling_str += "K"
        if self.castling & CastlingRights.WHITE_QUEENSIDE:
            castling_str += "Q"
        if self.castling & CastlingRights.BLACK_KINGSIDE:
            castling_str += "k"
        if self.castling & CastlingRights.BLACK_QUEENSIDE:
            castling_str += "q"
        if not castling_str:
            castling_str = "-"
        
        # En passant
        ep_str = SQUARE_NAMES[self.ep_square] if self.ep_square >= 0 else "-"
        
        return f"{position_str} {side} {castling_str} {ep_str} {self.halfmove_clock} {self.fullmove_number}"
    
    def hash(self) -> int:
        """Compute a hash of the position for repetition detection."""
        h = 0
        for color in Color:
            for pt in PieceType:
                h ^= int(self.board.pieces[color, pt]) * (color * 6 + pt + 1)
        h ^= int(self.side_to_move) << 60
        h ^= int(self.castling) << 56
        h ^= (self.ep_square + 1) << 48
        return h
    
    def is_in_check(self, color: Color | None = None) -> bool:
        """Check if the given color's king is in check."""
        if color is None:
            color = self.side_to_move
        
        king_sq = lsb(int(self.board.pieces[color, PieceType.KING]))
        if king_sq < 0:
            return False
        
        opponent = Color(1 - color)
        attacks = self.board.get_attack_map(opponent)
        return bool(attacks & (1 << king_sq))
    
    def is_checkmate(self) -> bool:
        """Check if the current position is checkmate."""
        if not self.is_in_check():
            return False
        return len(self.generate_legal_moves()) == 0
    
    def is_stalemate(self) -> bool:
        """Check if the current position is stalemate."""
        if self.is_in_check():
            return False
        return len(self.generate_legal_moves()) == 0
    
    def is_draw(self) -> bool:
        """Check for draw conditions (50-move rule, repetition, insufficient material)."""
        # 50-move rule
        if self.halfmove_clock >= 100:
            return True
        
        # Threefold repetition
        current_hash = self.hash()
        if self.position_history.count(current_hash) >= 2:
            return True
        
        # Insufficient material
        if self._is_insufficient_material():
            return True
        
        return False
    
    def _is_insufficient_material(self) -> bool:
        """Check for insufficient material to checkmate."""
        # Count pieces
        white_pieces = sum(
            bin(int(self.board.pieces[Color.WHITE, pt])).count("1")
            for pt in PieceType if pt != PieceType.KING
        )
        black_pieces = sum(
            bin(int(self.board.pieces[Color.BLACK, pt])).count("1")
            for pt in PieceType if pt != PieceType.KING
        )
        
        # King vs King
        if white_pieces == 0 and black_pieces == 0:
            return True
        
        # King + minor piece vs King
        if white_pieces + black_pieces == 1:
            for color in Color:
                knights = bin(int(self.board.pieces[color, PieceType.KNIGHT])).count("1")
                bishops = bin(int(self.board.pieces[color, PieceType.BISHOP])).count("1")
                if knights == 1 or bishops == 1:
                    return True
        
        return False
    
    def generate_pseudo_legal_moves(self) -> list[Move]:
        """Generate all pseudo-legal moves (may leave king in check)."""
        moves = []
        color = self.side_to_move
        opponent = Color(1 - color)
        
        # Pawn moves
        moves.extend(self._generate_pawn_moves())
        
        # Knight moves
        knights = int(self.board.pieces[color, PieceType.KNIGHT])
        for from_sq in iter_bits(knights):
            from qratum_chess.core import KNIGHT_ATTACKS
            attacks = int(KNIGHT_ATTACKS[from_sq])
            targets = attacks & ~int(self.board.occupancy[color])
            for to_sq in iter_bits(targets):
                moves.append(Move(from_sq, to_sq))
        
        # Bishop moves
        self._add_sliding_moves(moves, PieceType.BISHOP)
        
        # Rook moves
        self._add_sliding_moves(moves, PieceType.ROOK)
        
        # Queen moves
        self._add_sliding_moves(moves, PieceType.QUEEN)
        
        # King moves
        king_sq = lsb(int(self.board.pieces[color, PieceType.KING]))
        if king_sq >= 0:
            from qratum_chess.core import KING_ATTACKS
            attacks = int(KING_ATTACKS[king_sq])
            targets = attacks & ~int(self.board.occupancy[color])
            for to_sq in iter_bits(targets):
                moves.append(Move(from_sq=king_sq, to_sq=to_sq))
        
        # Castling
        moves.extend(self._generate_castling_moves())
        
        return moves
    
    def _generate_pawn_moves(self) -> list[Move]:
        """Generate pawn moves."""
        moves = []
        color = self.side_to_move
        opponent = Color(1 - color)
        pawns = int(self.board.pieces[color, PieceType.PAWN])
        
        if color == Color.WHITE:
            direction = 8
            start_rank = RANK_1 << 8  # Rank 2
            promo_rank = 7
        else:
            direction = -8
            start_rank = RANK_8 >> 8  # Rank 7
            promo_rank = 0
        
        empty = ~self.board.all_pieces
        opponent_pieces = int(self.board.occupancy[opponent])
        
        for from_sq in iter_bits(pawns):
            # Single push
            to_sq = from_sq + direction
            if 0 <= to_sq < 64 and (empty & (1 << to_sq)):
                if to_sq // 8 == promo_rank:
                    for promo in [PieceType.QUEEN, PieceType.ROOK,
                                  PieceType.BISHOP, PieceType.KNIGHT]:
                        moves.append(Move(from_sq, to_sq, promotion=promo))
                else:
                    moves.append(Move(from_sq, to_sq))
                
                # Double push from starting rank
                if (1 << from_sq) & start_rank:
                    to_sq2 = from_sq + 2 * direction
                    if empty & (1 << to_sq2):
                        moves.append(Move(from_sq, to_sq2))
            
            # Captures
            from_file = from_sq % 8
            for capture_offset in [direction - 1, direction + 1]:
                to_sq = from_sq + capture_offset
                to_file = to_sq % 8
                
                if 0 <= to_sq < 64 and abs(to_file - from_file) == 1:
                    is_capture = bool(opponent_pieces & (1 << to_sq))
                    is_ep = (to_sq == self.ep_square)
                    
                    if is_capture or is_ep:
                        if to_sq // 8 == promo_rank:
                            for promo in [PieceType.QUEEN, PieceType.ROOK,
                                          PieceType.BISHOP, PieceType.KNIGHT]:
                                moves.append(Move(from_sq, to_sq, promotion=promo))
                        else:
                            moves.append(Move(from_sq, to_sq, is_en_passant=is_ep))
        
        return moves
    
    def _add_sliding_moves(self, moves: list[Move], piece_type: PieceType) -> None:
        """Add sliding piece moves to the move list."""
        color = self.side_to_move
        pieces = int(self.board.pieces[color, piece_type])
        
        if piece_type == PieceType.BISHOP:
            directions = [7, 9, -7, -9]
        elif piece_type == PieceType.ROOK:
            directions = [1, -1, 8, -8]
        else:  # Queen
            directions = [1, -1, 8, -8, 7, 9, -7, -9]
        
        for from_sq in iter_bits(pieces):
            for direction in directions:
                current = from_sq
                while True:
                    prev_file = current % 8
                    prev_rank = current // 8
                    current += direction
                    
                    if current < 0 or current > 63:
                        break
                    
                    curr_file = current % 8
                    curr_rank = current // 8
                    
                    # Check for edge wrapping based on direction
                    # For orthogonal moves (±1), file should change by 1
                    # For orthogonal moves (±8), file should not change
                    # For diagonal moves, both rank and file should change by 1
                    if direction in [1, -1]:
                        if abs(curr_file - prev_file) != 1:
                            break
                    elif direction in [8, -8]:
                        if curr_file != prev_file:
                            break
                    else:  # Diagonal: ±7, ±9
                        if abs(curr_file - prev_file) != 1 or abs(curr_rank - prev_rank) != 1:
                            break
                    
                    if int(self.board.occupancy[color]) & (1 << current):
                        break
                    
                    moves.append(Move(from_sq, current))
                    
                    if self.board.all_pieces & (1 << current):
                        break
    
    def _generate_castling_moves(self) -> list[Move]:
        """Generate castling moves."""
        moves = []
        color = self.side_to_move
        
        if self.is_in_check():
            return moves
        
        if color == Color.WHITE:
            king_sq = 4  # e1
            if self.castling & CastlingRights.WHITE_KINGSIDE:
                if not (self.board.all_pieces & 0x60):  # f1, g1 empty
                    opponent_attacks = self.board.get_attack_map(Color.BLACK)
                    if not (opponent_attacks & 0x70):  # e1, f1, g1 not attacked
                        moves.append(Move(king_sq, 6, is_castling=True))  # e1-g1
            
            if self.castling & CastlingRights.WHITE_QUEENSIDE:
                if not (self.board.all_pieces & 0x0E):  # b1, c1, d1 empty
                    opponent_attacks = self.board.get_attack_map(Color.BLACK)
                    if not (opponent_attacks & 0x1C):  # c1, d1, e1 not attacked
                        moves.append(Move(king_sq, 2, is_castling=True))  # e1-c1
        else:
            king_sq = 60  # e8
            if self.castling & CastlingRights.BLACK_KINGSIDE:
                if not (self.board.all_pieces & 0x6000000000000000):  # f8, g8 empty
                    opponent_attacks = self.board.get_attack_map(Color.WHITE)
                    if not (opponent_attacks & 0x7000000000000000):
                        moves.append(Move(king_sq, 62, is_castling=True))  # e8-g8
            
            if self.castling & CastlingRights.BLACK_QUEENSIDE:
                if not (self.board.all_pieces & 0x0E00000000000000):  # b8, c8, d8 empty
                    opponent_attacks = self.board.get_attack_map(Color.WHITE)
                    if not (opponent_attacks & 0x1C00000000000000):
                        moves.append(Move(king_sq, 58, is_castling=True))  # e8-c8
        
        return moves
    
    def generate_legal_moves(self) -> list[Move]:
        """Generate all legal moves."""
        legal_moves = []
        for move in self.generate_pseudo_legal_moves():
            new_pos = self.make_move(move)
            if not new_pos.is_in_check(self.side_to_move):
                legal_moves.append(move)
        return legal_moves
    
    def make_move(self, move: Move) -> Position:
        """Make a move and return the new position."""
        new_board = self.board.copy()
        color = self.side_to_move
        opponent = Color(1 - color)
        
        # Get moving piece
        piece = self.board.piece_at(move.from_sq)
        if piece is None:
            raise ValueError(f"No piece at {SQUARE_NAMES[move.from_sq]}")
        
        _, piece_type = piece
        
        # Handle captures
        captured_piece = self.board.piece_at(move.to_sq)
        if captured_piece:
            cap_color, cap_type = captured_piece
            new_board.remove_piece(move.to_sq, cap_color, cap_type)
        
        # Handle en passant capture
        if move.is_en_passant:
            ep_captured_sq = move.to_sq + (-8 if color == Color.WHITE else 8)
            new_board.remove_piece(ep_captured_sq, opponent, PieceType.PAWN)
        
        # Move the piece
        new_board.remove_piece(move.from_sq, color, piece_type)
        
        # Handle promotion
        if move.promotion:
            new_board.set_piece(move.to_sq, color, move.promotion)
        else:
            new_board.set_piece(move.to_sq, color, piece_type)
        
        # Handle castling
        if move.is_castling:
            if move.to_sq > move.from_sq:  # Kingside
                rook_from = move.from_sq + 3
                rook_to = move.from_sq + 1
            else:  # Queenside
                rook_from = move.from_sq - 4
                rook_to = move.from_sq - 1
            new_board.move_piece(rook_from, rook_to, color, PieceType.ROOK)
        
        # Update castling rights
        new_castling = self.castling
        if piece_type == PieceType.KING:
            if color == Color.WHITE:
                new_castling &= ~CastlingRights.WHITE_BOTH
            else:
                new_castling &= ~CastlingRights.BLACK_BOTH
        elif piece_type == PieceType.ROOK:
            if move.from_sq == 0:  # a1
                new_castling &= ~CastlingRights.WHITE_QUEENSIDE
            elif move.from_sq == 7:  # h1
                new_castling &= ~CastlingRights.WHITE_KINGSIDE
            elif move.from_sq == 56:  # a8
                new_castling &= ~CastlingRights.BLACK_QUEENSIDE
            elif move.from_sq == 63:  # h8
                new_castling &= ~CastlingRights.BLACK_KINGSIDE
        
        # Handle rook captures for castling rights
        if move.to_sq == 0:
            new_castling &= ~CastlingRights.WHITE_QUEENSIDE
        elif move.to_sq == 7:
            new_castling &= ~CastlingRights.WHITE_KINGSIDE
        elif move.to_sq == 56:
            new_castling &= ~CastlingRights.BLACK_QUEENSIDE
        elif move.to_sq == 63:
            new_castling &= ~CastlingRights.BLACK_KINGSIDE
        
        # Update en passant square
        new_ep_square = -1
        if piece_type == PieceType.PAWN and abs(move.to_sq - move.from_sq) == 16:
            new_ep_square = (move.from_sq + move.to_sq) // 2
        
        # Update clocks
        new_halfmove = 0 if piece_type == PieceType.PAWN or captured_piece else self.halfmove_clock + 1
        new_fullmove = self.fullmove_number + (1 if color == Color.BLACK else 0)
        
        # Update position history
        new_history = self.position_history.copy()
        new_history.append(self.hash())
        
        return Position(
            board=new_board,
            side_to_move=opponent,
            castling=new_castling,
            ep_square=new_ep_square,
            halfmove_clock=new_halfmove,
            fullmove_number=new_fullmove,
            position_history=new_history,
        )
    
    def copy(self) -> Position:
        """Create a deep copy of the position."""
        return Position(
            board=self.board.copy(),
            side_to_move=self.side_to_move,
            castling=self.castling,
            ep_square=self.ep_square,
            halfmove_clock=self.halfmove_clock,
            fullmove_number=self.fullmove_number,
            position_history=self.position_history.copy(),
        )
    
    def __str__(self) -> str:
        """String representation of the position."""
        return f"{self.board.to_string()}\nFEN: {self.to_fen()}"
