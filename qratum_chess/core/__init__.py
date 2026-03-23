"""Bitboard representation and utilities for chess positions.

Implements 64-bit integer bitboards for efficient chess position representation
and manipulation. Supports piece placement, attack maps, and move generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum, IntFlag
from typing import Iterator

import numpy as np


class PieceType(IntEnum):
    """Chess piece types."""

    PAWN = 0
    KNIGHT = 1
    BISHOP = 2
    ROOK = 3
    QUEEN = 4
    KING = 5


class Color(IntEnum):
    """Chess piece colors."""

    WHITE = 0
    BLACK = 1


class CastlingRights(IntFlag):
    """Castling rights flags."""

    NONE = 0
    WHITE_KINGSIDE = 1
    WHITE_QUEENSIDE = 2
    BLACK_KINGSIDE = 4
    BLACK_QUEENSIDE = 8
    WHITE_BOTH = WHITE_KINGSIDE | WHITE_QUEENSIDE
    BLACK_BOTH = BLACK_KINGSIDE | BLACK_QUEENSIDE
    ALL = WHITE_BOTH | BLACK_BOTH


# Square indices (a1 = 0, h8 = 63)
SQUARE_NAMES = [f"{file}{rank}" for rank in range(1, 9) for file in "abcdefgh"]

# File and rank masks
FILE_A = 0x0101010101010101
FILE_B = FILE_A << 1
FILE_C = FILE_A << 2
FILE_D = FILE_A << 3
FILE_E = FILE_A << 4
FILE_F = FILE_A << 5
FILE_G = FILE_A << 6
FILE_H = FILE_A << 7

RANK_1 = 0xFF
RANK_2 = RANK_1 << 8
RANK_3 = RANK_1 << 16
RANK_4 = RANK_1 << 24
RANK_5 = RANK_1 << 32
RANK_6 = RANK_1 << 40
RANK_7 = RANK_1 << 48
RANK_8 = RANK_1 << 56

# Pre-computed knight attack masks
KNIGHT_ATTACKS = np.zeros(64, dtype=np.uint64)
for sq in range(64):
    attacks = np.uint64(0)
    bb = np.uint64(1) << sq
    # Knight move patterns
    if sq % 8 > 1:  # Not on a/b files
        if sq < 56:
            attacks |= bb << np.uint64(6)
        if sq >= 8:
            attacks |= bb >> np.uint64(10)
    if sq % 8 > 0:  # Not on a file
        if sq < 48:
            attacks |= bb << np.uint64(15)
        if sq >= 16:
            attacks |= bb >> np.uint64(17)
    if sq % 8 < 7:  # Not on h file
        if sq < 48:
            attacks |= bb << np.uint64(17)
        if sq >= 16:
            attacks |= bb >> np.uint64(15)
    if sq % 8 < 6:  # Not on g/h files
        if sq < 56:
            attacks |= bb << np.uint64(10)
        if sq >= 8:
            attacks |= bb >> np.uint64(6)
    KNIGHT_ATTACKS[sq] = attacks

# Pre-computed king attack masks
KING_ATTACKS = np.zeros(64, dtype=np.uint64)
for sq in range(64):
    attacks = np.uint64(0)
    bb = np.uint64(1) << sq
    if sq % 8 > 0:  # Not on a file
        attacks |= bb >> np.uint64(1)
        if sq < 56:
            attacks |= bb << np.uint64(7)
        if sq >= 8:
            attacks |= bb >> np.uint64(9)
    if sq % 8 < 7:  # Not on h file
        attacks |= bb << np.uint64(1)
        if sq < 56:
            attacks |= bb << np.uint64(9)
        if sq >= 8:
            attacks |= bb >> np.uint64(7)
    if sq < 56:
        attacks |= bb << np.uint64(8)
    if sq >= 8:
        attacks |= bb >> np.uint64(8)
    KING_ATTACKS[sq] = attacks


def popcount(bb: int) -> int:
    """Count the number of set bits in a bitboard."""
    return bin(bb).count("1")


def lsb(bb: int) -> int:
    """Get the index of the least significant bit."""
    if bb == 0:
        return -1
    return (bb & -bb).bit_length() - 1


def msb(bb: int) -> int:
    """Get the index of the most significant bit."""
    if bb == 0:
        return -1
    return bb.bit_length() - 1


def iter_bits(bb: int) -> Iterator[int]:
    """Iterate over set bits in a bitboard."""
    while bb:
        sq = lsb(bb)
        yield sq
        bb &= bb - 1


def shift_up(bb: int) -> int:
    """Shift bitboard one rank up."""
    return (bb << 8) & 0xFFFFFFFFFFFFFFFF


def shift_down(bb: int) -> int:
    """Shift bitboard one rank down."""
    return bb >> 8


def shift_left(bb: int) -> int:
    """Shift bitboard one file left."""
    return (bb >> 1) & ~FILE_H


def shift_right(bb: int) -> int:
    """Shift bitboard one file right."""
    return (bb << 1) & ~FILE_A


@dataclass
class BitBoard:
    """Bitboard representation for all pieces on the board.

    Uses 12 bitboards: 6 piece types × 2 colors.
    Each bitboard is a 64-bit integer where bit i is set if the piece is on square i.
    """

    # Piece bitboards [color][piece_type]
    pieces: np.ndarray  # Shape: (2, 6), dtype: uint64

    # Combined occupancy bitboards
    occupancy: np.ndarray = field(default=None)  # Shape: (2,), dtype: uint64 - per color
    all_pieces: int = 0  # All pieces combined

    def __post_init__(self) -> None:
        """Initialize derived attributes."""
        self._update_occupancy()

    def _update_occupancy(self) -> None:
        """Update occupancy bitboards from piece bitboards."""
        self.occupancy = np.array(
            [
                sum(int(self.pieces[Color.WHITE, pt]) for pt in PieceType),
                sum(int(self.pieces[Color.BLACK, pt]) for pt in PieceType),
            ],
            dtype=np.uint64,
        )
        self.all_pieces = int(self.occupancy[0]) | int(self.occupancy[1])

    @classmethod
    def empty(cls) -> BitBoard:
        """Create an empty board."""
        return cls(pieces=np.zeros((2, 6), dtype=np.uint64))

    @classmethod
    def starting_position(cls) -> BitBoard:
        """Create the starting chess position."""
        pieces = np.zeros((2, 6), dtype=np.uint64)

        # White pieces
        pieces[Color.WHITE, PieceType.PAWN] = RANK_2
        pieces[Color.WHITE, PieceType.ROOK] = 0x81  # a1, h1
        pieces[Color.WHITE, PieceType.KNIGHT] = 0x42  # b1, g1
        pieces[Color.WHITE, PieceType.BISHOP] = 0x24  # c1, f1
        pieces[Color.WHITE, PieceType.QUEEN] = 0x08  # d1
        pieces[Color.WHITE, PieceType.KING] = 0x10  # e1

        # Black pieces
        pieces[Color.BLACK, PieceType.PAWN] = RANK_7
        pieces[Color.BLACK, PieceType.ROOK] = 0x8100000000000000  # a8, h8
        pieces[Color.BLACK, PieceType.KNIGHT] = 0x4200000000000000  # b8, g8
        pieces[Color.BLACK, PieceType.BISHOP] = 0x2400000000000000  # c8, f8
        pieces[Color.BLACK, PieceType.QUEEN] = 0x0800000000000000  # d8
        pieces[Color.BLACK, PieceType.KING] = 0x1000000000000000  # e8

        return cls(pieces=pieces)

    def piece_at(self, square: int) -> tuple[Color, PieceType] | None:
        """Get the piece at a square, if any."""
        mask = 1 << square
        for color in Color:
            for piece_type in PieceType:
                if int(self.pieces[color, piece_type]) & mask:
                    return (color, piece_type)
        return None

    def set_piece(self, square: int, color: Color, piece_type: PieceType) -> None:
        """Place a piece on a square."""
        mask = np.uint64(1) << square
        self.pieces[color, piece_type] |= mask
        self._update_occupancy()

    def remove_piece(self, square: int, color: Color, piece_type: PieceType) -> None:
        """Remove a piece from a square."""
        mask = ~(np.uint64(1) << square)
        self.pieces[color, piece_type] &= mask
        self._update_occupancy()

    def move_piece(self, from_sq: int, to_sq: int, color: Color, piece_type: PieceType) -> None:
        """Move a piece from one square to another."""
        from_mask = np.uint64(1) << from_sq
        to_mask = np.uint64(1) << to_sq
        self.pieces[color, piece_type] ^= from_mask | to_mask
        self._update_occupancy()

    def get_attack_map(self, color: Color) -> int:
        """Get combined attack map for a color."""
        attacks = 0

        # Pawn attacks
        pawns = int(self.pieces[color, PieceType.PAWN])
        if color == Color.WHITE:
            attacks |= shift_up(shift_left(pawns))
            attacks |= shift_up(shift_right(pawns))
        else:
            attacks |= shift_down(shift_left(pawns))
            attacks |= shift_down(shift_right(pawns))

        # Knight attacks
        for sq in iter_bits(int(self.pieces[color, PieceType.KNIGHT])):
            attacks |= int(KNIGHT_ATTACKS[sq])

        # King attacks
        king_sq = lsb(int(self.pieces[color, PieceType.KING]))
        if king_sq >= 0:
            attacks |= int(KING_ATTACKS[king_sq])

        # Sliding piece attacks (simplified - full implementation uses magic bitboards)
        attacks |= self._get_sliding_attacks(color, PieceType.BISHOP)
        attacks |= self._get_sliding_attacks(color, PieceType.ROOK)
        attacks |= self._get_sliding_attacks(color, PieceType.QUEEN)

        return attacks

    def _get_sliding_attacks(self, color: Color, piece_type: PieceType) -> int:
        """Get sliding piece attacks (simplified ray-based implementation)."""
        attacks = 0
        pieces = int(self.pieces[color, piece_type])

        # Direction offsets
        if piece_type == PieceType.BISHOP:
            directions = [7, 9, -7, -9]  # Diagonal
        elif piece_type == PieceType.ROOK:
            directions = [1, -1, 8, -8]  # Orthogonal
        else:  # Queen
            directions = [1, -1, 8, -8, 7, 9, -7, -9]

        for sq in iter_bits(pieces):
            for direction in directions:
                attacks |= self._ray_attack(sq, direction)

        return attacks

    def _ray_attack(self, sq: int, direction: int) -> int:
        """Generate ray attacks from a square in a direction."""
        attacks = 0
        current = sq

        while True:
            prev_file = current % 8
            current += direction

            # Check bounds
            if current < 0 or current > 63:
                break

            # Check wrapping
            curr_file = current % 8
            if abs(curr_file - prev_file) > 1:
                break

            attacks |= 1 << current

            # Stop if we hit a piece
            if self.all_pieces & (1 << current):
                break

        return attacks

    def copy(self) -> BitBoard:
        """Create a deep copy of the bitboard."""
        return BitBoard(pieces=self.pieces.copy())

    def to_string(self) -> str:
        """Convert to ASCII board representation."""
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

        lines = []
        for rank in range(7, -1, -1):
            line = f"{rank + 1} "
            for file in range(8):
                sq = rank * 8 + file
                piece = self.piece_at(sq)
                if piece:
                    line += piece_chars[piece] + " "
                else:
                    line += ". "
            lines.append(line)
        lines.append("  a b c d e f g h")
        return "\n".join(lines)
