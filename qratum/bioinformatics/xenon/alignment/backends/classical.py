"""
Classical Alignment Backend

Bit-identical classical sequence alignment (legacy preserved).
Certificate: QRATUM-HARDENING-20251215-V5
"""

from typing import Dict

import numpy as np


class ClassicalAlignmentBackend:
    """
    Classical sequence alignment using Smith-Waterman algorithm.

    Provides bit-identical results to legacy implementations.
    """

    def __init__(self, match_score: int = 2, mismatch_penalty: int = -1, gap_penalty: int = -1):
        """
        Initialize classical alignment backend.

        Args:
            match_score: Score for matching characters
            mismatch_penalty: Penalty for mismatching characters
            gap_penalty: Penalty for gaps
        """
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

    def align(self, sequence1: str, sequence2: str) -> Dict:
        """
        Perform local sequence alignment using Smith-Waterman.

        Args:
            sequence1: First sequence
            sequence2: Second sequence

        Returns:
            Dictionary with alignment results
        """
        m, n = len(sequence1), len(sequence2)

        # Initialize scoring matrix
        score_matrix = np.zeros((m + 1, n + 1), dtype=np.int32)

        # Fill scoring matrix
        max_score = 0
        max_pos = (0, 0)

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                # Calculate match/mismatch score
                match = score_matrix[i - 1, j - 1] + (
                    self.match_score
                    if sequence1[i - 1] == sequence2[j - 1]
                    else self.mismatch_penalty
                )

                # Calculate gap scores
                delete = score_matrix[i - 1, j] + self.gap_penalty
                insert = score_matrix[i, j - 1] + self.gap_penalty

                # Take maximum (Smith-Waterman local alignment)
                score_matrix[i, j] = max(0, match, delete, insert)

                # Track maximum score
                if score_matrix[i, j] > max_score:
                    max_score = score_matrix[i, j]
                    max_pos = (i, j)

        # Traceback to get alignment
        aligned1, aligned2 = self._traceback(score_matrix, sequence1, sequence2, max_pos)

        return {
            "score": int(max_score),
            "aligned_sequence1": aligned1,
            "aligned_sequence2": aligned2,
            "length": len(aligned1),
            "identity": self._compute_identity(aligned1, aligned2),
            "score_matrix_shape": score_matrix.shape,
        }

    def _traceback(
        self, score_matrix: np.ndarray, sequence1: str, sequence2: str, max_pos: tuple
    ) -> tuple:
        """
        Traceback to recover alignment.

        Args:
            score_matrix: Filled scoring matrix
            sequence1: First sequence
            sequence2: Second sequence
            max_pos: Position of maximum score

        Returns:
            Tuple of aligned sequences
        """
        aligned1 = []
        aligned2 = []

        i, j = max_pos

        while i > 0 and j > 0 and score_matrix[i, j] > 0:
            current_score = score_matrix[i, j]
            diagonal = score_matrix[i - 1, j - 1]
            up = score_matrix[i - 1, j]
            left = score_matrix[i, j - 1]

            # Determine which direction we came from
            if sequence1[i - 1] == sequence2[j - 1]:
                match_score = diagonal + self.match_score
            else:
                match_score = diagonal + self.mismatch_penalty

            if current_score == match_score:
                aligned1.insert(0, sequence1[i - 1])
                aligned2.insert(0, sequence2[j - 1])
                i -= 1
                j -= 1
            elif current_score == up + self.gap_penalty:
                aligned1.insert(0, sequence1[i - 1])
                aligned2.insert(0, "-")
                i -= 1
            elif current_score == left + self.gap_penalty:
                aligned1.insert(0, "-")
                aligned2.insert(0, sequence2[j - 1])
                j -= 1
            else:
                break

        return "".join(aligned1), "".join(aligned2)

    def _compute_identity(self, aligned1: str, aligned2: str) -> float:
        """
        Compute sequence identity percentage.

        Args:
            aligned1: First aligned sequence
            aligned2: Second aligned sequence

        Returns:
            Identity percentage (0-1)
        """
        if len(aligned1) == 0:
            return 0.0

        matches = sum(1 for a, b in zip(aligned1, aligned2) if a == b and a != "-")
        return float(matches) / len(aligned1)
