"""Sequence analysis utilities for proteins and nucleotides.

Provides functionality for:
- FASTA parsing
- Sequence alignment
- Protein property calculations
- Conservation analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class ProteinSequence:
    """Protein sequence with metadata.

    Attributes:
        id: Sequence identifier (e.g., UniProt ID)
        name: Protein name
        sequence: Amino acid sequence (single-letter code)
        organism: Source organism
        properties: Additional sequence properties
    """

    id: str
    name: str
    sequence: str
    organism: Optional[str] = None
    properties: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.sequence = self.sequence.upper().strip()

    def length(self) -> int:
        """Return sequence length."""

        return len(self.sequence)

    def validate(self) -> bool:
        """Validate sequence contains only valid amino acids."""

        valid_aa = set("ACDEFGHIKLMNPQRSTVWY")
        return all(aa in valid_aa for aa in self.sequence)


class SequenceAnalyzer:
    """Sequence analysis and property calculation.

    Provides methods for analyzing protein sequences, computing
    biophysical properties, and performing sequence comparisons.
    """

    # Amino acid properties
    HYDROPHOBICITY = {
        "A": 1.8,
        "C": 2.5,
        "D": -3.5,
        "E": -3.5,
        "F": 2.8,
        "G": -0.4,
        "H": -3.2,
        "I": 4.5,
        "K": -3.9,
        "L": 3.8,
        "M": 1.9,
        "N": -3.5,
        "P": -1.6,
        "Q": -3.5,
        "R": -4.5,
        "S": -0.8,
        "T": -0.7,
        "V": 4.2,
        "W": -0.9,
        "Y": -1.3,
    }

    MOLECULAR_WEIGHT = {
        "A": 89.1,
        "C": 121.2,
        "D": 133.1,
        "E": 147.1,
        "F": 165.2,
        "G": 75.1,
        "H": 155.2,
        "I": 131.2,
        "K": 146.2,
        "L": 131.2,
        "M": 149.2,
        "N": 132.1,
        "P": 115.1,
        "Q": 146.2,
        "R": 174.2,
        "S": 105.1,
        "T": 119.1,
        "V": 117.1,
        "W": 204.2,
        "Y": 181.2,
    }

    def __init__(self):
        """Initialize sequence analyzer."""

        self._sequences: dict[str, ProteinSequence] = {}

    def parse_fasta(self, fasta_content: str) -> list[ProteinSequence]:
        """Parse FASTA format sequences.

        Args:
            fasta_content: FASTA format string

        Returns:
            List of ProteinSequence objects
        """

        sequences = []
        current_id = None
        current_name = None
        current_seq = []

        for line in fasta_content.strip().split("\n"):
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                # Save previous sequence
                if current_id is not None:
                    seq = ProteinSequence(
                        id=current_id,
                        name=current_name or current_id,
                        sequence="".join(current_seq),
                    )
                    sequences.append(seq)
                    self._sequences[current_id] = seq

                # Parse header
                header = line[1:].strip()
                parts = header.split("|")
                if len(parts) >= 2:
                    current_id = parts[1]
                    current_name = parts[2] if len(parts) > 2 else parts[1]
                else:
                    current_id = parts[0]
                    current_name = parts[0]
                current_seq = []
            else:
                current_seq.append(line)

        # Save last sequence
        if current_id is not None:
            seq = ProteinSequence(
                id=current_id, name=current_name or current_id, sequence="".join(current_seq)
            )
            sequences.append(seq)
            self._sequences[current_id] = seq

        return sequences

    def compute_molecular_weight(self, sequence: str) -> float:
        """Compute molecular weight of protein sequence.

        Args:
            sequence: Amino acid sequence

        Returns:
            Molecular weight in Daltons
        """

        weight = sum(self.MOLECULAR_WEIGHT.get(aa, 0.0) for aa in sequence.upper())
        # Subtract water molecules for peptide bonds
        weight -= (len(sequence) - 1) * 18.0
        return weight

    def compute_hydrophobicity(self, sequence: str) -> float:
        """Compute average hydrophobicity of sequence.

        Uses Kyte-Doolittle hydrophobicity scale.

        Args:
            sequence: Amino acid sequence

        Returns:
            Average hydrophobicity score
        """

        scores = [self.HYDROPHOBICITY.get(aa, 0.0) for aa in sequence.upper()]
        return float(np.mean(scores)) if scores else 0.0

    def compute_isoelectric_point(self, sequence: str) -> float:
        """Compute theoretical isoelectric point (pI).

        Simplified calculation using Henderson-Hasselbalch equation.

        Args:
            sequence: Amino acid sequence

        Returns:
            Estimated pI value
        """

        # Count charged residues
        n_term = 1
        c_term = 1
        k_count = sequence.upper().count("K")
        r_count = sequence.upper().count("R")
        h_count = sequence.upper().count("H")
        d_count = sequence.upper().count("D")
        e_count = sequence.upper().count("E")
        c_count = sequence.upper().count("C")
        y_count = sequence.upper().count("Y")

        # Simplified pI calculation (average of basic and acidic pKa)
        positive = n_term + k_count + r_count + 0.5 * h_count
        negative = c_term + d_count + e_count + 0.1 * c_count + 0.1 * y_count

        if positive > negative:
            return 8.0 + (positive - negative) / positive
        elif negative > positive:
            return 6.0 - (negative - positive) / negative
        else:
            return 7.0

    def compute_composition(self, sequence: str) -> dict[str, float]:
        """Compute amino acid composition percentages.

        Args:
            sequence: Amino acid sequence

        Returns:
            Dictionary of amino acid frequencies
        """

        sequence = sequence.upper()
        length = len(sequence)
        composition = {}

        for aa in "ACDEFGHIKLMNPQRSTVWY":
            count = sequence.count(aa)
            composition[aa] = (count / length * 100) if length > 0 else 0.0

        return composition

    def align_sequences(
        self,
        seq1: str,
        seq2: str,
        match_score: float = 2.0,
        mismatch_penalty: float = -1.0,
        gap_penalty: float = -2.0,
    ) -> tuple[str, str, float]:
        """Perform pairwise sequence alignment (Needleman-Wunsch).

        Args:
            seq1: First sequence
            seq2: Second sequence
            match_score: Score for matching residues
            mismatch_penalty: Penalty for mismatches
            gap_penalty: Penalty for gaps

        Returns:
            Tuple of (aligned_seq1, aligned_seq2, alignment_score)
        """

        seq1 = seq1.upper()
        seq2 = seq2.upper()
        m, n = len(seq1), len(seq2)

        # Initialize scoring matrix
        score_matrix = np.zeros((m + 1, n + 1))

        # Initialize first row and column
        for i in range(m + 1):
            score_matrix[i][0] = i * gap_penalty
        for j in range(n + 1):
            score_matrix[0][j] = j * gap_penalty

        # Fill scoring matrix
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                match = score_matrix[i - 1][j - 1] + (
                    match_score if seq1[i - 1] == seq2[j - 1] else mismatch_penalty
                )
                delete = score_matrix[i - 1][j] + gap_penalty
                insert = score_matrix[i][j - 1] + gap_penalty
                score_matrix[i][j] = max(match, delete, insert)

        # Traceback
        aligned1, aligned2 = [], []
        i, j = m, n

        while i > 0 or j > 0:
            if i > 0 and j > 0:
                current_score = score_matrix[i][j]
                diag_score = score_matrix[i - 1][j - 1]

                if seq1[i - 1] == seq2[j - 1]:
                    score_diff = match_score
                else:
                    score_diff = mismatch_penalty

                if abs(current_score - (diag_score + score_diff)) < 1e-6:
                    aligned1.append(seq1[i - 1])
                    aligned2.append(seq2[j - 1])
                    i -= 1
                    j -= 1
                    continue

            if i > 0 and abs(score_matrix[i][j] - (score_matrix[i - 1][j] + gap_penalty)) < 1e-6:
                aligned1.append(seq1[i - 1])
                aligned2.append("-")
                i -= 1
            elif j > 0:
                aligned1.append("-")
                aligned2.append(seq2[j - 1])
                j -= 1
            else:
                break

        aligned1 = "".join(reversed(aligned1))
        aligned2 = "".join(reversed(aligned2))
        final_score = float(score_matrix[m][n])

        return aligned1, aligned2, final_score

    def compute_conservation_score(self, sequences: list[str]) -> np.ndarray:
        """Compute conservation score for multiple sequence alignment.

        Args:
            sequences: List of aligned sequences (same length)

        Returns:
            Array of conservation scores (0-1) for each position
        """

        if not sequences:
            return np.array([])

        length = len(sequences[0])
        conservation = np.zeros(length)

        for pos in range(length):
            # Count amino acids at this position
            aa_counts: dict[str, int] = {}
            total = 0

            for seq in sequences:
                if pos < len(seq):
                    aa = seq[pos].upper()
                    if aa != "-":
                        aa_counts[aa] = aa_counts.get(aa, 0) + 1
                        total += 1

            if total > 0:
                # Shannon entropy-based conservation
                entropy = 0.0
                for count in aa_counts.values():
                    freq = count / total
                    entropy -= freq * np.log2(freq + 1e-10)

                # Normalize by maximum entropy (log2(20) for 20 amino acids)
                max_entropy = np.log2(20)
                conservation[pos] = 1.0 - (entropy / max_entropy)

        return conservation

    def find_motifs(self, sequence: str, motif: str) -> list[int]:
        """Find all occurrences of a motif in sequence.

        Args:
            sequence: Protein sequence to search
            motif: Motif pattern to find

        Returns:
            List of starting positions (0-indexed)
        """

        sequence = sequence.upper()
        motif = motif.upper()
        positions = []

        for i in range(len(sequence) - len(motif) + 1):
            if sequence[i : i + len(motif)] == motif:
                positions.append(i)

        return positions

    def compute_similarity(self, seq1: str, seq2: str) -> float:
        """Compute sequence similarity percentage.

        Args:
            seq1: First sequence
            seq2: Second sequence

        Returns:
            Similarity percentage (0-100)
        """

        min_len = min(len(seq1), len(seq2))
        if min_len == 0:
            return 0.0

        matches = sum(1 for i in range(min_len) if seq1[i].upper() == seq2[i].upper())
        return (matches / min_len) * 100.0

    def get_sequence(self, seq_id: str) -> Optional[ProteinSequence]:
        """Retrieve a parsed sequence by ID.

        Args:
            seq_id: Sequence identifier

        Returns:
            ProteinSequence if found, None otherwise
        """

        return self._sequences.get(seq_id)
