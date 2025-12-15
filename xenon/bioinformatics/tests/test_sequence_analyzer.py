"""Tests for sequence analyzer."""

import pytest
import numpy as np

from xenon.bioinformatics.sequence_analyzer import (
    ProteinSequence,
    SequenceAnalyzer,
)


class TestProteinSequence:
    """Test ProteinSequence class."""
    
    def test_create_sequence(self):
        """Test sequence creation."""
        seq = ProteinSequence(
            id="P12345",
            name="Test Protein",
            sequence="ACDEFGHIKLMNPQRSTVWY"
        )
        
        assert seq.id == "P12345"
        assert seq.name == "Test Protein"
        assert seq.length() == 20
    
    def test_sequence_validation(self):
        """Test sequence validation."""
        valid_seq = ProteinSequence(id="1", name="Valid", sequence="ACDEFGHIKLM")
        assert valid_seq.validate()
        
        invalid_seq = ProteinSequence(id="2", name="Invalid", sequence="ACXYZ")
        assert not invalid_seq.validate()


class TestSequenceAnalyzer:
    """Test SequenceAnalyzer class."""
    
    def test_parse_fasta(self):
        """Test FASTA parsing."""
        fasta_content = """>sp|P12345|PROT1 Protein 1
ACDEFGHIKLMNPQRSTVWY
>sp|P67890|PROT2 Protein 2
WYVSRTQPNMLKIHGFEDCA"""
        
        analyzer = SequenceAnalyzer()
        sequences = analyzer.parse_fasta(fasta_content)
        
        assert len(sequences) == 2
        assert sequences[0].id == "P12345"
        assert sequences[0].name == "PROT1 Protein 1"
        assert sequences[0].length() == 20
        assert sequences[1].id == "P67890"
    
    def test_molecular_weight(self):
        """Test molecular weight calculation."""
        analyzer = SequenceAnalyzer()
        
        # Simple sequence
        mw = analyzer.compute_molecular_weight("ALA")
        assert mw > 0
        
        # Known value: Glycine
        mw_gly = analyzer.compute_molecular_weight("G")
        assert abs(mw_gly - 75.1) < 1.0
    
    def test_hydrophobicity(self):
        """Test hydrophobicity calculation."""
        analyzer = SequenceAnalyzer()
        
        # Hydrophobic sequence
        hydro_seq = "IIIIIVVVVV"
        hydro = analyzer.compute_hydrophobicity(hydro_seq)
        assert hydro > 3.0
        
        # Hydrophilic sequence
        hydrophilic_seq = "DDDDEEEEEE"
        hydrophilic = analyzer.compute_hydrophobicity(hydrophilic_seq)
        assert hydrophilic < 0.0
    
    def test_isoelectric_point(self):
        """Test pI calculation."""
        analyzer = SequenceAnalyzer()
        
        # Basic sequence (lysine-rich)
        basic_seq = "KKKKKRRRRR"
        pi_basic = analyzer.compute_isoelectric_point(basic_seq)
        assert pi_basic > 7.0
        
        # Acidic sequence (aspartate-rich)
        acidic_seq = "DDDDDEEEEE"
        pi_acidic = analyzer.compute_isoelectric_point(acidic_seq)
        assert pi_acidic < 7.0
    
    def test_composition(self):
        """Test amino acid composition."""
        analyzer = SequenceAnalyzer()
        
        sequence = "AAACCCGGGG"
        composition = analyzer.compute_composition(sequence)
        
        assert abs(composition['A'] - 30.0) < 0.1
        assert abs(composition['C'] - 30.0) < 0.1
        assert abs(composition['G'] - 40.0) < 0.1
        assert abs(composition['D'] - 0.0) < 0.1
    
    def test_sequence_alignment(self):
        """Test pairwise alignment."""
        analyzer = SequenceAnalyzer()
        
        seq1 = "ACDEFGH"
        seq2 = "ACDXFGH"
        
        aligned1, aligned2, score = analyzer.align_sequences(seq1, seq2)
        
        assert len(aligned1) == len(aligned2)
        assert score > 0
        # Should align mostly, with one mismatch
        assert aligned1.replace('-', '').startswith('ACD')
    
    def test_conservation_score(self):
        """Test conservation score calculation."""
        analyzer = SequenceAnalyzer()
        
        # Perfectly conserved sequences
        sequences = ["ACDEFGH", "ACDEFGH", "ACDEFGH"]
        conservation = analyzer.compute_conservation_score(sequences)
        
        assert len(conservation) == 7
        assert all(score > 0.9 for score in conservation)
        
        # Variable sequences
        variable_seqs = ["AAAAAAA", "CCCCCCC", "GGGGGGG"]
        var_conservation = analyzer.compute_conservation_score(variable_seqs)
        assert all(score < 0.5 for score in var_conservation)
    
    def test_find_motifs(self):
        """Test motif finding."""
        analyzer = SequenceAnalyzer()
        
        sequence = "ACDEFGHIKLMNPQRSTVWYACDEFGH"
        motif = "DEF"
        
        positions = analyzer.find_motifs(sequence, motif)
        
        assert len(positions) == 2
        assert 2 in positions  # First occurrence at position 2
        assert 22 in positions  # Second occurrence
    
    def test_compute_similarity(self):
        """Test sequence similarity."""
        analyzer = SequenceAnalyzer()
        
        # Identical sequences
        similarity = analyzer.compute_similarity("ACDEFGH", "ACDEFGH")
        assert abs(similarity - 100.0) < 0.1
        
        # Different sequences
        similarity = analyzer.compute_similarity("AAAAAAA", "CCCCCCC")
        assert abs(similarity - 0.0) < 0.1
        
        # Partially similar
        similarity = analyzer.compute_similarity("ACDEFGH", "ACDXFGH")
        assert 70.0 < similarity < 90.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
