"""Tests for the PDB loader."""

from __future__ import annotations

import tempfile
from pathlib import Path

from xenon.molecular_dynamics_lab.core.pdb_loader import (Atom, Chain,
                                                          PDBLoader, Residue)

# Sample PDB content for testing
SAMPLE_PDB = """HEADER    PLANT PROTEIN                           15-APR-81   1CRN
TITLE     WATER STRUCTURE OF A HYDROPHOBIC PROTEIN AT ATOMIC RESOLUTION
ATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79           N
ATOM      2  CA  THR A   1      16.967  12.784   4.338  1.00 10.80           C
ATOM      3  C   THR A   1      15.685  12.755   5.133  1.00  9.19           C
ATOM      4  O   THR A   1      15.268  13.825   5.594  1.00  9.85           O
ATOM      5  CB  THR A   1      18.170  12.703   5.337  1.00 13.02           C
ATOM      6  OG1 THR A   1      19.334  12.829   4.463  1.00 15.06           O
ATOM      7  CG2 THR A   1      18.150  11.546   6.304  1.00 14.23           C
ATOM      8  N   THR A   2      15.115  11.545   5.265  1.00  7.81           N
ATOM      9  CA  THR A   2      13.856  11.469   6.066  1.00  8.31           C
ATOM     10  C   THR A   2      14.164  10.785   7.379  1.00  5.80           C
HETATM  328  O   HOH A 101       4.926  15.178   4.843  1.00 18.41           O
HETATM  329  O   HOH A 102       7.664  13.613   9.825  1.00 22.86           O
CONECT    1    2
CONECT    2    1    3    5
END
"""


class TestPDBLoader:
    """Test PDB loader functionality."""

    def test_load_from_string(self):
        """Test loading PDB from string."""
        loader = PDBLoader()
        structure = loader.load_from_string(SAMPLE_PDB, pdb_id="TEST")

        assert structure is not None
        assert structure.pdb_id == "TEST"
        assert len(structure.atoms) == 10
        assert len(structure.hetatms) == 2
        assert len(structure.chains) == 1

    def test_parse_atom_record(self):
        """Test parsing individual ATOM records."""
        loader = PDBLoader()
        atom_line = "ATOM      1  N   THR A   1      17.047  14.099   3.625  1.00 13.79           N"

        atom = loader._parse_atom_record(atom_line)

        assert atom is not None
        assert atom.serial == 1
        assert atom.name == "N"
        assert atom.residue_name == "THR"
        assert atom.chain_id == "A"
        assert atom.residue_seq == 1
        assert abs(atom.x - 17.047) < 0.001
        assert abs(atom.y - 14.099) < 0.001
        assert abs(atom.z - 3.625) < 0.001
        assert atom.element == "N"

    def test_chain_organization(self):
        """Test that atoms are organized into chains."""
        loader = PDBLoader()
        structure = loader.load_from_string(SAMPLE_PDB, pdb_id="TEST")

        assert len(structure.chains) == 1
        chain_a = structure.chains[0]
        assert chain_a.chain_id == "A"
        assert len(chain_a.residues) == 2  # THR 1 and THR 2

    def test_residue_organization(self):
        """Test that atoms are organized into residues."""
        loader = PDBLoader()
        structure = loader.load_from_string(SAMPLE_PDB, pdb_id="TEST")

        chain_a = structure.chains[0]
        residue_1 = chain_a.residues[0]

        assert residue_1.name == "THR"
        assert residue_1.sequence == 1
        assert len(residue_1.atoms) == 7  # N, CA, C, O, CB, OG1, CG2

    def test_hetatm_parsing(self):
        """Test parsing of HETATM records."""
        loader = PDBLoader()
        structure = loader.load_from_string(SAMPLE_PDB, pdb_id="TEST")

        assert len(structure.hetatms) == 2
        water = structure.hetatms[0]
        assert water.name == "O"
        assert water.residue_name == "HOH"

    def test_conect_parsing(self):
        """Test parsing of CONECT records."""
        loader = PDBLoader()
        structure = loader.load_from_string(SAMPLE_PDB, pdb_id="TEST")

        assert len(structure.bonds) >= 2
        # Check that bonds were created
        bond_pairs = [(b.atom1, b.atom2) for b in structure.bonds]
        assert (1, 2) in bond_pairs or (2, 1) in bond_pairs

    def test_load_from_file(self):
        """Test loading PDB from file."""
        loader = PDBLoader()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".pdb", delete=False) as f:
            f.write(SAMPLE_PDB)
            temp_path = Path(f.name)

        try:
            structure = loader.load_from_file(temp_path)
            assert structure is not None
            assert len(structure.atoms) == 10
        finally:
            temp_path.unlink()

    def test_to_3dmol_format(self):
        """Test conversion to 3Dmol.js format."""
        loader = PDBLoader()
        structure = loader.load_from_string(SAMPLE_PDB, pdb_id="TEST")

        mol_data = structure.to_3dmol_format()

        assert "atoms" in mol_data
        assert len(mol_data["atoms"]) == 10

        # Check atom format
        first_atom = mol_data["atoms"][0]
        assert "elem" in first_atom
        assert "x" in first_atom
        assert "y" in first_atom
        assert "z" in first_atom
        assert "resn" in first_atom
        assert "chain" in first_atom

    def test_to_dict(self):
        """Test structure serialization to dictionary."""
        loader = PDBLoader()
        structure = loader.load_from_string(SAMPLE_PDB, pdb_id="TEST")

        data = structure.to_dict()

        assert data["pdb_id"] == "TEST"
        assert data["num_atoms"] == 10
        assert "raw_content" in data

    def test_invalid_pdb(self):
        """Test handling of invalid PDB content."""
        loader = PDBLoader()

        # Empty string
        structure = loader.load_from_string("", pdb_id="EMPTY")
        assert len(structure.atoms) == 0

        # Invalid content
        structure = loader.load_from_string("Not a PDB file", pdb_id="INVALID")
        assert len(structure.atoms) == 0

    def test_header_parsing(self):
        """Test parsing of HEADER record."""
        loader = PDBLoader()
        structure = loader.load_from_string(SAMPLE_PDB, pdb_id="1CRN")

        assert "classification" in structure.header
        assert "date" in structure.header


class TestAtom:
    """Test Atom dataclass."""

    def test_atom_creation(self):
        """Test creating an Atom."""
        atom = Atom(
            serial=1,
            name="CA",
            residue_name="ALA",
            chain_id="A",
            residue_seq=1,
            x=1.0,
            y=2.0,
            z=3.0,
            occupancy=1.0,
            temp_factor=10.0,
            element="C",
        )

        assert atom.serial == 1
        assert atom.name == "CA"
        assert atom.element == "C"

    def test_atom_position(self):
        """Test atom position as tuple."""
        atom = Atom(
            serial=1,
            name="CA",
            residue_name="ALA",
            chain_id="A",
            residue_seq=1,
            x=1.5,
            y=2.5,
            z=3.5,
            occupancy=1.0,
            temp_factor=10.0,
            element="C",
        )

        pos = (atom.x, atom.y, atom.z)
        assert pos == (1.5, 2.5, 3.5)


class TestResidue:
    """Test Residue dataclass."""

    def test_residue_creation(self):
        """Test creating a Residue."""
        residue = Residue(
            name="ALA",
            sequence=1,
            insertion_code="",
            chain_id="A",
            atoms=[],
        )

        assert residue.name == "ALA"
        assert residue.sequence == 1

    def test_residue_with_atoms(self):
        """Test residue with atoms."""
        atoms = [
            Atom(1, "N", "ALA", "A", 1, 0, 0, 0, 1.0, 10.0, "N"),
            Atom(2, "CA", "ALA", "A", 1, 1, 0, 0, 1.0, 10.0, "C"),
            Atom(3, "C", "ALA", "A", 1, 2, 0, 0, 1.0, 10.0, "C"),
        ]
        residue = Residue("ALA", 1, "", "A", atoms)

        assert len(residue.atoms) == 3


class TestChain:
    """Test Chain dataclass."""

    def test_chain_creation(self):
        """Test creating a Chain."""
        chain = Chain(chain_id="A", residues=[])

        assert chain.chain_id == "A"
        assert len(chain.residues) == 0
