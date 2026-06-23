"""PDB (Protein Data Bank) file loader and parser.

Provides functionality for loading, parsing, and manipulating PDB files
from both local files and the RCSB PDB database.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Atom:
    """Atomic coordinate and metadata."""

    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_seq: int
    x: float
    y: float
    z: float
    occupancy: float = 1.0
    temp_factor: float = 0.0
    element: str = ""
    alt_loc: str = ""
    charge: str = ""

    @property
    def res_name(self) -> str:
        """Backward-compatible alias for :attr:`residue_name`."""
        return self.residue_name

    @property
    def res_seq(self) -> int:
        """Backward-compatible alias for :attr:`residue_seq`."""
        return self.residue_seq

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "serial": self.serial,
            "name": self.name,
            "alt_loc": self.alt_loc,
            "res_name": self.residue_name,
            "chain_id": self.chain_id,
            "res_seq": self.residue_seq,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "occupancy": self.occupancy,
            "temp_factor": self.temp_factor,
            "element": self.element,
            "charge": self.charge,
        }

    @property
    def coords(self) -> np.ndarray:
        """Return coordinates as numpy array."""
        return np.array([self.x, self.y, self.z])


@dataclass
class Residue:
    """Amino acid residue."""

    name: str
    sequence: int
    insertion_code: str = ""
    chain_id: str = ""
    atoms: list[Atom] = field(default_factory=list)

    @property
    def seq_num(self) -> int:
        """Backward-compatible alias for :attr:`sequence`."""
        return self.sequence

    def get_atom(self, name: str) -> Optional[Atom]:
        """Get atom by name."""
        for atom in self.atoms:
            if atom.name == name:
                return atom
        return None

    @property
    def alpha_carbon(self) -> Optional[Atom]:
        """Get alpha carbon (CA) atom."""
        return self.get_atom("CA")

    @property
    def center_of_mass(self) -> np.ndarray:
        """Calculate center of mass."""
        if not self.atoms:
            return np.zeros(3)
        coords = np.array([a.coords for a in self.atoms])
        return np.mean(coords, axis=0)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "chain_id": self.chain_id,
            "seq_num": self.seq_num,
            "atoms": [a.to_dict() for a in self.atoms],
        }


@dataclass
class Chain:
    """Protein chain."""

    chain_id: str
    residues: list[Residue] = field(default_factory=list)

    def get_residue(self, seq_num: int) -> Optional[Residue]:
        """Get residue by sequence number."""
        for res in self.residues:
            if res.seq_num == seq_num:
                return res
        return None

    @property
    def sequence(self) -> str:
        """Get one-letter amino acid sequence."""
        aa_map = {
            "ALA": "A",
            "ARG": "R",
            "ASN": "N",
            "ASP": "D",
            "CYS": "C",
            "GLN": "Q",
            "GLU": "E",
            "GLY": "G",
            "HIS": "H",
            "ILE": "I",
            "LEU": "L",
            "LYS": "K",
            "MET": "M",
            "PHE": "F",
            "PRO": "P",
            "SER": "S",
            "THR": "T",
            "TRP": "W",
            "TYR": "Y",
            "VAL": "V",
        }
        return "".join(aa_map.get(r.name, "X") for r in self.residues)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "chain_id": self.chain_id,
            "residues": [r.to_dict() for r in self.residues],
            "sequence": self.sequence,
        }


@dataclass
class Hetatm:
    """Hetero atom (ligand, water, ion)."""

    serial: int
    name: str
    residue_name: str
    chain_id: str
    residue_seq: int
    x: float
    y: float
    z: float
    element: str = ""

    @property
    def res_name(self) -> str:
        """Backward-compatible alias for :attr:`residue_name`."""
        return self.residue_name

    @property
    def res_seq(self) -> int:
        """Backward-compatible alias for :attr:`residue_seq`."""
        return self.residue_seq

    @property
    def coords(self) -> np.ndarray:
        """Return coordinates as numpy array."""
        return np.array([self.x, self.y, self.z])

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "serial": self.serial,
            "name": self.name,
            "res_name": self.res_name,
            "chain_id": self.chain_id,
            "res_seq": self.res_seq,
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "element": self.element,
        }


@dataclass
class Bond:
    """Covalent bond between atoms."""

    atom1_serial: int
    atom2_serial: int
    bond_order: int = 1

    @property
    def atom1(self) -> int:
        """Backward-compatible alias for :attr:`atom1_serial`."""
        return self.atom1_serial

    @property
    def atom2(self) -> int:
        """Backward-compatible alias for :attr:`atom2_serial`."""
        return self.atom2_serial

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "atom1": self.atom1_serial,
            "atom2": self.atom2_serial,
            "order": self.bond_order,
        }


@dataclass
class PDBStructure:
    """Complete PDB structure."""

    pdb_id: str
    title: str = ""
    authors: str = ""
    resolution: float = 0.0
    method: str = ""
    chains: list[Chain] = field(default_factory=list)
    hetatms: list[Hetatm] = field(default_factory=list)
    bonds: list[Bond] = field(default_factory=list)
    header: dict = field(default_factory=dict)
    raw_content: str = ""

    @property
    def all_atoms(self) -> list[Atom]:
        """Get all atoms from all chains."""
        atoms = []
        for chain in self.chains:
            for residue in chain.residues:
                atoms.extend(residue.atoms)
        return atoms

    @property
    def atoms(self) -> list[Atom]:
        """Alias for :attr:`all_atoms` (flat list of polymer atoms)."""
        return self.all_atoms

    def get_chain(self, chain_id: str) -> Optional[Chain]:
        """Return the chain with the given identifier, or None."""
        for chain in self.chains:
            if chain.chain_id == chain_id:
                return chain
        return None

    def to_3dmol_format(self) -> dict:
        """3Dmol.js view containing the polymer (ATOM) atoms only.

        Hetero atoms are available separately via :attr:`hetatms`; use
        :meth:`to_3dmol_model` for a combined polymer + hetero atom list.
        """
        atoms = [
            {
                "elem": atom.element or atom.name[0],
                "x": atom.x,
                "y": atom.y,
                "z": atom.z,
                "serial": atom.serial,
                "atom": atom.name,
                "resn": atom.residue_name,
                "chain": atom.chain_id,
                "resi": atom.residue_seq,
                "b": atom.temp_factor,
            }
            for atom in self.all_atoms
        ]
        return {"atoms": atoms}

    @property
    def num_atoms(self) -> int:
        """Total number of atoms."""
        return len(self.all_atoms)

    @property
    def num_residues(self) -> int:
        """Total number of residues."""
        return sum(len(c.residues) for c in self.chains)

    @property
    def bounding_box(self) -> tuple[np.ndarray, np.ndarray]:
        """Calculate bounding box (min, max)."""
        coords = [a.coords for a in self.all_atoms]
        if not coords:
            return np.zeros(3), np.zeros(3)
        coords_arr = np.array(coords)
        return coords_arr.min(axis=0), coords_arr.max(axis=0)

    @property
    def center(self) -> np.ndarray:
        """Calculate geometric center."""
        bbox_min, bbox_max = self.bounding_box
        return (bbox_min + bbox_max) / 2

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pdb_id": self.pdb_id,
            "title": self.title,
            "authors": self.authors,
            "resolution": self.resolution,
            "method": self.method,
            "chains": [c.to_dict() for c in self.chains],
            "hetatms": [h.to_dict() for h in self.hetatms],
            "bonds": [b.to_dict() for b in self.bonds],
            "header": self.header,
            "raw_content": self.raw_content,
            "num_atoms": self.num_atoms,
            "num_residues": self.num_residues,
        }

    def to_3dmol_model(self) -> dict:
        """Convert to 3Dmol.js compatible format."""
        atoms = []
        for atom in self.all_atoms:
            atoms.append(
                {
                    "elem": atom.element or atom.name[0],
                    "x": atom.x,
                    "y": atom.y,
                    "z": atom.z,
                    "serial": atom.serial,
                    "atom": atom.name,
                    "resn": atom.res_name,
                    "chain": atom.chain_id,
                    "resi": atom.res_seq,
                    "b": atom.temp_factor,
                }
            )

        for het in self.hetatms:
            atoms.append(
                {
                    "elem": het.element or het.name[0],
                    "x": het.x,
                    "y": het.y,
                    "z": het.z,
                    "serial": het.serial,
                    "atom": het.name,
                    "resn": het.res_name,
                    "chain": het.chain_id,
                    "resi": het.res_seq,
                    "hetflag": True,
                }
            )

        return {"atoms": atoms}


class PDBLoader:
    """PDB file loader with support for local files and RCSB database."""

    RCSB_BASE_URL = "https://files.rcsb.org/download"

    def __init__(self, cache_dir: Optional[Path] = None):
        """Initialize PDB loader.

        Args:
            cache_dir: Directory for caching downloaded PDB files
        """
        self.cache_dir = cache_dir
        if cache_dir:
            cache_dir.mkdir(parents=True, exist_ok=True)
        self._structures: dict[str, PDBStructure] = {}

    def load_from_file(self, filepath: Path) -> PDBStructure:
        """Load PDB structure from local file.

        Args:
            filepath: Path to PDB file

        Returns:
            Parsed PDBStructure
        """
        logger.info(f"Loading PDB from file: {filepath}")
        with open(filepath) as f:
            content = f.read()

        pdb_id = filepath.stem.upper()
        return self._parse_pdb(content, pdb_id)

    def load_from_rcsb(self, pdb_id: str) -> PDBStructure:
        """Load PDB structure from RCSB database.

        Args:
            pdb_id: 4-character PDB identifier

        Returns:
            Parsed PDBStructure

        Raises:
            ValueError: If PDB ID is invalid or not found
        """
        pdb_id = pdb_id.upper().strip()
        if not re.match(r"^[0-9A-Z]{4}$", pdb_id):
            raise ValueError(f"Invalid PDB ID format: {pdb_id}")

        # Check cache
        if self.cache_dir:
            cache_file = self.cache_dir / f"{pdb_id}.pdb"
            if cache_file.exists():
                logger.info(f"Loading {pdb_id} from cache")
                return self.load_from_file(cache_file)

        # Download from RCSB
        url = f"{self.RCSB_BASE_URL}/{pdb_id}.pdb"
        logger.info(f"Downloading PDB {pdb_id} from RCSB")

        try:
            with urlopen(url, timeout=30) as response:
                content = response.read().decode("utf-8")
        except URLError as e:
            raise ValueError(f"Failed to download PDB {pdb_id}: {e}") from e

        # Cache the downloaded file
        if self.cache_dir:
            with open(self.cache_dir / f"{pdb_id}.pdb", "w") as f:
                f.write(content)

        return self._parse_pdb(content, pdb_id)

    def load_from_string(self, content: str, pdb_id: str = "UNKNOWN") -> PDBStructure:
        """Load PDB structure from string content.

        Args:
            content: PDB file content as string
            pdb_id: Optional PDB identifier

        Returns:
            Parsed PDBStructure
        """
        return self._parse_pdb(content, pdb_id)

    def _parse_pdb(self, content: str, pdb_id: str) -> PDBStructure:
        """Parse PDB content into structure.

        Args:
            content: Raw PDB file content
            pdb_id: PDB identifier

        Returns:
            Parsed PDBStructure
        """
        structure = PDBStructure(pdb_id=pdb_id, raw_content=content)
        residue_map: dict[str, dict[int, Residue]] = {}

        for line in content.splitlines():
            record_type = line[:6].strip()

            if record_type == "HEADER":
                structure.pdb_id = pdb_id or line[62:66].strip()
                structure.header = {
                    "classification": line[10:50].strip(),
                    "date": line[50:59].strip(),
                    "id_code": line[62:66].strip(),
                }

            elif record_type == "TITLE":
                structure.title += line[10:80].strip() + " "

            elif record_type == "AUTHOR":
                structure.authors += line[10:80].strip() + " "

            elif record_type == "EXPDTA":
                structure.method = line[10:80].strip()

            elif record_type == "REMARK":
                if line[7:10].strip() == "2":
                    # Resolution
                    match = re.search(r"RESOLUTION\.\s+([\d.]+)", line)
                    if match:
                        try:
                            structure.resolution = float(match.group(1))
                        except ValueError:
                            pass

            elif record_type == "ATOM":
                atom = self._parse_atom_line(line)
                if atom:
                    chain_id = atom.chain_id
                    res_seq = atom.res_seq

                    if chain_id not in residue_map:
                        residue_map[chain_id] = {}

                    if res_seq not in residue_map[chain_id]:
                        residue_map[chain_id][res_seq] = Residue(
                            name=atom.residue_name,
                            sequence=res_seq,
                            chain_id=chain_id,
                        )

                    residue_map[chain_id][res_seq].atoms.append(atom)

            elif record_type == "HETATM":
                hetatm = self._parse_hetatm_line(line)
                if hetatm:
                    structure.hetatms.append(hetatm)

            elif record_type == "CONECT":
                bonds = self._parse_conect_line(line)
                structure.bonds.extend(bonds)

        # Build chains from residue map
        for chain_id in sorted(residue_map.keys()):
            chain = Chain(chain_id=chain_id)
            for seq_num in sorted(residue_map[chain_id].keys()):
                chain.residues.append(residue_map[chain_id][seq_num])
            structure.chains.append(chain)

        structure.title = structure.title.strip()
        structure.authors = structure.authors.strip()

        self._structures[structure.pdb_id] = structure
        logger.info(
            f"Parsed PDB {structure.pdb_id}: "
            f"{len(structure.chains)} chains, "
            f"{structure.num_residues} residues, "
            f"{structure.num_atoms} atoms"
        )

        return structure

    def _parse_atom_record(self, line: str) -> Optional[Atom]:
        """Alias for :meth:`_parse_atom_line`."""
        return self._parse_atom_line(line)

    def _parse_atom_line(self, line: str) -> Optional[Atom]:
        """Parse ATOM record line."""
        try:
            return Atom(
                serial=int(line[6:11].strip()),
                name=line[12:16].strip(),
                alt_loc=line[16].strip(),
                residue_name=line[17:20].strip(),
                chain_id=line[21].strip() or "A",
                residue_seq=int(line[22:26].strip()),
                x=float(line[30:38].strip()),
                y=float(line[38:46].strip()),
                z=float(line[46:54].strip()),
                occupancy=float(line[54:60].strip()) if line[54:60].strip() else 1.0,
                temp_factor=float(line[60:66].strip()) if line[60:66].strip() else 0.0,
                element=line[76:78].strip() if len(line) > 76 else "",
                charge=line[78:80].strip() if len(line) > 78 else "",
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse ATOM line: {e}")
            return None

    def _parse_hetatm_line(self, line: str) -> Optional[Hetatm]:
        """Parse HETATM record line."""
        try:
            return Hetatm(
                serial=int(line[6:11].strip()),
                name=line[12:16].strip(),
                residue_name=line[17:20].strip(),
                chain_id=line[21].strip() or "A",
                residue_seq=int(line[22:26].strip()),
                x=float(line[30:38].strip()),
                y=float(line[38:46].strip()),
                z=float(line[46:54].strip()),
                element=line[76:78].strip() if len(line) > 76 else "",
            )
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse HETATM line: {e}")
            return None

    def _parse_conect_line(self, line: str) -> list[Bond]:
        """Parse CONECT record line."""
        bonds = []
        try:
            atom1 = int(line[6:11].strip())
            # Parse bonded atoms (up to 4)
            for i in range(4):
                start = 11 + i * 5
                end = start + 5
                if end <= len(line) and line[start:end].strip():
                    atom2 = int(line[start:end].strip())
                    if atom1 < atom2:  # Avoid duplicates
                        bonds.append(Bond(atom1_serial=atom1, atom2_serial=atom2))
        except (ValueError, IndexError) as e:
            logger.debug(f"Failed to parse CONECT line: {e}")

        return bonds

    def get_structure(self, pdb_id: str) -> Optional[PDBStructure]:
        """Get cached structure by PDB ID."""
        return self._structures.get(pdb_id.upper())

    def clear_cache(self) -> None:
        """Clear the structure cache."""
        self._structures.clear()
