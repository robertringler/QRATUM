"""Protein structure analysis utilities.

Provides functionality for:
- PDB file parsing
- Structure quality assessment
- Binding site prediction
- Structure comparison
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class Atom:
    """Protein atom.
    
    Attributes:
        atom_id: Atom serial number
        atom_name: Atom name (e.g., CA, N, C, O)
        residue_name: Residue name (e.g., ALA, GLY)
        chain_id: Chain identifier
        residue_number: Residue sequence number
        x: X coordinate
        y: Y coordinate
        z: Z coordinate
        element: Element symbol
    """
    
    atom_id: int
    atom_name: str
    residue_name: str
    chain_id: str
    residue_number: int
    x: float
    y: float
    z: float
    element: str = ""
    
    def coordinates(self) -> np.ndarray:
        """Get coordinates as numpy array."""
        return np.array([self.x, self.y, self.z])
    
    def distance_to(self, other: Atom) -> float:
        """Compute distance to another atom."""
        return float(np.linalg.norm(self.coordinates() - other.coordinates()))


@dataclass
class Residue:
    """Protein residue.
    
    Attributes:
        residue_name: Three-letter residue code
        chain_id: Chain identifier
        residue_number: Sequence number
        atoms: List of atoms in residue
    """
    
    residue_name: str
    chain_id: str
    residue_number: int
    atoms: List[Atom] = field(default_factory=list)
    
    def get_alpha_carbon(self) -> Optional[Atom]:
        """Get alpha carbon atom."""
        for atom in self.atoms:
            if atom.atom_name == "CA":
                return atom
        return None
    
    def center_of_mass(self) -> np.ndarray:
        """Compute center of mass."""
        if not self.atoms:
            return np.array([0.0, 0.0, 0.0])
        
        coords = np.array([atom.coordinates() for atom in self.atoms])
        return np.mean(coords, axis=0)


@dataclass
class ProteinStructure:
    """Protein 3D structure.
    
    Attributes:
        pdb_id: PDB identifier
        title: Structure title
        method: Experimental method
        resolution: Resolution in Angstroms
        chains: Dictionary of chain ID to list of residues
    """
    
    pdb_id: str
    title: str = ""
    method: str = ""
    resolution: float = 0.0
    chains: Dict[str, List[Residue]] = field(default_factory=dict)
    
    def get_all_atoms(self) -> List[Atom]:
        """Get all atoms in structure."""
        atoms = []
        for chain_residues in self.chains.values():
            for residue in chain_residues:
                atoms.extend(residue.atoms)
        return atoms
    
    def get_chain(self, chain_id: str) -> List[Residue]:
        """Get residues for a chain."""
        return self.chains.get(chain_id, [])
    
    def num_residues(self) -> int:
        """Get total number of residues."""
        return sum(len(residues) for residues in self.chains.values())


class StructureAnalyzer:
    """Protein structure analysis and comparison.
    
    Provides tools for analyzing protein 3D structures, predicting
    binding sites, and comparing structures.
    """
    
    def __init__(self):
        """Initialize structure analyzer."""
        self._structures: Dict[str, ProteinStructure] = {}
    
    def parse_pdb(self, pdb_content: str) -> ProteinStructure:
        """Parse PDB format file (simplified).
        
        Args:
            pdb_content: PDB file content
        
        Returns:
            ProteinStructure object
        """
        structure = ProteinStructure(pdb_id="unknown")
        current_chain = "A"
        residues_by_chain: Dict[str, Dict[int, Residue]] = {}
        
        for line in pdb_content.split('\n'):
            if line.startswith('HEADER'):
                structure.pdb_id = line[62:66].strip()
            
            elif line.startswith('TITLE'):
                structure.title += line[10:].strip() + " "
            
            elif line.startswith('REMARK   2 RESOLUTION'):
                try:
                    res_str = line.split()[-2]
                    structure.resolution = float(res_str)
                except (ValueError, IndexError):
                    pass
            
            elif line.startswith('ATOM') or line.startswith('HETATM'):
                try:
                    atom_id = int(line[6:11].strip())
                    atom_name = line[12:16].strip()
                    residue_name = line[17:20].strip()
                    chain_id = line[21].strip() or 'A'
                    residue_number = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                    element = line[76:78].strip() if len(line) > 76 else ""
                    
                    atom = Atom(
                        atom_id=atom_id,
                        atom_name=atom_name,
                        residue_name=residue_name,
                        chain_id=chain_id,
                        residue_number=residue_number,
                        x=x, y=y, z=z,
                        element=element,
                    )
                    
                    # Organize by chain and residue
                    if chain_id not in residues_by_chain:
                        residues_by_chain[chain_id] = {}
                    
                    if residue_number not in residues_by_chain[chain_id]:
                        residues_by_chain[chain_id][residue_number] = Residue(
                            residue_name=residue_name,
                            chain_id=chain_id,
                            residue_number=residue_number,
                        )
                    
                    residues_by_chain[chain_id][residue_number].atoms.append(atom)
                
                except (ValueError, IndexError):
                    continue
        
        # Convert to structure format
        for chain_id, residue_dict in residues_by_chain.items():
            structure.chains[chain_id] = [
                residue_dict[num] for num in sorted(residue_dict.keys())
            ]
        
        self._structures[structure.pdb_id] = structure
        return structure
    
    def compute_rmsd(
        self,
        structure1: ProteinStructure,
        structure2: ProteinStructure,
        chain1: str = 'A',
        chain2: str = 'A',
    ) -> float:
        """Compute root-mean-square deviation between structures.
        
        Args:
            structure1: First structure
            structure2: Second structure
            chain1: Chain in first structure
            chain2: Chain in second structure
        
        Returns:
            RMSD in Angstroms
        """
        residues1 = structure1.get_chain(chain1)
        residues2 = structure2.get_chain(chain2)
        
        # Get alpha carbons
        coords1 = []
        coords2 = []
        
        min_len = min(len(residues1), len(residues2))
        for i in range(min_len):
            ca1 = residues1[i].get_alpha_carbon()
            ca2 = residues2[i].get_alpha_carbon()
            
            if ca1 and ca2:
                coords1.append(ca1.coordinates())
                coords2.append(ca2.coordinates())
        
        if not coords1:
            return 0.0
        
        coords1_arr = np.array(coords1)
        coords2_arr = np.array(coords2)
        
        # Center coordinates
        coords1_arr -= np.mean(coords1_arr, axis=0)
        coords2_arr -= np.mean(coords2_arr, axis=0)
        
        # Compute RMSD
        diff = coords1_arr - coords2_arr
        rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))
        
        return float(rmsd)
    
    def find_binding_sites(
        self,
        structure: ProteinStructure,
        ligand_coords: Optional[np.ndarray] = None,
        distance_cutoff: float = 5.0,
    ) -> List[Residue]:
        """Identify potential binding site residues.
        
        Args:
            structure: Protein structure
            ligand_coords: Ligand coordinates (if known)
            distance_cutoff: Distance threshold in Angstroms
        
        Returns:
            List of residues in binding site
        """
        if ligand_coords is None:
            # Use geometric center as proxy
            all_atoms = structure.get_all_atoms()
            if not all_atoms:
                return []
            coords = np.array([atom.coordinates() for atom in all_atoms])
            ligand_coords = np.mean(coords, axis=0)
        
        binding_residues = []
        
        for chain_residues in structure.chains.values():
            for residue in chain_residues:
                # Check if any atom is close to ligand
                for atom in residue.atoms:
                    dist = np.linalg.norm(atom.coordinates() - ligand_coords)
                    if dist <= distance_cutoff:
                        binding_residues.append(residue)
                        break
        
        return binding_residues
    
    def compute_secondary_structure(
        self,
        structure: ProteinStructure,
        chain_id: str = 'A',
    ) -> Dict[int, str]:
        """Predict secondary structure (simplified).
        
        Uses phi/psi angles to classify as helix, sheet, or coil.
        
        Args:
            structure: Protein structure
            chain_id: Chain to analyze
        
        Returns:
            Dictionary mapping residue number to structure type
        """
        residues = structure.get_chain(chain_id)
        secondary = {}
        
        for i in range(1, len(residues) - 1):
            # Simplified: would need to compute dihedral angles
            # For now, assign based on residue type tendencies
            res_name = residues[i].residue_name
            
            # Helix-forming residues
            if res_name in ('ALA', 'GLU', 'LEU', 'MET'):
                secondary[residues[i].residue_number] = 'H'  # Helix
            # Sheet-forming residues
            elif res_name in ('VAL', 'ILE', 'TYR', 'PHE', 'TRP'):
                secondary[residues[i].residue_number] = 'E'  # Extended (sheet)
            # Others
            else:
                secondary[residues[i].residue_number] = 'C'  # Coil
        
        return secondary
    
    def assess_structure_quality(
        self,
        structure: ProteinStructure,
    ) -> Dict[str, float]:
        """Assess structure quality metrics.
        
        Args:
            structure: Protein structure
        
        Returns:
            Dictionary of quality metrics
        """
        quality = {
            "resolution": structure.resolution,
            "num_residues": structure.num_residues(),
            "num_chains": len(structure.chains),
            "completeness": 0.0,
        }
        
        # Check completeness (percentage of residues with all backbone atoms)
        complete_residues = 0
        total_residues = 0
        
        for chain_residues in structure.chains.values():
            for residue in chain_residues:
                total_residues += 1
                atom_names = {atom.atom_name for atom in residue.atoms}
                
                # Check for backbone atoms
                if {'N', 'CA', 'C', 'O'}.issubset(atom_names):
                    complete_residues += 1
        
        if total_residues > 0:
            quality["completeness"] = complete_residues / total_residues
        
        return quality
    
    def find_clashes(
        self,
        structure: ProteinStructure,
        distance_threshold: float = 2.0,
    ) -> List[Tuple[Atom, Atom, float]]:
        """Find steric clashes in structure.
        
        Args:
            structure: Protein structure
            distance_threshold: Minimum allowed distance
        
        Returns:
            List of (atom1, atom2, distance) tuples
        """
        atoms = structure.get_all_atoms()
        clashes = []
        
        for i in range(len(atoms)):
            for j in range(i + 1, len(atoms)):
                atom1 = atoms[i]
                atom2 = atoms[j]
                
                # Skip atoms in same residue
                if (atom1.chain_id == atom2.chain_id and
                    atom1.residue_number == atom2.residue_number):
                    continue
                
                dist = atom1.distance_to(atom2)
                
                if dist < distance_threshold:
                    clashes.append((atom1, atom2, dist))
        
        return clashes
    
    def compute_surface_residues(
        self,
        structure: ProteinStructure,
        chain_id: str = 'A',
        solvent_radius: float = 1.4,
    ) -> List[int]:
        """Identify surface-exposed residues.
        
        Args:
            structure: Protein structure
            chain_id: Chain to analyze
            solvent_radius: Radius of solvent probe
        
        Returns:
            List of surface residue numbers
        """
        residues = structure.get_chain(chain_id)
        surface = []
        
        # Simplified: residues are surface if their CA is far from center
        all_atoms = structure.get_all_atoms()
        if not all_atoms:
            return []
        
        coords = np.array([atom.coordinates() for atom in all_atoms])
        center = np.mean(coords, axis=0)
        
        # Compute radius of gyration
        distances = np.linalg.norm(coords - center, axis=1)
        rg = np.sqrt(np.mean(distances**2))
        
        # Surface residues are those with CA beyond 70% of Rg
        threshold = 0.7 * rg
        
        for residue in residues:
            ca = residue.get_alpha_carbon()
            if ca:
                dist = np.linalg.norm(ca.coordinates() - center)
                if dist >= threshold:
                    surface.append(residue.residue_number)
        
        return surface
    
    def get_structure(self, pdb_id: str) -> Optional[ProteinStructure]:
        """Retrieve a parsed structure by PDB ID.
        
        Args:
            pdb_id: PDB identifier
        
        Returns:
            ProteinStructure if found
        """
        return self._structures.get(pdb_id)
