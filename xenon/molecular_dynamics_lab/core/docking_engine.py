"""Molecular docking simulation engine.

Provides interactive docking simulation capabilities including:
- Rigid body docking
- Flexible docking
- Scoring functions
- Pose generation and optimization
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np

from .pdb_loader import PDBStructure

logger = logging.getLogger(__name__)


class ScoringFunction(Enum):
    """Docking scoring function types."""

    VINA = "vina"
    AUTODOCK = "autodock"
    GLIDE = "glide"
    SIMPLE = "simple"


class DockingMethod(Enum):
    """Docking method types."""

    RIGID = "rigid"
    FLEXIBLE = "flexible"
    INDUCED_FIT = "induced_fit"


@dataclass
class DockingConfig:
    """Configuration for docking simulation."""

    method: DockingMethod = DockingMethod.RIGID
    scoring: ScoringFunction = ScoringFunction.SIMPLE
    exhaustiveness: int = 8
    num_poses: int = 20
    energy_range: float = 3.0  # kcal/mol
    grid_spacing: float = 0.375  # Angstroms
    box_size: tuple[float, float, float] = (20.0, 20.0, 20.0)
    box_center: Optional[tuple[float, float, float]] = None
    random_seed: int = 42
    max_iterations: int = 1000
    convergence_threshold: float = 0.01
    temperature: float = 300.0  # Kelvin

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "method": self.method.value,
            "scoring": self.scoring.value,
            "exhaustiveness": self.exhaustiveness,
            "num_poses": self.num_poses,
            "energy_range": self.energy_range,
            "grid_spacing": self.grid_spacing,
            "box_size": list(self.box_size),
            "box_center": list(self.box_center) if self.box_center else None,
            "random_seed": self.random_seed,
            "max_iterations": self.max_iterations,
            "convergence_threshold": self.convergence_threshold,
            "temperature": self.temperature,
        }


@dataclass
class DockingPose:
    """A single docking pose result."""

    pose_id: int
    score: float
    rmsd: float
    transformation: np.ndarray  # 4x4 transformation matrix
    ligand_coords: np.ndarray  # Transformed ligand coordinates
    interactions: list[dict] = field(default_factory=list)
    cluster_id: int = 0

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "pose_id": self.pose_id,
            "score": self.score,
            "rmsd": self.rmsd,
            "transformation": self.transformation.tolist(),
            "ligand_coords": self.ligand_coords.tolist(),
            "interactions": self.interactions,
            "cluster_id": self.cluster_id,
        }


@dataclass
class DockingResult:
    """Complete docking simulation result."""

    receptor_id: str
    ligand_id: str
    config: DockingConfig
    poses: list[DockingPose] = field(default_factory=list)
    grid_energies: Optional[np.ndarray] = None
    runtime_seconds: float = 0.0
    converged: bool = False

    @property
    def best_pose(self) -> Optional[DockingPose]:
        """Get the best scoring pose."""
        if not self.poses:
            return None
        return min(self.poses, key=lambda p: p.score)

    @property
    def num_poses(self) -> int:
        """Get number of poses."""
        return len(self.poses)

    def get_poses_within_energy(self, energy_cutoff: float) -> list[DockingPose]:
        """Get poses within energy cutoff from best.

        Args:
            energy_cutoff: Energy cutoff in kcal/mol

        Returns:
            List of poses within cutoff
        """
        if not self.poses:
            return []

        best_score = self.best_pose.score
        return [p for p in self.poses if p.score - best_score <= energy_cutoff]

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "receptor_id": self.receptor_id,
            "ligand_id": self.ligand_id,
            "config": self.config.to_dict(),
            "poses": [p.to_dict() for p in self.poses],
            "runtime_seconds": self.runtime_seconds,
            "converged": self.converged,
            "best_score": self.best_pose.score if self.best_pose else None,
        }


class InteractionType(Enum):
    """Types of protein-ligand interactions."""

    HYDROGEN_BOND = "hydrogen_bond"
    HYDROPHOBIC = "hydrophobic"
    PI_STACKING = "pi_stacking"
    PI_CATION = "pi_cation"
    SALT_BRIDGE = "salt_bridge"
    HALOGEN_BOND = "halogen_bond"
    METAL_COORD = "metal_coordination"


@dataclass
class Interaction:
    """Protein-ligand interaction."""

    interaction_type: InteractionType
    receptor_atom: int  # Atom serial
    ligand_atom: int  # Atom serial
    distance: float
    energy: float
    receptor_residue: str
    receptor_chain: str

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "type": self.interaction_type.value,
            "receptor_atom": self.receptor_atom,
            "ligand_atom": self.ligand_atom,
            "distance": self.distance,
            "energy": self.energy,
            "receptor_residue": self.receptor_residue,
            "receptor_chain": self.receptor_chain,
        }


class DockingEngine:
    """Interactive molecular docking simulation engine.

    Provides methods for docking small molecule ligands into
    protein receptor binding sites.
    """

    # Van der Waals radii for common elements
    VDW_RADII = {
        "H": 1.20,
        "C": 1.70,
        "N": 1.55,
        "O": 1.52,
        "S": 1.80,
        "P": 1.80,
        "F": 1.47,
        "Cl": 1.75,
        "Br": 1.85,
        "I": 1.98,
        "Fe": 2.00,
        "Zn": 1.39,
    }

    # Hydrogen bond donors/acceptors by atom name
    HB_DONORS = {"N", "O", "S"}  # Simplified
    HB_ACCEPTORS = {"N", "O", "S", "F"}

    # Hydrophobic atoms
    HYDROPHOBIC = {"C"}

    def __init__(self, config: Optional[DockingConfig] = None):
        """Initialize docking engine.

        Args:
            config: Docking configuration
        """
        self.config = config or DockingConfig()
        self._rng = np.random.default_rng(self.config.random_seed)
        self._receptor: Optional[PDBStructure] = None
        self._ligand: Optional[PDBStructure] = None
        self._grid: Optional[np.ndarray] = None

    def set_receptor(self, receptor: PDBStructure) -> None:
        """Set the receptor protein structure.

        Args:
            receptor: Receptor PDB structure
        """
        self._receptor = receptor
        self._grid = None  # Reset grid
        logger.info(f"Set receptor: {receptor.pdb_id} with {receptor.num_atoms} atoms")

    def set_ligand(self, ligand: PDBStructure) -> None:
        """Set the ligand structure.

        Args:
            ligand: Ligand PDB structure
        """
        self._ligand = ligand
        logger.info(f"Set ligand: {ligand.pdb_id} with {ligand.num_atoms} atoms")

    def define_binding_site(
        self,
        center: tuple[float, float, float],
        size: tuple[float, float, float],
    ) -> None:
        """Define the binding site box.

        Args:
            center: Center coordinates (x, y, z)
            size: Box dimensions (dx, dy, dz)
        """
        self.config.box_center = center
        self.config.box_size = size
        logger.info(f"Defined binding site: center={center}, size={size}")

    def auto_detect_binding_site(
        self,
        reference_ligand: Optional[PDBStructure] = None,
        padding: float = 4.0,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        """Automatically detect binding site.

        Args:
            reference_ligand: Reference ligand for site detection
            padding: Padding around ligand in Angstroms

        Returns:
            Tuple of (center, size)
        """
        if reference_ligand:
            # Use reference ligand position
            coords = np.array([a.coords for a in reference_ligand.all_atoms])
        elif self._receptor and self._receptor.hetatms:
            # Use HETATM records (existing ligands/cofactors)
            coords = np.array(
                [h.coords for h in self._receptor.hetatms if h.res_name not in ("HOH", "WAT")]
            )
        else:
            # Use geometric center of receptor
            coords = np.array([a.coords for a in self._receptor.all_atoms])

        if len(coords) == 0:
            # Default to receptor center
            bbox = self._receptor.bounding_box
            center = tuple(self._receptor.center)
            size = tuple((bbox[1] - bbox[0]) / 2)
            return center, size

        center = tuple(np.mean(coords, axis=0))
        bbox_min = coords.min(axis=0) - padding
        bbox_max = coords.max(axis=0) + padding
        size = tuple(bbox_max - bbox_min)

        self.config.box_center = center
        self.config.box_size = size

        logger.info(f"Auto-detected binding site: center={center}, size={size}")
        return center, size

    def _prepare_grid(self) -> np.ndarray:
        """Prepare energy grid for docking.

        Returns:
            3D grid of energy values
        """
        if self._grid is not None:
            return self._grid

        center = np.array(self.config.box_center)
        size = np.array(self.config.box_size)
        spacing = self.config.grid_spacing

        # Calculate grid dimensions
        n_points = (size / spacing).astype(int) + 1

        # Initialize grid
        grid = np.zeros(tuple(n_points))

        # Calculate receptor contribution to grid
        receptor_atoms = self._receptor.all_atoms

        for i in range(n_points[0]):
            for j in range(n_points[1]):
                for k in range(n_points[2]):
                    point = center - size / 2 + np.array([i, j, k]) * spacing

                    # Sum interactions with receptor atoms
                    energy = 0.0
                    for atom in receptor_atoms:
                        dist = np.linalg.norm(point - atom.coords)
                        if dist < 10.0:  # Cutoff
                            energy += self._lennard_jones(dist, atom.element)

                    grid[i, j, k] = energy

        self._grid = grid
        logger.info(f"Prepared grid: shape={grid.shape}")
        return grid

    def _lennard_jones(self, r: float, element: str, epsilon: float = 0.1) -> float:
        """Calculate Lennard-Jones potential.

        Args:
            r: Distance in Angstroms
            element: Element symbol
            epsilon: Well depth

        Returns:
            Energy in kcal/mol
        """
        sigma = self.VDW_RADII.get(element, 1.7)

        if r < 0.1:
            return 1000.0  # Clash penalty

        ratio = sigma / r
        return 4 * epsilon * (ratio**12 - ratio**6)

    def _score_pose(
        self,
        ligand_coords: np.ndarray,
    ) -> tuple[float, list[Interaction]]:
        """Score a ligand pose.

        Args:
            ligand_coords: Ligand atomic coordinates

        Returns:
            Tuple of (score, interactions)
        """
        if not self._receptor or not self._ligand:
            return 0.0, []

        score = 0.0
        interactions = []

        receptor_atoms = self._receptor.all_atoms
        ligand_atoms = self._ligand.all_atoms

        # Calculate pairwise interactions
        for i, lig_atom in enumerate(ligand_atoms):
            lig_coord = ligand_coords[i]

            for rec_atom in receptor_atoms:
                dist = np.linalg.norm(lig_coord - rec_atom.coords)

                if dist > 8.0:  # Cutoff
                    continue

                # Van der Waals
                vdw_energy = self._lennard_jones(dist, rec_atom.element)
                score += vdw_energy

                # Hydrogen bonds
                if dist < 3.5:
                    lig_elem = lig_atom.element or lig_atom.name[0]
                    rec_elem = rec_atom.element or rec_atom.name[0]

                    if (lig_elem in self.HB_DONORS and rec_elem in self.HB_ACCEPTORS) or (
                        lig_elem in self.HB_ACCEPTORS and rec_elem in self.HB_DONORS
                    ):
                        hb_energy = -2.5 * (1 - (dist - 2.8) ** 2 / 0.7**2)
                        hb_energy = max(hb_energy, -2.5)
                        score += hb_energy

                        if hb_energy < -0.5:
                            interactions.append(
                                Interaction(
                                    interaction_type=InteractionType.HYDROGEN_BOND,
                                    receptor_atom=rec_atom.serial,
                                    ligand_atom=lig_atom.serial,
                                    distance=dist,
                                    energy=hb_energy,
                                    receptor_residue=f"{rec_atom.res_name}{rec_atom.res_seq}",
                                    receptor_chain=rec_atom.chain_id,
                                )
                            )

                # Hydrophobic contacts
                if dist < 4.0:
                    lig_elem = lig_atom.element or lig_atom.name[0]
                    rec_elem = rec_atom.element or rec_atom.name[0]

                    if lig_elem in self.HYDROPHOBIC and rec_elem in self.HYDROPHOBIC:
                        hp_energy = -0.3 * (4.0 - dist)
                        score += hp_energy

                        if hp_energy < -0.5:
                            interactions.append(
                                Interaction(
                                    interaction_type=InteractionType.HYDROPHOBIC,
                                    receptor_atom=rec_atom.serial,
                                    ligand_atom=lig_atom.serial,
                                    distance=dist,
                                    energy=hp_energy,
                                    receptor_residue=f"{rec_atom.res_name}{rec_atom.res_seq}",
                                    receptor_chain=rec_atom.chain_id,
                                )
                            )

        return score, interactions

    def _generate_random_pose(self) -> np.ndarray:
        """Generate random transformation matrix.

        Returns:
            4x4 transformation matrix
        """
        center = np.array(self.config.box_center)
        size = np.array(self.config.box_size)

        # Random translation within box
        translation = center + (self._rng.random(3) - 0.5) * size

        # Random rotation
        angles = self._rng.random(3) * 2 * np.pi

        # Build rotation matrix
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)

        rotation = np.array(
            [
                [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
                [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
                [-sy, sx * cy, cx * cy],
            ]
        )

        # Build transformation matrix
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation

        return transform

    def _apply_transform(
        self,
        coords: np.ndarray,
        transform: np.ndarray,
    ) -> np.ndarray:
        """Apply transformation to coordinates.

        Args:
            coords: Nx3 coordinate array
            transform: 4x4 transformation matrix

        Returns:
            Transformed coordinates
        """
        # Center coordinates at origin
        centroid = np.mean(coords, axis=0)
        centered = coords - centroid

        # Apply rotation
        rotated = centered @ transform[:3, :3].T

        # Apply translation
        return rotated + transform[:3, 3]

    def _optimize_pose(
        self,
        initial_transform: np.ndarray,
        ligand_coords: np.ndarray,
    ) -> tuple[np.ndarray, float, list[Interaction]]:
        """Optimize a docking pose using gradient descent.

        Args:
            initial_transform: Initial transformation matrix
            ligand_coords: Original ligand coordinates

        Returns:
            Tuple of (optimized_transform, score, interactions)
        """
        transform = initial_transform.copy()
        step_size = 0.1
        best_score = float("inf")
        best_transform = transform.copy()
        best_interactions = []

        for iteration in range(self.config.max_iterations):
            # Apply current transform
            current_coords = self._apply_transform(ligand_coords, transform)
            score, interactions = self._score_pose(current_coords)

            if score < best_score:
                best_score = score
                best_transform = transform.copy()
                best_interactions = interactions

            # Generate neighboring transforms
            improved = False
            for dim in range(6):  # 3 translation + 3 rotation
                for direction in [-1, 1]:
                    new_transform = transform.copy()

                    if dim < 3:
                        new_transform[dim, 3] += direction * step_size
                    else:
                        # Small rotation
                        angle = direction * step_size * 0.1
                        axis = dim - 3
                        c, s = np.cos(angle), np.sin(angle)

                        if axis == 0:
                            rot = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
                        elif axis == 1:
                            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
                        else:
                            rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

                        new_transform[:3, :3] = new_transform[:3, :3] @ rot

                    new_coords = self._apply_transform(ligand_coords, new_transform)
                    new_score, _ = self._score_pose(new_coords)

                    if new_score < score:
                        transform = new_transform
                        score = new_score
                        improved = True
                        break

                if improved:
                    break

            # Check convergence
            if not improved:
                step_size *= 0.5
                if step_size < self.config.convergence_threshold:
                    break

        return best_transform, best_score, best_interactions

    def dock(self) -> DockingResult:
        """Run docking simulation.

        Returns:
            DockingResult with poses
        """
        import time

        if not self._receptor:
            raise ValueError("Receptor not set")
        if not self._ligand:
            raise ValueError("Ligand not set")

        if not self.config.box_center:
            self.auto_detect_binding_site()

        start_time = time.time()
        logger.info(f"Starting docking: {self._ligand.pdb_id} → {self._receptor.pdb_id}")

        # Get ligand coordinates
        ligand_coords = np.array([a.coords for a in self._ligand.all_atoms])

        poses = []
        reference_coords = None

        # Generate and optimize poses
        for i in range(self.config.exhaustiveness * self.config.num_poses):
            # Generate random initial pose
            initial_transform = self._generate_random_pose()

            # Optimize pose
            transform, score, interactions = self._optimize_pose(initial_transform, ligand_coords)

            # Calculate RMSD to reference (first pose)
            transformed_coords = self._apply_transform(ligand_coords, transform)
            if reference_coords is None:
                reference_coords = transformed_coords
                rmsd = 0.0
            else:
                diff = transformed_coords - reference_coords
                rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

            pose = DockingPose(
                pose_id=len(poses),
                score=score,
                rmsd=rmsd,
                transformation=transform,
                ligand_coords=transformed_coords,
                interactions=[inter.to_dict() for inter in interactions],
            )
            poses.append(pose)

        # Sort by score and keep best poses
        poses.sort(key=lambda p: p.score)
        best_score = poses[0].score if poses else 0

        # Filter by energy range
        filtered_poses = [p for p in poses if p.score - best_score <= self.config.energy_range][
            : self.config.num_poses
        ]

        # Cluster similar poses
        self._cluster_poses(filtered_poses)

        runtime = time.time() - start_time
        logger.info(
            f"Docking complete: {len(filtered_poses)} poses, "
            f"best score: {best_score:.2f}, runtime: {runtime:.2f}s"
        )

        return DockingResult(
            receptor_id=self._receptor.pdb_id,
            ligand_id=self._ligand.pdb_id,
            config=self.config,
            poses=filtered_poses,
            runtime_seconds=runtime,
            converged=True,
        )

    def _cluster_poses(
        self,
        poses: list[DockingPose],
        rmsd_cutoff: float = 2.0,
    ) -> None:
        """Cluster poses by RMSD.

        Args:
            poses: List of poses to cluster
            rmsd_cutoff: RMSD cutoff for clustering
        """
        cluster_id = 0

        for i, pose in enumerate(poses):
            if pose.cluster_id != 0:
                continue

            cluster_id += 1
            pose.cluster_id = cluster_id

            for j in range(i + 1, len(poses)):
                other = poses[j]
                if other.cluster_id != 0:
                    continue

                diff = pose.ligand_coords - other.ligand_coords
                rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

                if rmsd < rmsd_cutoff:
                    other.cluster_id = cluster_id

    def interactive_dock(
        self,
        ligand_position: tuple[float, float, float],
        ligand_rotation: tuple[float, float, float],
    ) -> tuple[float, list[dict]]:
        """Perform interactive docking with user-specified pose.

        Args:
            ligand_position: Ligand center position
            ligand_rotation: Ligand rotation angles (degrees)

        Returns:
            Tuple of (score, interactions)
        """
        if not self._receptor or not self._ligand:
            raise ValueError("Receptor and ligand must be set")

        # Build transformation matrix
        angles = np.radians(ligand_rotation)
        cx, cy, cz = np.cos(angles)
        sx, sy, sz = np.sin(angles)

        rotation = np.array(
            [
                [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
                [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
                [-sy, sx * cy, cx * cy],
            ]
        )

        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = np.array(ligand_position)

        # Apply transform and score
        ligand_coords = np.array([a.coords for a in self._ligand.all_atoms])
        transformed = self._apply_transform(ligand_coords, transform)
        score, interactions = self._score_pose(transformed)

        return score, [i.to_dict() for i in interactions]
