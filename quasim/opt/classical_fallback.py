"""Classical optimization fallbacks for problems exceeding quantum capacity.

This module provides classical alternatives for quantum algorithms when:
- Problem size exceeds max_qubits (typically 20)
- Quantum backend unavailable
- User explicitly requests classical computation

Supports:
- Molecular energy calculations (PySCF)
- MaxCut optimization (NetworkX/brute force)
- General combinatorial optimization
"""

from __future__ import annotations

from typing import Any


class ClassicalFallback:
    """Classical optimization fallbacks for quantum algorithms.

    Automatically invoked when:
    - Problem size > max_qubits (typically 20)
    - Quantum backend unavailable
    - User explicitly requests classical
    """

    def __init__(self, max_problem_size: int = 20):
        """Initialize classical fallback system.

        Args:
            max_problem_size: Maximum problem size for exact algorithms
        """
        self.max_problem_size = max_problem_size

    def solve_molecular_energy(
        self,
        molecule: str,
        bond_length: float,
        basis: str = "sto3g",
        method: str = "RHF",
    ) -> dict[str, Any]:
        """Classical quantum chemistry via PySCF.

        Args:
            molecule: Molecule formula (e.g., "H2", "LiH")
            bond_length: Bond length in Angstroms
            basis: Basis set for calculation
            method: Quantum chemistry method (RHF, CCSD, FCI)

        Returns:
            Dict with energy, method info, and convergence status
        """
        try:
            from pyscf import gto, scf

            # Parse molecule string and build geometry
            if molecule == "H2":
                geometry = f"H 0 0 0; H 0 0 {bond_length}"
            elif molecule == "LiH":
                geometry = f"Li 0 0 0; H 0 0 {bond_length}"
            elif molecule == "BeH2":
                geometry = f"Be 0 0 0; H 0 0 {bond_length}; H 0 0 {-bond_length}"
            else:
                raise ValueError(f"Unsupported molecule: {molecule}")

            # Build molecule
            mol = gto.M(atom=geometry, basis=basis, unit="Angstrom")

            # Compute energy based on method
            if method == "RHF" or method == "HF":
                mf = scf.RHF(mol)
                energy = mf.kernel()
                converged = mf.converged
            elif method == "CCSD":
                from pyscf import cc

                mf = scf.RHF(mol).run()
                mycc = cc.CCSD(mf)
                energy = mycc.kernel()[0] + mf.e_tot
                converged = mycc.converged
            elif method == "FCI":
                from pyscf import fci

                mf = scf.RHF(mol).run()
                cisolver = fci.FCI(mol, mf.mo_coeff)
                energy = cisolver.kernel()[0] + mol.energy_nuc()
                converged = True
            else:
                raise ValueError(f"Unknown method: {method}")

            return {
                "energy": float(energy),
                "method": method,
                "basis": basis,
                "molecule": molecule,
                "converged": converged,
                "classical": True,
            }

        except ImportError as e:
            raise ImportError(
                "PySCF required for classical quantum chemistry. "
                "Install: pip install pyscf"
            ) from e

    def solve_maxcut(
        self, edges: list[tuple[int, int]], method: str = "exact"
    ) -> dict[str, Any]:
        """Classical MaxCut via NetworkX or Gurobi.

        Args:
            edges: List of graph edges [(i, j), ...]
            method: Solution method ('exact', 'greedy', 'networkx')

        Returns:
            Dict with solution, cut value, and method info
        """
        n_nodes = max(max(e) for e in edges) + 1

        if method == "exact":
            # Brute force for small graphs
            if n_nodes > self.max_problem_size:
                raise ValueError(
                    f"Problem too large for exact solution ({n_nodes} > {self.max_problem_size}). "
                    "Use method='greedy' or 'networkx'"
                )

            best_cut = 0
            best_partition = None

            # Try all 2^n partitions
            for partition_int in range(2**n_nodes):
                partition = [
                    (partition_int >> i) & 1 for i in range(n_nodes)
                ]

                # Count edges crossing partition
                cut_size = sum(
                    1
                    for i, j in edges
                    if partition[i] != partition[j]
                )

                if cut_size > best_cut:
                    best_cut = cut_size
                    best_partition = partition

            solution = "".join(str(p) for p in best_partition)
            return {
                "solution": solution,
                "cut_value": best_cut,
                "method": "exact",
                "classical": True,
                "optimal": True,
            }

        elif method == "greedy":
            # Greedy heuristic
            partition = [0] * n_nodes

            for node in range(n_nodes):
                # Try placing node in each partition
                partition[node] = 0
                cut_0 = sum(
                    1
                    for i, j in edges
                    if (i == node or j == node) and partition[i] != partition[j]
                )

                partition[node] = 1
                cut_1 = sum(
                    1
                    for i, j in edges
                    if (i == node or j == node) and partition[i] != partition[j]
                )

                # Choose partition with better cut
                partition[node] = 1 if cut_1 > cut_0 else 0

            cut_value = sum(1 for i, j in edges if partition[i] != partition[j])
            solution = "".join(str(p) for p in partition)

            return {
                "solution": solution,
                "cut_value": cut_value,
                "method": "greedy",
                "classical": True,
                "optimal": False,
            }

        elif method == "networkx":
            try:
                import networkx as nx

                G = nx.Graph(edges)
                # Use NetworkX approximation algorithm
                cut = nx.algorithms.approximation.max_cut(G)
                cut_value = len(
                    [
                        e
                        for e in edges
                        if (e[0] in cut and e[1] not in cut)
                        or (e[0] not in cut and e[1] in cut)
                    ]
                )

                # Convert to bitstring
                partition = [1 if i in cut else 0 for i in range(n_nodes)]
                solution = "".join(str(p) for p in partition)

                return {
                    "solution": solution,
                    "cut_value": cut_value,
                    "method": "networkx",
                    "classical": True,
                    "optimal": False,
                }
            except ImportError as e:
                raise ImportError("NetworkX required. Install: pip install networkx") from e

        else:
            raise ValueError(f"Unknown method: {method}")

    def solve_optimization(
        self, problem_type: str, problem_data: dict[str, Any], **kwargs: Any
    ) -> dict[str, Any]:
        """Generalized classical optimization router.

        Args:
            problem_type: Type of optimization problem
            problem_data: Problem-specific data
            **kwargs: Additional solver parameters

        Returns:
            Dict with solution and metadata
        """
        if problem_type == "molecular_energy":
            return self.solve_molecular_energy(
                molecule=problem_data["molecule"],
                bond_length=problem_data["bond_length"],
                basis=problem_data.get("basis", "sto3g"),
                method=kwargs.get("method", "RHF"),
            )
        elif problem_type == "maxcut":
            return self.solve_maxcut(
                edges=problem_data["edges"], method=kwargs.get("method", "exact")
            )
        else:
            raise ValueError(f"Unsupported problem type: {problem_type}")


__all__ = ["ClassicalFallback"]
