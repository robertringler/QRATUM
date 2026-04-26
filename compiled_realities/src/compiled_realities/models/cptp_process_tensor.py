"""CPTP Process-Tensor Model implementation (M2).

Finite-dimensional quantum system with Lindblad master equation.
"""

from typing import Any, Dict

import numpy as np
from scipy.linalg import expm

from .base import BaseModel


class CPTPProcessTensorModel(BaseModel):
    """
    CPTP (Completely Positive Trace Preserving) Process-Tensor Model.
    
    Simulates a finite-dimensional quantum system evolving under
    a Lindblad master equation with explicit time ordering.
    """
    
    def __init__(self, config: Dict[str, Any], seed: int):
        """
        Initialize CPTP model.
        
        Args:
            config: Configuration with:
                - dimension: Hilbert space dimension
                - environment_coupling: Coupling strength to environment
                - t1_t2_separation: Time steps between t1 and t2
                - n_timesteps: Total number of timesteps
                - dt: Time step size
                - lindblad_operators: Type of decoherence
        """
        super().__init__(config, seed)
        
        self.dim = config['dimension']
        self.gamma = config['environment_coupling']
        self.t1 = config['t1_t2_separation']
        self.t2 = 2 * config['t1_t2_separation']  # t2 > t1
        self.n_timesteps = config.get('n_timesteps', 50)
        self.dt = config.get('dt', 0.1)
        
        # Initialize system state (pure state |0⟩)
        self.init_state = self._get_initial_density_matrix()
        
        # Define Hamiltonian and Lindblad operators
        self.H = self._get_hamiltonian()
        self.L_ops = self._get_lindblad_operators(config.get('lindblad_operators', 'dephasing'))
        
    def _get_initial_density_matrix(self) -> np.ndarray:
        """Get initial density matrix (ground state)."""
        rho = np.zeros((self.dim, self.dim), dtype=complex)
        rho[0, 0] = 1.0
        return rho
    
    def _get_hamiltonian(self) -> np.ndarray:
        """Get system Hamiltonian (random but reproducible)."""
        # Generate random Hermitian Hamiltonian
        H_real = self.rng.randn(self.dim, self.dim)
        H = (H_real + H_real.T) / 2.0
        return H
    
    def _get_lindblad_operators(self, operator_type: str):
        """Get Lindblad operators for decoherence."""
        if operator_type == 'dephasing':
            # Dephasing in computational basis
            L = [np.diag([1.0 if i == j else 0.0 for i in range(self.dim)]) 
                 for j in range(self.dim)]
        elif operator_type == 'amplitude_damping':
            # Amplitude damping operators
            L = []
            for i in range(self.dim - 1):
                L_i = np.zeros((self.dim, self.dim), dtype=complex)
                L_i[i, i + 1] = 1.0
                L.append(L_i)
        else:
            # Default: single dephasing operator
            L = [np.diag([1.0 if i == 0 else 0.0 for i in range(self.dim)])]
        
        return L
    
    def _lindblad_step(self, rho: np.ndarray, dt: float) -> np.ndarray:
        """
        Single timestep of Lindblad master equation.
        
        dρ/dt = -i[H, ρ] + Σ_k (L_k ρ L_k† - 1/2{L_k†L_k, ρ})
        """
        # Hamiltonian evolution
        commutator = -1j * (self.H @ rho - rho @ self.H)
        
        # Dissipative evolution
        dissipator = np.zeros_like(rho)
        for L in self.L_ops:
            L_dag = L.conj().T
            L_dag_L = L_dag @ L
            dissipator += (L @ rho @ L_dag - 0.5 * (L_dag_L @ rho + rho @ L_dag_L))
        
        drho = commutator + self.gamma * dissipator
        return rho + dt * drho
    
    def _apply_intervention(self, rho: np.ndarray, intervention: int) -> np.ndarray:
        """
        Apply intervention at t2.
        
        Args:
            rho: Density matrix before intervention
            intervention: 0 (no intervention) or 1 (apply unitary)
            
        Returns:
            Density matrix after intervention
        """
        if intervention == 0:
            return rho
        else:
            # Apply a random unitary (Pauli-like rotation)
            # For reproducibility, use a fixed rotation based on seed
            angle = np.pi / 4
            if self.dim == 2:
                # Pauli-X rotation
                U = np.array([[np.cos(angle), -1j * np.sin(angle)],
                             [-1j * np.sin(angle), np.cos(angle)]], dtype=complex)
            else:
                # Generic unitary rotation
                norm_H = np.linalg.norm(self.H)
                U = expm(-1j * angle * self.H / norm_H) if norm_H > 1e-12 else np.eye(self.dim, dtype=complex)
            
            return U @ rho @ U.conj().T
    
    def run_simulation(self, intervention: int) -> Dict[str, Any]:
        """
        Run CPTP simulation.
        
        Args:
            intervention: Intervention value at t2 (0 or 1)
            
        Returns:
            Simulation results
        """
        rho = self.init_state.copy()
        trajectory = [rho.copy()]
        observables = []
        
        for t in range(self.n_timesteps):
            # Apply intervention at t2
            if t == self.t2:
                rho = self._apply_intervention(rho, intervention)
            
            # Evolve one timestep
            rho = self._lindblad_step(rho, self.dt)
            trajectory.append(rho.copy())
            
            # Measure observable (population of ground state)
            obs = np.real(rho[0, 0])
            observables.append(obs)
        
        return {
            'observables_t1': observables[self.t1] if self.t1 < len(observables) else observables[-1],
            'full_trajectory': trajectory,
            'observables_full': observables,
            'metadata': {
                'dimension': self.dim,
                'gamma': self.gamma,
                't1': self.t1,
                't2': self.t2,
                'intervention': intervention
            }
        }
    
    def get_observable_t1(self, results: Dict[str, Any]) -> float:
        """
        Extract observable at t1.
        
        CRITICAL: This function must NOT access data after t1.
        """
        return results['observables_t1']
