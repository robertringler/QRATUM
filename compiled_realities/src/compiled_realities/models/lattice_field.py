"""Lattice Field Model implementation (M3).

Discrete 1+1D lattice with tunable dispersion and effective metric.
"""

from typing import Any, Dict

import numpy as np
from scipy.fft import fft, ifft

from .base import BaseModel


class LatticeFieldModel(BaseModel):
    """
    Lattice Field Model with effective metric reconstruction.
    
    Simulates a 1+1D discrete lattice field with tunable dispersion
    and flow velocity for effective light cone tilting.
    """
    
    def __init__(self, config: Dict[str, Any], seed: int):
        """
        Initialize Lattice Field Model.
        
        Args:
            config: Configuration with:
                - n_sites: Number of lattice sites
                - cutoff_scale: Momentum cutoff scale
                - dispersion_strength: Dispersion parameter
                - flow_velocity_ratio: Flow velocity / light speed
                - dissipation_rate: Energy dissipation rate
                - n_timesteps: Total timesteps
                - dt: Time step size
                - boundary_conditions: 'periodic' or 'open'
                - lattice_spacing: Spatial lattice spacing
        """
        super().__init__(config, seed)
        
        self.n_sites = config['n_sites']
        self.cutoff = config['cutoff_scale']
        self.dispersion = config['dispersion_strength']
        self.flow_vel = config['flow_velocity_ratio']
        self.dissipation = config['dissipation_rate']
        self.n_timesteps = config.get('n_timesteps', 100)
        self.dt = config.get('dt', 0.01)
        self.dx = config.get('lattice_spacing', 1.0)
        
        # Set intervention times
        self.t1 = self.n_timesteps // 4
        self.t2 = self.n_timesteps // 2
        
        # Initialize field (random initial conditions)
        self.init_field = self._get_initial_field()
        
        # Precompute momentum grid
        self.k_grid = 2 * np.pi * np.fft.fftfreq(self.n_sites, d=self.dx)
        
        # Compute dispersion relation
        self.omega_k = self._compute_dispersion_relation()
        
        # Store effective metric data
        self.effective_metric_data = self._compute_effective_metric_data()
        
    def _get_initial_field(self) -> np.ndarray:
        """Get initial field configuration."""
        # Gaussian wave packet with random phase
        x = np.arange(self.n_sites) * self.dx
        x0 = self.n_sites * self.dx / 2
        sigma = self.n_sites * self.dx / 10
        
        phase = self.rng.uniform(0, 2 * np.pi)
        field = np.exp(-((x - x0) ** 2) / (2 * sigma ** 2)) * np.exp(1j * phase)
        
        # Add small random perturbations
        field += 0.01 * (self.rng.randn(self.n_sites) + 1j * self.rng.randn(self.n_sites))
        
        return field
    
    def _compute_dispersion_relation(self) -> np.ndarray:
        """
        Compute dispersion relation ω(k) with tunable dispersion.
        
        ω(k) = c|k| * (1 + α|k|²) with cutoff
        """
        k_abs = np.abs(self.k_grid)
        
        # Apply cutoff
        k_eff = np.where(k_abs < self.cutoff, k_abs, self.cutoff)
        
        # Dispersion relation
        omega = k_eff * (1.0 + self.dispersion * k_eff ** 2)
        
        return omega
    
    def _compute_effective_metric_data(self) -> Dict[str, Any]:
        """
        Compute effective metric data from dispersion relation.
        
        Returns:
            Dictionary with metric reconstruction data
        """
        # Extract effective light speed from dispersion
        k_small = self.k_grid[1] if len(self.k_grid) > 1 else 1.0
        c_eff = self.omega_k[1] / (k_small + 1e-12) if abs(k_small) > 1e-12 else 1.0
        
        # Compute dispersion deviation
        linear_omega = c_eff * np.abs(self.k_grid)
        dispersion_deviation = np.mean(np.abs(self.omega_k - linear_omega)) / (c_eff + 1e-10)
        
        # Attenuation from dissipation
        attenuation_margin = 1.0 / (self.dissipation + 1e-10)
        
        # Reconstruction error (deviation from perfect linear dispersion)
        if np.max(np.abs(self.omega_k)) > 0:
            reconstruction_error = dispersion_deviation
        else:
            reconstruction_error = 1.0
        
        return {
            'c_eff': c_eff,
            'dispersion_deviation': dispersion_deviation,
            'attenuation_margin': attenuation_margin,
            'reconstruction_error': reconstruction_error,
            'flow_velocity': self.flow_vel,
        }
    
    def _apply_intervention(self, field: np.ndarray, intervention: int) -> np.ndarray:
        """
        Apply intervention at t2.
        
        Args:
            field: Field before intervention
            intervention: 0 (no intervention) or 1 (apply local perturbation)
            
        Returns:
            Field after intervention
        """
        if intervention == 0:
            return field
        else:
            # Apply localized perturbation at center
            perturb = field.copy()
            center = self.n_sites // 2
            width = max(1, self.n_sites // 20)
            
            # Add Gaussian perturbation
            for i in range(center - width, center + width):
                if 0 <= i < self.n_sites:
                    perturb[i] += 0.1 * np.exp(-((i - center) ** 2) / (2 * width ** 2))
            
            return perturb
    
    def _evolve_step(self, field: np.ndarray) -> np.ndarray:
        """
        Evolve field one timestep using spectral method.
        
        ∂ψ/∂t = -iω(k)ψ(k) - γψ(k) + flow term
        """
        # Transform to Fourier space
        field_k = fft(field)
        
        # Evolution in Fourier space
        # Dispersion evolution
        evolution = np.exp(-1j * self.omega_k * self.dt)
        
        # Dissipation
        damping = np.exp(-self.dissipation * self.dt)
        
        # Flow term (Doppler shift)
        phase_shift = np.exp(-1j * self.flow_vel * self.k_grid * self.dt)
        
        # Combined evolution
        field_k = field_k * evolution * damping * phase_shift
        
        # Transform back to real space
        field_new = ifft(field_k)
        
        return field_new
    
    def run_simulation(self, intervention: int) -> Dict[str, Any]:
        """
        Run lattice field simulation.
        
        Args:
            intervention: Intervention value at t2 (0 or 1)
            
        Returns:
            Simulation results
        """
        field = self.init_field.copy()
        trajectory = [field.copy()]
        observables = []
        
        for t in range(self.n_timesteps):
            # Apply intervention at t2
            if t == self.t2:
                field = self._apply_intervention(field, intervention)
            
            # Evolve one timestep
            field = self._evolve_step(field)
            trajectory.append(field.copy())
            
            # Measure observable (total energy at boundary)
            # Use boundary sites to avoid direct intervention effect
            obs = np.real(np.abs(field[0]) ** 2 + np.abs(field[-1]) ** 2)
            observables.append(obs)
        
        return {
            'observables_t1': observables[self.t1] if self.t1 < len(observables) else observables[-1],
            'full_trajectory': trajectory,
            'observables_full': observables,
            'effective_metric_data': self.effective_metric_data,
            'metadata': {
                'n_sites': self.n_sites,
                'cutoff': self.cutoff,
                'dispersion': self.dispersion,
                'flow_velocity': self.flow_vel,
                'dissipation': self.dissipation,
                't1': self.t1,
                't2': self.t2,
                'intervention': intervention,
            }
        }
    
    def get_observable_t1(self, results: Dict[str, Any]) -> float:
        """
        Extract observable at t1.
        
        CRITICAL: This function must NOT access data after t1.
        """
        return results['observables_t1']
