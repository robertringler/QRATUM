"""Base model interface."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import numpy as np


class BaseModel(ABC):
    """Base class for all simulation models."""

    def __init__(self, config: Dict[str, Any], seed: int):
        """
        Initialize model.
        
        Args:
            config: Model configuration
            seed: Random seed
        """
        self.config = config
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    @abstractmethod
    def run_simulation(self, intervention: int) -> Dict[str, Any]:
        """
        Run simulation with specified intervention.
        
        Args:
            intervention: Intervention value (0 or 1) applied at t2
            
        Returns:
            Dictionary containing simulation results including:
            - observables_t1: Observables measured at t1
            - full_trajectory: Complete system trajectory
            - metadata: Additional simulation metadata
        """
        pass

    @abstractmethod
    def get_observable_t1(self, results: Dict[str, Any]) -> float:
        """
        Extract observable at t1 from simulation results.
        
        This function MUST NOT access data from times > t1 (audit layer enforces this).
        
        Args:
            results: Simulation results
            
        Returns:
            Observable value at t1
        """
        pass

    def run_twin_simulations(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Run twin simulations with I=0 and I=1.
        
        Uses same seed to ensure identical initial conditions up to t2.
        
        Returns:
            Tuple of (results_I0, results_I1)
        """
        # Reset RNG to same state for both runs
        self.rng = np.random.RandomState(self.seed)
        results_I0 = self.run_simulation(intervention=0)

        self.rng = np.random.RandomState(self.seed)
        results_I1 = self.run_simulation(intervention=1)

        return results_I0, results_I1
