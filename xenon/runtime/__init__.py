"""XENON runtime (meta-kernel).

The main execution loop that orchestrates hypothesis generation,
simulation, experiment selection, and Bayesian updating.
"""

from .xenon_kernel import XENONRuntime

__all__ = ["XENONRuntime"]
