"""
Global Seed Authority for QRATUM

Single source of truth for deterministic seeding across all XENON components.
Certificate: QRATUM-HARDENING-20251215-V5
"""

# Global seed locked for production
GLOBAL_SEED = 42


def get_global_seed() -> int:
    """
    Get the global seed for reproducible execution.

    Returns:
        int: The global seed value (42 for production)
    """
    return GLOBAL_SEED
