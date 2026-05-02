"""WebXR VR/AR components."""

from .haptic_engine import HapticConfig, HapticEngine, HapticFeedback
from .vr_controller import VRConfig, VRController

__all__ = [
    "VRController",
    "VRConfig",
    "HapticEngine",
    "HapticConfig",
    "HapticFeedback",
]
