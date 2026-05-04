"""Lighting and material system."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class LightType(Enum):
    """Types of lights."""

    DIRECTIONAL = "directional"
    POINT = "point"
    SPOT = "spot"
    AMBIENT = "ambient"


@dataclass
class Light:
    """Light source.

    Attributes:
        type: Type of light
        position: Light position (for point/spot lights)
        direction: Light direction (for directional/spot lights)
        color: Light color as RGB
        intensity: Light intensity
        radius: Light radius (for point/spot lights)
        angle: Spot light cone angle in degrees
    """

    type: LightType = LightType.DIRECTIONAL
    position: np.ndarray = None
    direction: np.ndarray = None
    color: np.ndarray = None
    intensity: float = 1.0
    radius: float = 10.0
    angle: float = 45.0

    def __post_init__(self) -> None:
        """Initialize default values."""

        if self.position is None:
            self.position = np.array([0.0, 10.0, 0.0])
        if self.direction is None:
            self.direction = np.array([0.0, -1.0, 0.0])
        if self.color is None:
            self.color = np.array([1.0, 1.0, 1.0])

    def get_illumination(self, point: np.ndarray, normal: np.ndarray) -> float:
        """Calculate illumination at a point.

        Args:
            point: Point position
            normal: Surface normal at point

        Returns:
            Illumination intensity
        """

        if self.type == LightType.AMBIENT:
            return self.intensity

        if self.type == LightType.DIRECTIONAL:
            light_dir = -self.direction / np.linalg.norm(self.direction)
            diffuse = max(0.0, np.dot(normal, light_dir))
            return self.intensity * diffuse

        if self.type == LightType.POINT:
            light_dir = self.position - point
            distance = np.linalg.norm(light_dir)
            light_dir = light_dir / distance

            # Attenuation
            attenuation = 1.0 / (1.0 + distance / self.radius)
            diffuse = max(0.0, np.dot(normal, light_dir))

            return self.intensity * diffuse * attenuation

        return 0.0


@dataclass
class PBRMaterial:
    """Physically-based rendering material.

    Attributes:
        albedo: Base color as RGB
        metallic: Metallic factor (0-1)
        roughness: Roughness factor (0-1)
        ao: Ambient occlusion (0-1)
        emissive: Emissive color as RGB
        emissive_strength: Emissive intensity
        alpha: Transparency (0-1, 1=opaque)
    """

    albedo: np.ndarray = None
    metallic: float = 0.0
    roughness: float = 0.5
    ao: float = 1.0
    emissive: np.ndarray = None
    emissive_strength: float = 0.0
    alpha: float = 1.0

    def __post_init__(self) -> None:
        """Initialize default values."""

        if self.albedo is None:
            self.albedo = np.array([0.8, 0.8, 0.8])
        if self.emissive is None:
            self.emissive = np.array([0.0, 0.0, 0.0])

    def compute_shading(
        self, normal: np.ndarray, view_dir: np.ndarray, light_dir: np.ndarray
    ) -> np.ndarray:
        """Compute PBR shading.

        Args:
            normal: Surface normal
            view_dir: View direction
            light_dir: Light direction

        Returns:
            Shaded color as RGB
        """

        # Normalize vectors
        N = normal / np.linalg.norm(normal)
        V = view_dir / np.linalg.norm(view_dir)
        L = light_dir / np.linalg.norm(light_dir)
        H = (V + L) / np.linalg.norm(V + L)

        # Diffuse
        NdotL = max(0.0, np.dot(N, L))
        diffuse = self.albedo * NdotL

        # Specular (simplified)
        NdotH = max(0.0, np.dot(N, H))
        specular = (1.0 - self.roughness) * (NdotH ** (1.0 / (self.roughness + 0.001)))

        # Combine
        color = diffuse + specular * (1.0 - self.metallic) + self.emissive * self.emissive_strength

        return np.clip(color, 0.0, 1.0)
