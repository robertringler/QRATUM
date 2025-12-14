"""Camera and camera control system."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Camera:
    """3D camera with projection settings.

    Attributes:
        position: Camera position in world space
        target: Point camera is looking at
        up: Up vector
        fov: Field of view in degrees
        aspect_ratio: Aspect ratio (width/height)
        near: Near clipping plane
        far: Far clipping plane
    """

    position: np.ndarray = None
    target: np.ndarray = None
    up: np.ndarray = None
    fov: float = 45.0
    aspect_ratio: float = 16.0 / 9.0
    near: float = 0.1
    far: float = 1000.0

    def __post_init__(self) -> None:
        """Initialize default values."""
        if self.position is None:
            self.position = np.array([5.0, 5.0, 5.0])
        if self.target is None:
            self.target = np.array([0.0, 0.0, 0.0])
        if self.up is None:
            self.up = np.array([0.0, 1.0, 0.0])

    def get_view_matrix(self) -> np.ndarray:
        """Get view matrix for camera transformation.

        Returns:
            4x4 view matrix
        """
        # Calculate camera basis vectors
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        # Build view matrix
        view = np.eye(4)
        view[0, :3] = right
        view[1, :3] = up
        view[2, :3] = -forward
        view[:3, 3] = -np.array([
            np.dot(right, self.position),
            np.dot(up, self.position),
            np.dot(-forward, self.position),
        ])

        return view

    def get_projection_matrix(self) -> np.ndarray:
        """Get projection matrix.

        Returns:
            4x4 projection matrix
        """
        fov_rad = np.radians(self.fov)
        f = 1.0 / np.tan(fov_rad / 2.0)

        proj = np.zeros((4, 4))
        proj[0, 0] = f / self.aspect_ratio
        proj[1, 1] = f
        proj[2, 2] = (self.far + self.near) / (self.near - self.far)
        proj[2, 3] = (2.0 * self.far * self.near) / (self.near - self.far)
        proj[3, 2] = -1.0

        return proj

    def look_at(self, target: np.ndarray) -> None:
        """Point camera at target.

        Args:
            target: Target position
        """
        self.target = target

    def orbit(self, angle_x: float, angle_y: float, distance: Optional[float] = None) -> None:
        """Orbit camera around target.

        Args:
            angle_x: Horizontal angle in radians
            angle_y: Vertical angle in radians
            distance: Distance from target (None to keep current)
        """
        if distance is None:
            distance = np.linalg.norm(self.position - self.target)

        # Compute new position
        x = distance * np.cos(angle_y) * np.sin(angle_x)
        y = distance * np.sin(angle_y)
        z = distance * np.cos(angle_y) * np.cos(angle_x)

        self.position = self.target + np.array([x, y, z])


class CameraController:
    """Interactive camera controller.

    Args:
        camera: Camera to control
    """

    def __init__(self, camera: Camera) -> None:
        """Initialize camera controller."""
        self.camera = camera
        self.orbit_speed = 0.01
        self.zoom_speed = 0.1
        self.pan_speed = 0.01

    def rotate(self, dx: float, dy: float) -> None:
        """Rotate camera around target.

        Args:
            dx: Horizontal movement
            dy: Vertical movement
        """
        # Calculate current angles
        rel_pos = self.camera.position - self.camera.target
        distance = np.linalg.norm(rel_pos)

        angle_x = np.arctan2(rel_pos[0], rel_pos[2])
        angle_y = np.arcsin(rel_pos[1] / distance)

        # Apply rotation
        angle_x += dx * self.orbit_speed
        angle_y += dy * self.orbit_speed

        # Clamp vertical angle
        angle_y = np.clip(angle_y, -np.pi / 2 + 0.01, np.pi / 2 - 0.01)

        # Update camera
        self.camera.orbit(angle_x, angle_y, distance)

    def zoom(self, delta: float) -> None:
        """Zoom camera in/out.

        Args:
            delta: Zoom amount
        """
        rel_pos = self.camera.position - self.camera.target
        distance = np.linalg.norm(rel_pos)
        new_distance = max(0.1, distance * (1.0 - delta * self.zoom_speed))

        direction = rel_pos / distance
        self.camera.position = self.camera.target + direction * new_distance

    def pan(self, dx: float, dy: float) -> None:
        """Pan camera parallel to view plane.

        Args:
            dx: Horizontal movement
            dy: Vertical movement
        """
        forward = self.camera.target - self.camera.position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(forward, self.camera.up)
        right = right / np.linalg.norm(right)

        up = np.cross(right, forward)

        offset = right * dx * self.pan_speed + up * dy * self.pan_speed
        self.camera.position += offset
        self.camera.target += offset
