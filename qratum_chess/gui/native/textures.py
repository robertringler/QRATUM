"""Texture Management for QRATUM-Chess GPU rendering.

Handles texture atlas creation, management, and heatmap rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class TextureFormat(Enum):
    """Texture formats."""

    RGBA8 = "rgba8"
    RGB8 = "rgb8"
    R8 = "r8"
    RG16F = "rg16f"
    RGBA16F = "rgba16f"
    RGBA32F = "rgba32f"
    DEPTH24 = "depth24"
    DEPTH32F = "depth32f"


class TextureFilter(Enum):
    """Texture filtering modes."""

    NEAREST = "nearest"
    LINEAR = "linear"
    LINEAR_MIPMAP = "linear_mipmap"


class TextureWrap(Enum):
    """Texture wrap modes."""

    REPEAT = "repeat"
    CLAMP = "clamp"
    MIRROR = "mirror"


@dataclass
class TextureSpec:
    """Texture specification.

    Attributes:
        width: Texture width
        height: Texture height
        format: Pixel format
        filter_mode: Filtering mode
        wrap_mode: Wrap mode
        generate_mipmaps: Whether to generate mipmaps
    """

    width: int
    height: int
    format: TextureFormat = TextureFormat.RGBA8
    filter_mode: TextureFilter = TextureFilter.LINEAR
    wrap_mode: TextureWrap = TextureWrap.CLAMP
    generate_mipmaps: bool = False


@dataclass
class Texture:
    """GPU texture representation.

    Attributes:
        name: Texture name
        spec: Texture specification
        handle: GPU texture handle
        data: CPU-side data copy
    """

    name: str
    spec: TextureSpec
    handle: int = 0
    data: np.ndarray | None = None

    @property
    def is_valid(self) -> bool:
        """Check if texture has valid GPU handle."""
        return self.handle > 0


@dataclass
class AtlasRegion:
    """Region within a texture atlas.

    Attributes:
        name: Region name
        x: X offset in pixels
        y: Y offset in pixels
        width: Region width
        height: Region height
        u0, v0, u1, v1: Texture coordinates
    """

    name: str
    x: int
    y: int
    width: int
    height: int

    @property
    def u0(self) -> float:
        return 0.0  # Set by atlas

    @property
    def v0(self) -> float:
        return 0.0

    @property
    def u1(self) -> float:
        return 1.0

    @property
    def v1(self) -> float:
        return 1.0


class TextureAtlas:
    """Texture atlas for efficient batch rendering.

    Packs multiple small textures into a single large texture
    for reduced draw calls.
    """

    def __init__(self, width: int = 4096, height: int = 4096) -> None:
        """Initialize texture atlas.

        Args:
            width: Atlas width
            height: Atlas height
        """
        self.width = width
        self.height = height

        self._texture: Texture | None = None
        self._regions: dict[str, AtlasRegion] = {}
        self._data = np.zeros((height, width, 4), dtype=np.uint8)

        # Simple packing state
        self._pack_x = 0
        self._pack_y = 0
        self._row_height = 0

    def create(self) -> None:
        """Create the atlas texture."""
        self._texture = Texture(
            name="atlas",
            spec=TextureSpec(
                width=self.width,
                height=self.height,
                format=TextureFormat.RGBA8,
                filter_mode=TextureFilter.LINEAR,
                wrap_mode=TextureWrap.CLAMP,
            ),
            handle=1,
            data=self._data,
        )

    def add_region(self, name: str, image: np.ndarray) -> AtlasRegion | None:
        """Add an image to the atlas.

        Args:
            name: Region name
            image: RGBA image data

        Returns:
            Atlas region or None if failed
        """
        h, w = image.shape[:2]

        # Simple row-based packing
        if self._pack_x + w > self.width:
            self._pack_x = 0
            self._pack_y += self._row_height
            self._row_height = 0

        if self._pack_y + h > self.height:
            return None  # Atlas full

        # Copy image data
        self._data[self._pack_y : self._pack_y + h, self._pack_x : self._pack_x + w] = image

        # Create region
        region = AtlasRegion(
            name=name,
            x=self._pack_x,
            y=self._pack_y,
            width=w,
            height=h,
        )

        self._regions[name] = region

        # Update packing state
        self._pack_x += w
        self._row_height = max(self._row_height, h)

        return region

    def get_region(self, name: str) -> AtlasRegion | None:
        """Get a region by name.

        Args:
            name: Region name

        Returns:
            Atlas region or None
        """
        return self._regions.get(name)

    def get_uv(self, name: str) -> tuple[float, float, float, float] | None:
        """Get UV coordinates for a region.

        Args:
            name: Region name

        Returns:
            (u0, v0, u1, v1) or None
        """
        region = self._regions.get(name)
        if not region:
            return None

        u0 = region.x / self.width
        v0 = region.y / self.height
        u1 = (region.x + region.width) / self.width
        v1 = (region.y + region.height) / self.height

        return (u0, v0, u1, v1)

    @property
    def texture(self) -> Texture | None:
        """Get the atlas texture."""
        return self._texture

    def clear(self) -> None:
        """Clear the atlas."""
        self._data.fill(0)
        self._regions.clear()
        self._pack_x = 0
        self._pack_y = 0
        self._row_height = 0


class TextureManager:
    """Manages texture creation and lifetime.

    Features:
    - Texture creation and deletion
    - Texture atlas management
    - Heatmap texture generation
    - Colormap application
    """

    # Built-in colormaps
    COLORMAPS = {
        "viridis": [
            (68, 1, 84),
            (72, 40, 120),
            (62, 74, 137),
            (49, 104, 142),
            (38, 130, 142),
            (31, 158, 137),
            (53, 183, 121),
            (109, 205, 89),
            (180, 222, 44),
            (253, 231, 37),
        ],
        "hot": [
            (0, 0, 0),
            (87, 0, 0),
            (175, 0, 0),
            (255, 55, 0),
            (255, 125, 0),
            (255, 195, 0),
            (255, 255, 50),
            (255, 255, 150),
            (255, 255, 200),
            (255, 255, 255),
        ],
        "cool": [
            (0, 255, 255),
            (25, 230, 255),
            (50, 205, 255),
            (75, 180, 255),
            (100, 155, 255),
            (125, 130, 255),
            (150, 105, 255),
            (175, 80, 255),
            (200, 55, 255),
            (225, 30, 255),
        ],
        "plasma": [
            (13, 8, 135),
            (75, 3, 161),
            (125, 3, 168),
            (168, 34, 150),
            (203, 70, 121),
            (229, 107, 93),
            (248, 148, 65),
            (253, 195, 40),
            (240, 249, 33),
            (240, 249, 33),
        ],
    }

    def __init__(self, max_textures: int = 256) -> None:
        """Initialize texture manager.

        Args:
            max_textures: Maximum number of textures
        """
        self.max_textures = max_textures
        self._textures: dict[str, Texture] = {}
        self._next_handle = 1

        # Create default atlas
        self.atlas = TextureAtlas()

    def create_texture(
        self,
        name: str,
        spec: TextureSpec,
        data: np.ndarray | None = None,
    ) -> Texture:
        """Create a new texture.

        Args:
            name: Texture name
            spec: Texture specification
            data: Initial pixel data

        Returns:
            Created texture
        """
        if name in self._textures:
            self.delete_texture(name)

        texture = Texture(
            name=name,
            spec=spec,
            handle=self._next_handle,
            data=data.copy() if data is not None else None,
        )

        self._textures[name] = texture
        self._next_handle += 1

        return texture

    def create_heatmap_texture(
        self,
        name: str,
        values: np.ndarray,
        colormap: str = "viridis",
        size: int = 64,
    ) -> Texture:
        """Create a heatmap texture from values.

        Args:
            name: Texture name
            values: 2D array of values (0-1)
            colormap: Colormap name
            size: Output texture size per cell

        Returns:
            Created heatmap texture
        """
        h, w = values.shape

        # Get colormap
        cmap = self.COLORMAPS.get(colormap, self.COLORMAPS["viridis"])

        # Create color buffer
        buffer = np.zeros((h * size, w * size, 4), dtype=np.uint8)

        for y in range(h):
            for x in range(w):
                value = np.clip(values[y, x], 0, 1)

                # Interpolate colormap
                idx = value * (len(cmap) - 1)
                idx_low = int(idx)
                idx_high = min(idx_low + 1, len(cmap) - 1)
                t = idx - idx_low

                color_low = cmap[idx_low]
                color_high = cmap[idx_high]

                r = int(color_low[0] + t * (color_high[0] - color_low[0]))
                g = int(color_low[1] + t * (color_high[1] - color_low[1]))
                b = int(color_low[2] + t * (color_high[2] - color_low[2]))
                a = int(value * 200 + 55)

                # Fill cell
                y_start = y * size
                x_start = x * size
                buffer[y_start : y_start + size, x_start : x_start + size] = [r, g, b, a]

        spec = TextureSpec(
            width=w * size,
            height=h * size,
            format=TextureFormat.RGBA8,
            filter_mode=TextureFilter.LINEAR,
        )

        return self.create_texture(name, spec, buffer)

    def create_colormap_texture(self, colormap: str = "viridis", width: int = 256) -> Texture:
        """Create a 1D colormap texture.

        Args:
            colormap: Colormap name
            width: Texture width

        Returns:
            Created colormap texture
        """
        cmap = self.COLORMAPS.get(colormap, self.COLORMAPS["viridis"])

        buffer = np.zeros((1, width, 4), dtype=np.uint8)

        for x in range(width):
            t = x / (width - 1)
            idx = t * (len(cmap) - 1)
            idx_low = int(idx)
            idx_high = min(idx_low + 1, len(cmap) - 1)
            frac = idx - idx_low

            color_low = cmap[idx_low]
            color_high = cmap[idx_high]

            r = int(color_low[0] + frac * (color_high[0] - color_low[0]))
            g = int(color_low[1] + frac * (color_high[1] - color_low[1]))
            b = int(color_low[2] + frac * (color_high[2] - color_low[2]))

            buffer[0, x] = [r, g, b, 255]

        spec = TextureSpec(
            width=width,
            height=1,
            format=TextureFormat.RGBA8,
            filter_mode=TextureFilter.LINEAR,
            wrap_mode=TextureWrap.CLAMP,
        )

        return self.create_texture(f"colormap_{colormap}", spec, buffer)

    def update_texture(self, name: str, data: np.ndarray, x: int = 0, y: int = 0) -> bool:
        """Update texture data.

        Args:
            name: Texture name
            data: New pixel data
            x: X offset
            y: Y offset

        Returns:
            True if update successful
        """
        texture = self._textures.get(name)
        if not texture or texture.data is None:
            return False

        h, w = data.shape[:2]
        if y + h > texture.spec.height or x + w > texture.spec.width:
            return False

        texture.data[y : y + h, x : x + w] = data
        return True

    def get_texture(self, name: str) -> Texture | None:
        """Get a texture by name.

        Args:
            name: Texture name

        Returns:
            Texture or None
        """
        return self._textures.get(name)

    def delete_texture(self, name: str) -> bool:
        """Delete a texture.

        Args:
            name: Texture name

        Returns:
            True if deleted
        """
        if name in self._textures:
            del self._textures[name]
            return True
        return False

    def clear(self) -> None:
        """Delete all textures."""
        self._textures.clear()
        self.atlas.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get texture usage statistics.

        Returns:
            Dictionary with stats
        """
        total_memory = 0
        for tex in self._textures.values():
            if tex.data is not None:
                total_memory += tex.data.nbytes

        return {
            "texture_count": len(self._textures),
            "total_memory_bytes": total_memory,
            "total_memory_mb": total_memory / (1024 * 1024),
            "atlas_size": (self.atlas.width, self.atlas.height),
            "atlas_regions": len(self.atlas._regions),
        }
