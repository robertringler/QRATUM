"""GPU Shader Management for QRATUM-Chess.

Provides shader compilation, caching, and management for
high-performance GPU rendering.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ShaderType(Enum):
    """Shader types."""

    VERTEX = "vertex"
    FRAGMENT = "fragment"
    GEOMETRY = "geometry"
    COMPUTE = "compute"


@dataclass
class ShaderProgram:
    """Compiled shader program.

    Attributes:
        name: Shader program name
        handle: GPU program handle
        uniforms: Uniform variable locations
        attributes: Attribute locations
    """

    name: str
    handle: int = 0
    uniforms: dict[str, int] = None
    attributes: dict[str, int] = None

    def __post_init__(self):
        self.uniforms = self.uniforms or {}
        self.attributes = self.attributes or {}


# Vertex shader for board rendering
BOARD_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec3 aPosition;
layout(location = 1) in vec2 aTexCoord;
layout(location = 2) in vec4 aColor;

uniform mat4 uProjection;
uniform mat4 uView;
uniform mat4 uModel;

out vec2 vTexCoord;
out vec4 vColor;

void main() {
    gl_Position = uProjection * uView * uModel * vec4(aPosition, 1.0);
    vTexCoord = aTexCoord;
    vColor = aColor;
}
"""

# Fragment shader for board rendering
BOARD_FRAGMENT_SHADER = """
#version 330 core

in vec2 vTexCoord;
in vec4 vColor;

uniform sampler2D uTexture;
uniform bool uUseTexture;
uniform vec4 uHighlightColor;
uniform float uHighlightIntensity;

out vec4 FragColor;

void main() {
    vec4 color;
    if (uUseTexture) {
        color = texture(uTexture, vTexCoord) * vColor;
    } else {
        color = vColor;
    }
    
    // Apply highlight
    if (uHighlightIntensity > 0.0) {
        color = mix(color, uHighlightColor, uHighlightIntensity * 0.5);
    }
    
    FragColor = color;
}
"""

# Heatmap shader
HEATMAP_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 aPosition;
layout(location = 1) in float aValue;

uniform mat4 uProjection;
uniform vec2 uOffset;
uniform vec2 uScale;

out float vValue;

void main() {
    vec2 pos = aPosition * uScale + uOffset;
    gl_Position = uProjection * vec4(pos, 0.0, 1.0);
    vValue = aValue;
}
"""

HEATMAP_FRAGMENT_SHADER = """
#version 330 core

in float vValue;

uniform vec3 uColorLow;
uniform vec3 uColorHigh;
uniform float uOpacity;

out vec4 FragColor;

void main() {
    vec3 color = mix(uColorLow, uColorHigh, vValue);
    FragColor = vec4(color, uOpacity * vValue);
}
"""

# Tree node shader
TREE_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 aPosition;
layout(location = 1) in float aSize;
layout(location = 2) in vec4 aColor;

uniform mat4 uProjection;
uniform vec2 uPan;
uniform float uZoom;

out vec4 vColor;
out float vSize;

void main() {
    vec2 pos = (aPosition + uPan) * uZoom;
    gl_Position = uProjection * vec4(pos, 0.0, 1.0);
    gl_PointSize = aSize * uZoom;
    vColor = aColor;
    vSize = aSize;
}
"""

TREE_FRAGMENT_SHADER = """
#version 330 core

in vec4 vColor;
in float vSize;

out vec4 FragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    
    if (dist > 0.5) {
        discard;
    }
    
    // Soft edges
    float alpha = 1.0 - smoothstep(0.4, 0.5, dist);
    FragColor = vec4(vColor.rgb, vColor.a * alpha);
}
"""

# Quantum amplitude shader
QUANTUM_VERTEX_SHADER = """
#version 330 core

layout(location = 0) in vec2 aPosition;
layout(location = 1) in vec2 aAmplitude;  // (magnitude, phase)

uniform mat4 uProjection;
uniform float uBarWidth;

out vec4 vColor;

// HSV to RGB conversion
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    gl_Position = uProjection * vec4(aPosition, 0.0, 1.0);
    
    // Color based on phase
    float hue = (aAmplitude.y + 3.14159) / (2.0 * 3.14159);
    float saturation = 0.8;
    float value = 0.9;
    vColor = vec4(hsv2rgb(vec3(hue, saturation, value)), 1.0);
}
"""

QUANTUM_FRAGMENT_SHADER = """
#version 330 core

in vec4 vColor;

out vec4 FragColor;

void main() {
    FragColor = vColor;
}
"""

# Anti-holographic overlay shader
ANTI_HOLOGRAPHIC_FRAGMENT_SHADER = """
#version 330 core

in vec2 vTexCoord;

uniform sampler2D uBaseTexture;
uniform float uStochasticity;
uniform float uDestabilization;
uniform float uTime;

out vec4 FragColor;

// Simple noise function
float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec4 base = texture(uBaseTexture, vTexCoord);
    
    // Add stochastic noise
    float noiseVal = noise(vTexCoord * 100.0 + vec2(uTime));
    float noiseIntensity = uStochasticity * 0.1;
    vec3 color = base.rgb + vec3(noiseVal - 0.5) * noiseIntensity;
    
    // Add destabilization edge glow
    vec2 uv = vTexCoord * 2.0 - 1.0;
    float edgeDist = max(abs(uv.x), abs(uv.y));
    float edgeGlow = smoothstep(0.9, 1.0, edgeDist) * uDestabilization;
    color = mix(color, vec3(1.0, 0.0, 0.43), edgeGlow);
    
    FragColor = vec4(color, base.a);
}
"""


class ShaderManager:
    """Manages shader compilation and caching.

    Provides:
    - Shader compilation from source
    - Program linking
    - Uniform and attribute location caching
    - Hot-reloading support
    """

    # Built-in shader sources
    BUILTIN_SHADERS = {
        "board_vertex": BOARD_VERTEX_SHADER,
        "board_fragment": BOARD_FRAGMENT_SHADER,
        "heatmap_vertex": HEATMAP_VERTEX_SHADER,
        "heatmap_fragment": HEATMAP_FRAGMENT_SHADER,
        "tree_vertex": TREE_VERTEX_SHADER,
        "tree_fragment": TREE_FRAGMENT_SHADER,
        "quantum_vertex": QUANTUM_VERTEX_SHADER,
        "quantum_fragment": QUANTUM_FRAGMENT_SHADER,
        "anti_holographic_fragment": ANTI_HOLOGRAPHIC_FRAGMENT_SHADER,
    }

    def __init__(self) -> None:
        """Initialize shader manager."""
        self._programs: dict[str, ShaderProgram] = {}
        self._compiled_shaders: dict[str, int] = {}

    def compile_shader(self, source: str, shader_type: ShaderType) -> int:
        """Compile a shader from source.

        Args:
            source: GLSL shader source
            shader_type: Type of shader

        Returns:
            Shader handle (simulated)
        """
        # In production, this would use actual OpenGL/Vulkan calls
        # For now, return a simulated handle
        key = f"{shader_type.value}_{hash(source)}"
        if key not in self._compiled_shaders:
            self._compiled_shaders[key] = len(self._compiled_shaders) + 1
        return self._compiled_shaders[key]

    def create_program(
        self,
        name: str,
        vertex_source: str,
        fragment_source: str,
        geometry_source: str | None = None,
    ) -> ShaderProgram:
        """Create a shader program.

        Args:
            name: Program name
            vertex_source: Vertex shader source
            fragment_source: Fragment shader source
            geometry_source: Optional geometry shader source

        Returns:
            Compiled shader program
        """
        # Compile shaders
        vertex_shader = self.compile_shader(vertex_source, ShaderType.VERTEX)
        fragment_shader = self.compile_shader(fragment_source, ShaderType.FRAGMENT)

        geometry_shader = None
        if geometry_source:
            geometry_shader = self.compile_shader(geometry_source, ShaderType.GEOMETRY)

        # Create program (simulated)
        program = ShaderProgram(
            name=name,
            handle=len(self._programs) + 1,
        )

        # Parse and cache uniform/attribute locations from source
        program.uniforms = self._parse_uniforms(vertex_source + fragment_source)
        program.attributes = self._parse_attributes(vertex_source)

        self._programs[name] = program
        return program

    def get_program(self, name: str) -> ShaderProgram | None:
        """Get a compiled program by name.

        Args:
            name: Program name

        Returns:
            Shader program or None
        """
        return self._programs.get(name)

    def create_builtin_programs(self) -> None:
        """Create all built-in shader programs."""
        # Board shader
        self.create_program(
            "board",
            self.BUILTIN_SHADERS["board_vertex"],
            self.BUILTIN_SHADERS["board_fragment"],
        )

        # Heatmap shader
        self.create_program(
            "heatmap",
            self.BUILTIN_SHADERS["heatmap_vertex"],
            self.BUILTIN_SHADERS["heatmap_fragment"],
        )

        # Tree shader
        self.create_program(
            "tree",
            self.BUILTIN_SHADERS["tree_vertex"],
            self.BUILTIN_SHADERS["tree_fragment"],
        )

        # Quantum shader
        self.create_program(
            "quantum",
            self.BUILTIN_SHADERS["quantum_vertex"],
            self.BUILTIN_SHADERS["quantum_fragment"],
        )

    def _parse_uniforms(self, source: str) -> dict[str, int]:
        """Parse uniform declarations from shader source.

        Args:
            source: Shader source code

        Returns:
            Dictionary of uniform names to locations
        """
        uniforms = {}
        location = 0

        for line in source.split("\n"):
            line = line.strip()
            if line.startswith("uniform "):
                # Extract uniform name
                parts = line.rstrip(";").split()
                if len(parts) >= 3:
                    name = parts[-1]
                    uniforms[name] = location
                    location += 1

        return uniforms

    def _parse_attributes(self, source: str) -> dict[str, int]:
        """Parse attribute declarations from shader source.

        Args:
            source: Vertex shader source code

        Returns:
            Dictionary of attribute names to locations
        """
        attributes = {}

        for line in source.split("\n"):
            line = line.strip()
            if "layout(location" in line and "in " in line:
                # Extract location and name
                try:
                    loc_start = line.find("location = ") + 11
                    loc_end = line.find(")", loc_start)
                    location = int(line[loc_start:loc_end])

                    name_start = line.rfind(" ") + 1
                    name = line[name_start:].rstrip(";")

                    attributes[name] = location
                except (ValueError, IndexError):
                    pass

        return attributes

    def delete_program(self, name: str) -> None:
        """Delete a shader program.

        Args:
            name: Program name
        """
        if name in self._programs:
            del self._programs[name]

    def clear(self) -> None:
        """Clear all programs and shaders."""
        self._programs.clear()
        self._compiled_shaders.clear()
