"""3Dmol.js-based molecular viewer with interactive 3D capabilities.

Provides a Python backend for generating 3Dmol.js compatible visualization
configurations and rendering molecular structures in web browsers.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .pdb_loader import PDBStructure

logger = logging.getLogger(__name__)


class RenderStyle(Enum):
    """Molecular rendering styles."""

    CARTOON = "cartoon"
    STICK = "stick"
    SPHERE = "sphere"
    LINE = "line"
    CROSS = "cross"
    SURFACE = "surface"
    RIBBON = "ribbon"
    TRACE = "trace"


class ColorScheme(Enum):
    """Molecular coloring schemes."""

    ELEMENT = "element"
    CHAIN = "chain"
    RESIDUE = "residue"
    SECONDARY_STRUCTURE = "ss"
    BFACTOR = "b"
    SPECTRUM = "spectrum"
    CUSTOM = "custom"


class SurfaceType(Enum):
    """Molecular surface types."""

    VDW = "VDW"
    SAS = "SAS"  # Solvent Accessible Surface
    SES = "SES"  # Solvent Excluded Surface
    MS = "MS"  # Molecular Surface


@dataclass
class ViewerConfig:
    """Configuration for molecular viewer."""

    background_color: str = "#000000"
    width: int = 800
    height: int = 600
    show_axes: bool = False
    show_unit_cell: bool = False
    projection: str = "perspective"  # perspective or orthographic
    fog_near: float = 100.0
    fog_far: float = 200.0
    camera_fov: float = 30.0
    antialiasing: bool = True
    ambient_light: float = 0.4
    direct_light: float = 0.8
    enable_click_selection: bool = True
    enable_hover_highlight: bool = True
    webxr_enabled: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "backgroundColor": self.background_color,
            "width": self.width,
            "height": self.height,
            "showAxes": self.show_axes,
            "showUnitCell": self.show_unit_cell,
            "projection": self.projection,
            "fog": {"near": self.fog_near, "far": self.fog_far},
            "cameraFov": self.camera_fov,
            "antialiasing": self.antialiasing,
            "lighting": {
                "ambient": self.ambient_light,
                "direct": self.direct_light,
            },
            "interaction": {
                "clickSelection": self.enable_click_selection,
                "hoverHighlight": self.enable_hover_highlight,
            },
            "webxr": self.webxr_enabled,
        }


@dataclass
class StyleSpec:
    """Style specification for molecular rendering."""

    style: RenderStyle = RenderStyle.CARTOON
    color: Optional[str] = None
    color_scheme: ColorScheme = ColorScheme.ELEMENT
    opacity: float = 1.0
    radius: float = 0.5
    thickness: float = 0.2
    arrows: bool = True  # For cartoon style
    tubes: bool = True  # For cartoon style
    selection: Optional[dict] = None

    def to_3dmol_spec(self) -> dict:
        """Convert to 3Dmol.js style specification."""
        spec: dict[str, Any] = {}

        style_config: dict[str, Any] = {}

        if self.color:
            style_config["color"] = self.color
        else:
            style_config["colorscheme"] = self.color_scheme.value

        if self.opacity < 1.0:
            style_config["opacity"] = self.opacity

        if self.style == RenderStyle.CARTOON:
            style_config["arrows"] = self.arrows
            style_config["tubes"] = self.tubes
            style_config["thickness"] = self.thickness
        elif self.style in (RenderStyle.STICK, RenderStyle.SPHERE):
            style_config["radius"] = self.radius
        elif self.style == RenderStyle.LINE:
            style_config["linewidth"] = self.thickness

        spec[self.style.value] = style_config

        return spec


@dataclass
class Selection:
    """Atom selection criteria."""

    chain: Optional[str] = None
    residue_name: Optional[str] = None
    residue_range: Optional[tuple[int, int]] = None
    atom_name: Optional[str] = None
    element: Optional[str] = None
    hetflag: Optional[bool] = None
    within_distance: Optional[tuple[float, dict]] = None

    def to_3dmol_selection(self) -> dict:
        """Convert to 3Dmol.js selection object."""
        sel: dict[str, Any] = {}

        if self.chain:
            sel["chain"] = self.chain
        if self.residue_name:
            sel["resn"] = self.residue_name
        if self.residue_range:
            sel["resi"] = f"{self.residue_range[0]}-{self.residue_range[1]}"
        if self.atom_name:
            sel["atom"] = self.atom_name
        if self.element:
            sel["elem"] = self.element
        if self.hetflag is not None:
            sel["hetflag"] = self.hetflag
        if self.within_distance:
            sel["within"] = {
                "distance": self.within_distance[0],
                "sel": self.within_distance[1],
            }

        return sel


@dataclass
class Label:
    """Text label in the viewer."""

    text: str
    position: tuple[float, float, float]
    font_size: int = 14
    font_color: str = "#ffffff"
    background_color: str = "#000000"
    background_opacity: float = 0.8
    border_color: str = "#ffffff"
    border_thickness: float = 1.0
    alignment: str = "center"

    def to_3dmol_label(self) -> tuple[str, dict]:
        """Convert to 3Dmol.js label specification."""
        return self.text, {
            "position": {"x": self.position[0], "y": self.position[1], "z": self.position[2]},
            "fontSize": self.font_size,
            "fontColor": self.font_color,
            "backgroundColor": self.background_color,
            "backgroundOpacity": self.background_opacity,
            "borderColor": self.border_color,
            "borderThickness": self.border_thickness,
            "alignment": self.alignment,
        }


@dataclass
class Surface:
    """Molecular surface configuration."""

    surface_type: SurfaceType = SurfaceType.VDW
    opacity: float = 0.8
    color: Optional[str] = None
    color_scheme: ColorScheme = ColorScheme.ELEMENT
    selection: Optional[Selection] = None

    def to_3dmol_surface(self) -> dict:
        """Convert to 3Dmol.js surface specification."""
        spec: dict[str, Any] = {
            "type": self.surface_type.value,
            "opacity": self.opacity,
        }

        if self.color:
            spec["color"] = self.color
        else:
            spec["colorscheme"] = self.color_scheme.value

        return spec


class MolecularViewer:
    """Interactive 3D molecular viewer using 3Dmol.js.

    This class generates JavaScript code and HTML for rendering
    molecules in a web browser using the 3Dmol.js library.
    """

    def __init__(self, config: Optional[ViewerConfig] = None):
        """Initialize molecular viewer.

        Args:
            config: Viewer configuration
        """
        self.config = config or ViewerConfig()
        self._structures: list[tuple[PDBStructure, list[StyleSpec]]] = []
        self._labels: list[Label] = []
        self._surfaces: list[tuple[Selection, Surface]] = []
        self._animations: list[dict] = []
        self._click_callbacks: list[str] = []
        self._hover_callbacks: list[str] = []

    def add_structure(
        self,
        structure: PDBStructure,
        styles: Optional[list[StyleSpec]] = None,
    ) -> None:
        """Add a molecular structure to the viewer.

        Args:
            structure: PDB structure to display
            styles: List of style specifications
        """
        if styles is None:
            styles = [StyleSpec(style=RenderStyle.CARTOON)]

        self._structures.append((structure, styles))
        logger.info(f"Added structure {structure.pdb_id} with {len(styles)} styles")

    def add_label(self, label: Label) -> None:
        """Add a text label to the viewer.

        Args:
            label: Label configuration
        """
        self._labels.append(label)

    def add_surface(
        self,
        selection: Optional[Selection] = None,
        surface: Optional[Surface] = None,
    ) -> None:
        """Add a molecular surface.

        Args:
            selection: Atom selection for surface
            surface: Surface configuration
        """
        if selection is None:
            selection = Selection()
        if surface is None:
            surface = Surface()

        self._surfaces.append((selection, surface))

    def add_animation(
        self,
        animation_type: str,
        duration: float = 1000,
        loop: bool = False,
        **kwargs,
    ) -> None:
        """Add an animation to the viewer.

        Args:
            animation_type: Type of animation (spin, rock, zoom, etc.)
            duration: Duration in milliseconds
            loop: Whether to loop the animation
            **kwargs: Additional animation parameters
        """
        anim = {
            "type": animation_type,
            "duration": duration,
            "loop": loop,
            **kwargs,
        }
        self._animations.append(anim)

    def add_click_callback(self, callback_js: str) -> None:
        """Add a JavaScript callback for atom clicks.

        Args:
            callback_js: JavaScript function body
        """
        self._click_callbacks.append(callback_js)

    def add_hover_callback(self, callback_js: str) -> None:
        """Add a JavaScript callback for atom hover.

        Args:
            callback_js: JavaScript function body
        """
        self._hover_callbacks.append(callback_js)

    def generate_viewer_state(self) -> dict:
        """Generate the complete viewer state as a dictionary.

        Returns:
            Dictionary containing all viewer state
        """
        state = {
            "config": self.config.to_dict(),
            "structures": [],
            "labels": [],
            "surfaces": [],
            "animations": self._animations,
        }

        for structure, styles in self._structures:
            struct_state = {
                "pdb_id": structure.pdb_id,
                "pdb_data": structure.raw_content,
                "model": structure.to_3dmol_model(),
                "styles": [s.to_3dmol_spec() for s in styles],
                "center": structure.center.tolist(),
            }
            state["structures"].append(struct_state)

        for label in self._labels:
            text, spec = label.to_3dmol_label()
            state["labels"].append({"text": text, "spec": spec})

        for selection, surface in self._surfaces:
            state["surfaces"].append(
                {
                    "selection": selection.to_3dmol_selection(),
                    "surface": surface.to_3dmol_surface(),
                }
            )

        return state

    def generate_javascript(self) -> str:
        """Generate 3Dmol.js initialization JavaScript.

        Returns:
            JavaScript code for initializing the viewer
        """
        state = self.generate_viewer_state()

        js_code = f"""
// Initialize 3Dmol.js viewer
const viewerConfig = {json.dumps(state['config'], indent=2)};
const viewer = $3Dmol.createViewer('viewer-container', {{
    backgroundColor: viewerConfig.backgroundColor,
    antialias: viewerConfig.antialiasing,
}});

viewer.setProjection(viewerConfig.projection);
"""

        # Add structures
        for i, struct in enumerate(state["structures"]):
            js_code += f"""
// Add structure {struct['pdb_id']}
const model{i} = viewer.addModel({json.dumps(struct['pdb_data'])}, 'pdb');
"""
            for style in struct["styles"]:
                js_code += f"viewer.setStyle({{}}, {json.dumps(style)});\n"

        # Add surfaces
        for surface_spec in state["surfaces"]:
            js_code += f"""
viewer.addSurface($3Dmol.SurfaceType.{surface_spec['surface']['type']},
    {json.dumps(surface_spec['surface'])},
    {json.dumps(surface_spec['selection'])}
);
"""

        # Add labels
        for label in state["labels"]:
            js_code += f"""
viewer.addLabel({json.dumps(label['text'])}, {json.dumps(label['spec'])});
"""

        # Add callbacks
        if self._click_callbacks:
            js_code += """
viewer.setClickable({}, true, function(atom) {
"""
            for cb in self._click_callbacks:
                js_code += f"    {cb}\n"
            js_code += "});\n"

        if self._hover_callbacks:
            js_code += """
viewer.setHoverable({}, true, function(atom) {
"""
            for cb in self._hover_callbacks:
                js_code += f"    {cb}\n"
            js_code += "});\n"

        # Add animations
        for anim in self._animations:
            if anim["type"] == "spin":
                js_code += f"""
viewer.spin({anim.get('axis', 'y')}, {anim.get('speed', 1)});
"""
            elif anim["type"] == "rock":
                js_code += f"""
viewer.rock({anim.get('axis', 'y')}, {anim.get('angle', 20)});
"""

        js_code += """
// Zoom to fit all molecules
viewer.zoomTo();
viewer.render();
"""

        return js_code

    def generate_html(
        self,
        title: str = "Molecular Dynamics Lab",
        include_controls: bool = True,
    ) -> str:
        """Generate complete HTML page with 3Dmol.js viewer.

        Args:
            title: Page title
            include_controls: Whether to include UI controls

        Returns:
            Complete HTML document
        """
        js_code = self.generate_javascript()

        controls_html = ""
        if include_controls:
            controls_html = """
    <div id="controls" style="position: absolute; top: 10px; left: 10px; z-index: 100; background: rgba(0,0,0,0.7); padding: 15px; border-radius: 8px; color: white; font-family: Arial, sans-serif;">
        <h3 style="margin: 0 0 10px 0;">Viewer Controls</h3>

        <div style="margin-bottom: 10px;">
            <label>Style:</label><br>
            <select id="styleSelect" onchange="changeStyle(this.value)" style="width: 100%; padding: 5px;">
                <option value="cartoon">Cartoon</option>
                <option value="stick">Stick</option>
                <option value="sphere">Sphere</option>
                <option value="line">Line</option>
                <option value="surface">Surface</option>
            </select>
        </div>

        <div style="margin-bottom: 10px;">
            <label>Color:</label><br>
            <select id="colorSelect" onchange="changeColor(this.value)" style="width: 100%; padding: 5px;">
                <option value="element">Element</option>
                <option value="chain">Chain</option>
                <option value="residue">Residue</option>
                <option value="ss">Secondary Structure</option>
                <option value="spectrum">Spectrum</option>
            </select>
        </div>

        <div style="margin-bottom: 10px;">
            <label>
                <input type="checkbox" id="spinToggle" onchange="toggleSpin(this.checked)">
                Spin Animation
            </label>
        </div>

        <div style="margin-bottom: 10px;">
            <label>
                <input type="checkbox" id="surfaceToggle" onchange="toggleSurface(this.checked)">
                Show Surface
            </label>
        </div>

        <div style="margin-bottom: 10px;">
            <button onclick="viewer.zoomTo()" style="width: 100%; padding: 8px; cursor: pointer;">Reset View</button>
        </div>

        <div style="margin-bottom: 10px;">
            <button onclick="exportPNG()" style="width: 100%; padding: 8px; cursor: pointer;">Export PNG</button>
        </div>

        <div id="atomInfo" style="margin-top: 15px; padding-top: 10px; border-top: 1px solid #444;">
            <strong>Selected Atom:</strong>
            <div id="atomDetails">Click an atom to see details</div>
        </div>
    </div>
"""

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background: #1a1a2e;
            overflow: hidden;
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        #viewer-container {{
            width: 100vw;
            height: 100vh;
            position: relative;
        }}
        #loading {{
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #00d4ff;
            font-size: 24px;
            z-index: 200;
        }}
        .spinner {{
            border: 4px solid #333;
            border-top: 4px solid #00d4ff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 15px;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
    </style>
</head>
<body>
    <div id="loading">
        <div class="spinner"></div>
        <div>Loading Molecular Structure...</div>
    </div>

    <div id="viewer-container"></div>

    {controls_html}

    <script>
        let viewer;
        let currentSurface = null;
        let isSpinning = false;

        document.addEventListener('DOMContentLoaded', function() {{
            {js_code}

            document.getElementById('loading').style.display = 'none';

            // Click callback to show atom info
            viewer.setClickable({{}}, true, function(atom, viewer, event, container) {{
                if (atom) {{
                    const info = `
                        <strong>${{atom.resn}} ${{atom.resi}}</strong><br>
                        Atom: ${{atom.atom}}<br>
                        Element: ${{atom.elem}}<br>
                        Chain: ${{atom.chain}}<br>
                        Position: (${{atom.x.toFixed(2)}}, ${{atom.y.toFixed(2)}}, ${{atom.z.toFixed(2)}})
                    `;
                    document.getElementById('atomDetails').innerHTML = info;

                    // Highlight selected atom
                    viewer.setStyle({{'serial': atom.serial}}, {{'sphere': {{'color': '#ff0000', 'radius': 0.5}}}});
                    viewer.render();
                }}
            }});
        }});

        function changeStyle(style) {{
            const styleSpec = {{}};
            styleSpec[style] = {{'colorscheme': document.getElementById('colorSelect').value}};
            viewer.setStyle({{}}, styleSpec);
            viewer.render();
        }}

        function changeColor(scheme) {{
            const style = document.getElementById('styleSelect').value;
            const styleSpec = {{}};
            styleSpec[style] = {{'colorscheme': scheme}};
            viewer.setStyle({{}}, styleSpec);
            viewer.render();
        }}

        function toggleSpin(enabled) {{
            if (enabled) {{
                viewer.spin('y');
                isSpinning = true;
            }} else {{
                viewer.spin(false);
                isSpinning = false;
            }}
        }}

        function toggleSurface(show) {{
            if (show) {{
                currentSurface = viewer.addSurface($3Dmol.SurfaceType.VDW, {{
                    opacity: 0.7,
                    colorscheme: 'element'
                }});
            }} else if (currentSurface) {{
                viewer.removeSurface(currentSurface);
                currentSurface = null;
            }}
            viewer.render();
        }}

        function exportPNG() {{
            const uri = viewer.pngURI();
            const link = document.createElement('a');
            link.href = uri;
            link.download = 'molecule.png';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }}
    </script>
</body>
</html>
"""
        return html

    def to_json(self) -> str:
        """Export viewer state as JSON.

        Returns:
            JSON string of viewer state
        """
        return json.dumps(self.generate_viewer_state(), indent=2)
