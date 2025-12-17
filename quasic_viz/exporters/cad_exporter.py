"""CAD exporter supporting multiple formats (OBJ, glTF, FBX, STEP)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

SUPPORTED_FORMATS = ["obj", "glb", "fbx", "step", "step_ap242"]


class CADExporter:
    """Export mesh to various CAD formats.

    Supports OBJ, glTF/GLB, FBX, and STEP formats for CAD/CAM pipelines.
    """

    @staticmethod
    def export_mesh(mesh: Any, filename: str | Path, format: str = "obj") -> None:
        """Export mesh to specified format.

        Args:
            mesh: Mesh object with vertices and faces attributes
            filename: Output file path
            format: Export format (obj, glb, fbx, step, step_ap242)

        Raises:
            ValueError: If format is not supported
            ImportError: If required library (trimesh) is not available
        """

        if format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {format}. Supported: {SUPPORTED_FORMATS}")

        filename = Path(filename)

        # Try to use trimesh for full format support
        try:
            import trimesh

            # Extract vertices and faces from mesh
            if hasattr(mesh, "vertices"):
                vertices = mesh.vertices
            elif isinstance(mesh, dict):
                vertices = mesh["vertices"]
            else:
                raise ValueError("Mesh must have 'vertices' attribute or be a dict")

            if hasattr(mesh, "faces"):
                faces = mesh.faces
            elif isinstance(mesh, dict):
                faces = mesh["faces"]
            else:
                raise ValueError("Mesh must have 'faces' attribute or be a dict")

            tmesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            tmesh.export(file_obj=str(filename), file_type=format)

        except ImportError as exc:
            # Fallback for basic OBJ export without trimesh
            if format == "obj":
                CADExporter._export_obj_fallback(mesh, filename)
            else:
                raise ImportError(
                    f"trimesh required for {format} export. Install with: pip install trimesh"
                ) from exc

    @staticmethod
    def _export_obj_fallback(mesh: Any, filename: Path) -> None:
        """Export mesh to OBJ format without trimesh dependency.

        Args:
            mesh: Mesh object or dictionary
            filename: Output file path
        """

        # Extract vertices and faces
        if hasattr(mesh, "vertices"):
            vertices = mesh.vertices
        elif isinstance(mesh, dict):
            vertices = mesh["vertices"]
        else:
            raise ValueError("Mesh must have 'vertices' attribute or be a dict")

        if hasattr(mesh, "faces"):
            faces = mesh.faces
        elif isinstance(mesh, dict):
            faces = mesh["faces"]
        else:
            raise ValueError("Mesh must have 'faces' attribute or be a dict")

        with open(filename, "w") as f:
            f.write("# QuASIC Visualization Export\n")
            f.write(f"# Vertices: {len(vertices)}\n")
            f.write(f"# Faces: {len(faces)}\n\n")

            # Write vertices
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write faces (OBJ is 1-indexed)
            f.write("\n")
            for face in faces:
                f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

    @staticmethod
    def get_supported_formats() -> list[str]:
        """Get list of supported export formats.

        Returns:
            List of supported format strings
        """

        return SUPPORTED_FORMATS.copy()
