"""Wavefront OBJ format exporter."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class OBJExporter:
    """Export meshes to Wavefront OBJ format."""

    def export_mesh(self, mesh: Any, output_path: Path, include_normals: bool = True) -> None:
        """Export mesh to OBJ file.

        Args:
            mesh: Tire mesh to export
            output_path: Output file path
            include_normals: Whether to include vertex normals
        """

        output_path = Path(output_path)

        with open(output_path, "w") as f:
            # Write header
            f.write("# QUBIC Tire Mesh Export\n")
            f.write(f"# Vertices: {mesh.num_vertices}\n")
            f.write(f"# Faces: {mesh.num_faces}\n\n")

            # Write vertices
            for v in mesh.vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")

            # Write normals
            if include_normals:
                f.write("\n")
                for n in mesh.normals:
                    f.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")

            # Write texture coordinates
            f.write("\n")
            for uv in mesh.uvs:
                f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")

            # Write faces
            f.write("\n")
            for face in mesh.faces:
                if include_normals:
                    f.write(
                        f"f {face[0] + 1}/{face[0] + 1}/{face[0] + 1} "
                        f"{face[1] + 1}/{face[1] + 1}/{face[1] + 1} "
                        f"{face[2] + 1}/{face[2] + 1}/{face[2] + 1}\n"
                    )
                else:
                    f.write(f"f {face[0] + 1} {face[1] + 1} {face[2] + 1}\n")

        # Generate MTL file
        mtl_path = output_path.with_suffix(".mtl")
        self._generate_mtl(mtl_path)

        print(f"Exported mesh to {output_path}")

    def _generate_mtl(self, mtl_path: Path) -> None:
        """Generate material file.

        Args:
            mtl_path: Path to MTL file
        """

        with open(mtl_path, "w") as f:
            f.write("# QUBIC Tire Material\n")
            f.write("newmtl tire_material\n")
            f.write("Ka 0.1 0.1 0.1\n")
            f.write("Kd 0.1 0.1 0.1\n")
            f.write("Ks 0.5 0.5 0.5\n")
            f.write("Ns 32.0\n")
