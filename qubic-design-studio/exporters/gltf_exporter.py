"""glTF 2.0 format exporter."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class GLTFExporter:
    """Export meshes to glTF 2.0 format."""

    def export_mesh(
        self, mesh: Any, output_path: Path, include_animations: bool = False
    ) -> None:
        """Export mesh to glTF file.

        Args:
            mesh: Tire mesh to export
            output_path: Output file path
            include_animations: Whether to include animations
        """
        output_path = Path(output_path)

        # Build glTF structure
        gltf = {
            "asset": {"version": "2.0", "generator": "QUBIC Design Studio"},
            "scene": 0,
            "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}],
            "meshes": [
                {
                    "primitives": [
                        {
                            "attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2},
                            "indices": 3,
                        }
                    ]
                }
            ],
            "accessors": [
                {
                    "bufferView": 0,
                    "componentType": 5126,  # FLOAT
                    "count": mesh.num_vertices,
                    "type": "VEC3",
                    "min": mesh.vertices.min(axis=0).tolist(),
                    "max": mesh.vertices.max(axis=0).tolist(),
                },
                {
                    "bufferView": 1,
                    "componentType": 5126,  # FLOAT
                    "count": mesh.num_vertices,
                    "type": "VEC3",
                },
                {
                    "bufferView": 2,
                    "componentType": 5126,  # FLOAT
                    "count": mesh.num_vertices,
                    "type": "VEC2",
                },
                {
                    "bufferView": 3,
                    "componentType": 5125,  # UNSIGNED_INT
                    "count": mesh.num_faces * 3,
                    "type": "SCALAR",
                },
            ],
            "bufferViews": [
                {"buffer": 0, "byteOffset": 0, "byteLength": mesh.vertices.nbytes},
                {
                    "buffer": 0,
                    "byteOffset": mesh.vertices.nbytes,
                    "byteLength": mesh.normals.nbytes,
                },
                {
                    "buffer": 0,
                    "byteOffset": mesh.vertices.nbytes + mesh.normals.nbytes,
                    "byteLength": mesh.uvs.nbytes,
                },
                {
                    "buffer": 0,
                    "byteOffset": mesh.vertices.nbytes + mesh.normals.nbytes + mesh.uvs.nbytes,
                    "byteLength": mesh.faces.nbytes,
                },
            ],
            "buffers": [
                {
                    "byteLength": mesh.vertices.nbytes
                    + mesh.normals.nbytes
                    + mesh.uvs.nbytes
                    + mesh.faces.nbytes
                }
            ],
        }

        # Write glTF JSON
        with open(output_path, "w") as f:
            json.dump(gltf, f, indent=2)

        print(f"Exported mesh to {output_path}")
