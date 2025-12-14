# QUBIC Design Studio

CAD/CAM export and spatial computing interfaces for tire visualization.

## Features

- **CAD Export**: OBJ, FBX, STEP, glTF format support
- **AR/VR Integration**: HoloLens, ARKit, VR headset support
- **Spatial Coordinates**: Real-world coordinate mapping
- **Mesh Streaming**: Optimized mesh transfer protocols

## Exporters

### OBJ Exporter
```python
from qubic_design_studio.exporters.obj_exporter import OBJExporter

exporter = OBJExporter()
exporter.export_mesh(tire_mesh, "tire.obj")
```

### glTF Exporter
```python
from qubic_design_studio.exporters.gltf_exporter import GLTFExporter

exporter = GLTFExporter()
exporter.export_mesh(tire_mesh, "tire.gltf", include_animations=True)
```

## Spatial Computing

### HoloLens Integration
```python
from qubic_design_studio.spatial.holo_adapter import HoloLensAdapter

adapter = HoloLensAdapter()
spatial_mesh = adapter.convert_to_spatial_mesh(tire_mesh)
adapter.stream_to_device(spatial_mesh)
```

## Installation

```bash
pip install -e .
```

## License

Apache License 2.0
