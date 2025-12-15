# QUBIC-VIZ - Production-Grade 3D Visualization Engine

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

QUBIC-VIZ is a production-grade 3D visualization and rendering engine specifically designed for quantum-enhanced tire simulation and digital twin visualization.

## Features

- **Multi-Backend Rendering**: Support for CPU, CUDA, OpenGL, and Vulkan backends
- **Tire-Specific Rendering**: Specialized renderers for tire geometry, thermal maps, stress distributions, and wear patterns
- **GPU Acceleration**: CUDA-accelerated compute kernels with automatic CPU fallback
- **PBR Materials**: Physically-based rendering for realistic material appearance
- **Scene Graph**: Hierarchical scene management for complex visualizations
- **Field Visualization**: Scalar and vector field rendering on 3D meshes
- **Video Export**: MP4/WebM video generation from animation sequences
- **QuASIM Integration**: Native adapters for QuASIM tire simulation results

## Architecture

```
qubic-viz/
├── core/              # Core rendering components
│   ├── renderer.py    # Scene renderer with multi-backend support
│   ├── scene_graph.py # Hierarchical scene management
│   ├── camera.py      # Camera and camera controller
│   └── lighting.py    # Lighting and PBR materials
├── engines/           # Specialized rendering engines
│   ├── tire_renderer.py      # Tire visualization
│   ├── mesh_generator.py     # Tire mesh generation
│   ├── deformation_engine.py # Contact patch deformation
│   └── field_visualizer.py   # Field visualization
├── adapters/          # Data adapters
│   ├── quasim_adapter.py     # QuASIM data adapter
│   └── tire_data_adapter.py  # Tire simulation adapter
├── gpu/               # GPU acceleration
│   ├── kernels.py            # GPU compute kernels
│   ├── compute_pipeline.py   # Compute orchestration
│   └── memory_manager.py     # GPU memory management
└── shaders/           # GLSL shaders
    ├── tire_surface.glsl     # PBR tire surface shader
    ├── thermal_field.glsl    # Thermal visualization
    └── stress_field.glsl     # Stress visualization
```

## Installation

### From Source

```bash
pip install -e .
```

### With GPU Support

```bash
pip install -e ".[gpu]"
```

## Quick Start

### Basic Tire Rendering

```python
from qubic_viz import TireRenderer, RenderConfig
from qubic_viz.engines import TireMeshGenerator
from quasim.domains.tire import TireGeometry

# Create tire geometry
geometry = TireGeometry(
    outer_diameter_mm=700.0,
    width_mm=225.0,
    rim_diameter_inch=17.0
)

# Generate mesh
mesh_gen = TireMeshGenerator(resolution=32)
tire_mesh = mesh_gen.generate_tire_mesh(geometry)

# Create renderer
config = RenderConfig(width=1920, height=1080, use_gpu=True)
renderer = TireRenderer(config)

# Render 3D tire
image = renderer.render_tire_3d(tire_mesh)
```

### Visualizing Simulation Results

```python
from qubic_viz.adapters import TireDataAdapter
from quasim.domains.tire import run_tire_simulation

# Run simulation
sim_result = run_tire_simulation(...)

# Extract visualization data
viz_data = TireDataAdapter.extract_visualization_data(sim_result)

# Render performance dashboard
dashboard = renderer.render_performance_dashboard(sim_result)

# Render thermal map
thermal_viz = renderer.render_thermal_map(
    tire_mesh,
    viz_data['thermal_map']
)

# Render stress distribution
stress_viz = renderer.render_stress_distribution(
    tire_mesh,
    viz_data['stress_distribution']
)
```

### Creating Animations

```python
from pathlib import Path

# Render animation sequence
frames = renderer.render_sequence(
    scene=my_scene,
    camera=my_camera,
    num_frames=120,
    output_path=Path("tire_animation.mp4")
)
```

## GPU Acceleration

QUBIC-VIZ automatically detects and uses available GPU acceleration:

```python
from qubic_viz.gpu import GPUKernels, ComputePipeline

# Create GPU kernels (auto-detects CUDA)
kernels = GPUKernels(device="cuda")

# Create compute pipeline
pipeline = ComputePipeline(device="cuda", memory_limit_mb=4096)

# Check GPU availability
if kernels.cuda_available:
    print("CUDA acceleration enabled")
else:
    print("Using CPU fallback")
```

## Field Visualization

Visualize scalar and vector fields on tire meshes:

```python
from qubic_viz.engines import FieldVisualizer

visualizer = FieldVisualizer(tire_mesh)

# Visualize temperature field
temp_image = visualizer.visualize_scalar_field(
    field_data=temperature_array,
    field_name="Temperature (°C)",
    colormap="hot"
)

# Visualize stress vectors
stress_image = visualizer.visualize_vector_field(
    vector_data=stress_vectors,
    field_name="Stress Vectors"
)

# Create 2D heatmap projection
heatmap = visualizer.create_heatmap_2d(
    field_data=temperature_array,
    projection="top"
)
```

## Integration with QuASIM

QUBIC-VIZ provides native integration with QuASIM tire simulations:

```python
from qubic_viz.adapters import QuASIMDataAdapter

# Convert QuASIM data to visualization format
viz_data = QuASIMDataAdapter.convert_to_visualization_format(
    data=quasim_result,
    data_type="tire"
)

# Create animation frames from simulation sequence
frames = QuASIMDataAdapter.create_animation_frames(
    simulation_sequence=sim_results,
    frame_skip=2
)
```

## Performance

- **CPU Rendering**: 10-30 fps for typical tire meshes (32x64 resolution)
- **CUDA Rendering**: 60-120 fps with GPU acceleration
- **Memory Usage**: ~100-500 MB for typical tire visualizations
- **Video Export**: 30 fps MP4 with H.264 encoding

## Requirements

- Python 3.10+
- NumPy >= 1.24.0
- Matplotlib >= 3.7.0
- Pillow >= 10.0.0
- OpenCV (optional, for video export)
- PyTorch >= 2.0.0 (optional, for GPU acceleration)
- Trimesh >= 4.0.0 (optional, for advanced mesh operations)

## License

Apache License 2.0 - See LICENSE file for details

## Contributing

See CONTRIBUTING.md for development guidelines

## Support

For issues and questions:

- GitHub Issues: <https://github.com/robertringler/Qubic/issues>
- Documentation: <https://github.com/robertringler/Qubic/docs>

## Citation

```bibtex
@software{qubic_viz,
  title={QUBIC-VIZ: Production-Grade 3D Visualization Engine},
  author={QuASIM Team},
  year={2024},
  url={https://github.com/robertringler/Qubic}
}
```
