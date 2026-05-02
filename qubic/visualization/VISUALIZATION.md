# QUBIC Visualization Subsystem

## Overview

The QUBIC Visualization Subsystem is a production-ready, unified visualization framework for simulation results. It consolidates scattered visualization code and provides end-to-end rendering pipelines for tire simulations, quantum circuits, and generic mesh/field data.

## Architecture

```
qubic/visualization/
├── core/              # Core data models, camera, and lighting
├── adapters/          # Simulation-specific data adapters
├── backends/          # Rendering backends (matplotlib, headless, GPU)
├── pipelines/         # Visualization workflows
├── exporters/         # Export to images, videos, and interactive formats
├── examples/          # Runnable demonstration scripts
├── tests/             # Comprehensive test suite
└── cli.py             # Command-line interface
```

### Data Flow

```
Simulation Data → Adapter → VisualizationData → Pipeline → Backend → Output
```

1. **Adapters** convert simulation-specific data to unified `VisualizationData` format
2. **Pipelines** orchestrate rendering workflows (static, time-series, streaming)
3. **Backends** perform the actual rendering (matplotlib, headless, GPU)
4. **Exporters** write results to files (PNG, JPEG, MP4, GIF, HTML)

## Installation

### Basic Installation

```bash
pip install -e .
```

This installs the core visualization subsystem with matplotlib support.

### Full Installation (with all features)

```bash
pip install -e ".[viz]"
```

This includes:
- `imageio` and `imageio-ffmpeg` for video export
- `plotly` for interactive HTML visualizations
- `torch` for GPU-accelerated rendering
- `websockets` for streaming support

## Quick Start

### Command-Line Usage

```bash
# Render a tire visualization
qubic-viz example tire --output-dir ./viz_output

# Render a quantum state visualization
qubic-viz example quantum --output-dir ./viz_output

# Render custom simulation data
qubic-viz render --input data.json --output result.png --adapter mesh --field temperature

# Create animation from time-series
qubic-viz animate --input timeseries.json --output animation.mp4 --fps 30
```

### Python API Usage

```python
from qubic.visualization.adapters.tire import TireSimulationAdapter
from qubic.visualization.pipelines.static import StaticPipeline

# Load tire simulation data
adapter = TireSimulationAdapter()
tire_data = adapter.create_synthetic_tire(resolution=48)

# Create visualization pipeline
pipeline = StaticPipeline(backend="headless", dpi=150)

# Render and save
pipeline.render_and_save(
    data=tire_data,
    output_path="tire_temperature.png",
    scalar_field="temperature",
    colormap="hot"
)
```

## Components

### Core Data Model

`VisualizationData` is the unified data structure for all visualizations:

```python
from qubic.visualization.core.data_model import VisualizationData
import numpy as np

# Create visualization data
vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
faces = np.array([[0, 1, 2]])
temperature = np.array([300.0, 310.0, 305.0])

data = VisualizationData(
    vertices=vertices,
    faces=faces,
    scalar_fields={"temperature": temperature}
)

# Automatically computes normals and validates data
bbox_min, bbox_max = data.get_bounding_box()
temp_range = data.get_field_range("temperature")
```

### Adapters

Adapters convert simulation-specific data formats to `VisualizationData`:

#### TireSimulationAdapter

```python
from qubic.visualization.adapters.tire import TireSimulationAdapter

adapter = TireSimulationAdapter()

# From dictionary
tire_data = adapter.load_data({
    "vertices": vertices,
    "faces": faces,
    "temperature": temperature_field,
    "stress_von_mises": stress_field
})

# Synthetic test data
tire_data = adapter.create_synthetic_tire(resolution=32)
```

#### QuantumSimulationAdapter

```python
from qubic.visualization.adapters.quantum import QuantumSimulationAdapter
import numpy as np

adapter = QuantumSimulationAdapter()

# From amplitude array
amplitudes = np.array([1.0+0j, 0.0+0j, 0.0+0j, 0.0+0j])
quantum_data = adapter.load_data(amplitudes)

# Synthetic test states
quantum_data = adapter.create_synthetic_state(
    n_qubits=3,
    state_type="superposition"  # or "entangled", "basis"
)
```

#### MeshAdapter

```python
from qubic.visualization.adapters.mesh import MeshAdapter

adapter = MeshAdapter()

# Generic mesh
mesh_data = adapter.load_data({
    "vertices": vertices,
    "faces": faces,
    "custom_field": field_data
})

# Test meshes
sphere = adapter.create_test_mesh("sphere", resolution=20)
cube = adapter.create_test_mesh("cube")
cylinder = adapter.create_test_mesh("cylinder", resolution=20)
```

#### TimeSeriesAdapter

```python
from qubic.visualization.adapters.timeseries import TimeSeriesAdapter

adapter = TimeSeriesAdapter()

# Load time-series
timesteps = [
    {"vertices": v1, "faces": f1, "time": 0.0, "field": field1},
    {"vertices": v2, "faces": f2, "time": 0.1, "field": field2},
]
adapter.load_data(timesteps)

# Access timesteps
data_t0 = adapter.get_timestep(0)
n_steps = adapter.get_num_timesteps()
start, end = adapter.get_time_range()
```

### Backends

#### MatplotlibBackend (CPU Rendering)

```python
from qubic.visualization.backends.matplotlib_backend import MatplotlibBackend

backend = MatplotlibBackend(figsize=(12, 10), dpi=150)
fig = backend.render(
    data=vis_data,
    scalar_field="temperature",
    colormap="viridis"
)
backend.save("output.png")
```

#### HeadlessBackend (CI/Cluster)

```python
from qubic.visualization.backends.headless_backend import HeadlessBackend

backend = HeadlessBackend(dpi=150)
backend.render(data=vis_data)
backend.save("output.png")  # No display, file only
```

#### GPUBackend (GPU-Accelerated with CPU Fallback)

```python
from qubic.visualization.backends.gpu_backend import GPUBackend

# Automatically detects GPU availability
backend = GPUBackend(dpi=150)
backend.render(data=vis_data)
backend.save("output.png")

# Force CPU fallback
backend = GPUBackend(force_cpu=True)
```

### Pipelines

#### StaticPipeline (Single-Frame Rendering)

```python
from qubic.visualization.pipelines.static import StaticPipeline

pipeline = StaticPipeline(backend="headless", dpi=150)

# Render and save
pipeline.render_and_save(
    data=vis_data,
    output_path="output.png",
    scalar_field="temperature",
    colormap="hot"
)

# Or render and show (interactive)
pipeline.render_and_show(data=vis_data)
```

#### TimeSeriesPipeline (Animations)

```python
from qubic.visualization.pipelines.timeseries import TimeSeriesPipeline

pipeline = TimeSeriesPipeline(dpi=150)

# Render animation
pipeline.render_animation(
    adapter=timeseries_adapter,
    output_path="animation.mp4",
    scalar_field="temperature",
    fps=30,
    format="mp4"  # or "gif"
)

# Render individual frames
frames = pipeline.render_frames(
    adapter=timeseries_adapter,
    output_dir="./frames/"
)
```

#### StreamingPipeline (Real-Time WebSocket)

```python
from qubic.visualization.pipelines.streaming import StreamingPipeline
import asyncio

pipeline = StreamingPipeline(max_fps=10)

# Data generator function
def generate_data():
    return updated_visualization_data

# Start streaming
server = pipeline.create_server(host="0.0.0.0", port=8765)
loop = asyncio.get_event_loop()
loop.run_until_complete(server)
loop.create_task(pipeline.stream_data(
    data_generator=generate_data,
    scalar_field="field"
))
loop.run_forever()
```

### Exporters

#### ImageExporter

```python
from qubic.visualization.exporters.image import ImageExporter

exporter = ImageExporter(dpi=300, figsize=(16, 12))

# PNG export
exporter.export_png(
    data=vis_data,
    output_path="highres.png",
    scalar_field="temperature",
    transparent=False
)

# JPEG export
exporter.export_jpeg(
    data=vis_data,
    output_path="web_image.jpg",
    quality=95
)
```

#### VideoExporter

```python
from qubic.visualization.exporters.video import VideoExporter

exporter = VideoExporter(dpi=150)

# MP4 export
exporter.export_mp4(
    adapter=timeseries_adapter,
    output_path="simulation.mp4",
    fps=30,
    bitrate="5000k"
)

# GIF export
exporter.export_gif(
    adapter=timeseries_adapter,
    output_path="animation.gif",
    fps=10,
    loop=0  # infinite loop
)
```

#### InteractiveExporter

```python
from qubic.visualization.exporters.interactive import InteractiveExporter

exporter = InteractiveExporter()

# HTML export (requires plotly)
exporter.export_html(
    data=vis_data,
    output_path="interactive.html",
    scalar_field="temperature",
    title="Interactive Tire Visualization"
)

# JSON export for custom viewers
exporter.export_json(
    data=vis_data,
    output_path="data.json",
    scalar_field="temperature"
)
```

## Extending the System

### Adding a New Adapter

```python
from qubic.visualization.adapters.base import SimulationAdapter
from qubic.visualization.core.data_model import VisualizationData
import numpy as np

class MySimulationAdapter(SimulationAdapter):
    def load_data(self, source):
        # Parse your simulation format
        vertices, faces, fields = parse_simulation(source)
        
        return VisualizationData(
            vertices=vertices,
            faces=faces,
            scalar_fields=fields,
            metadata={"adapter_type": "my_simulation"}
        )
    
    def validate_source(self, source):
        # Validate that source can be loaded
        return is_valid_format(source)
```

### Custom Rendering Backend

Extend existing backends or create new ones:

```python
from qubic.visualization.backends.matplotlib_backend import MatplotlibBackend

class MyCustomBackend(MatplotlibBackend):
    def render(self, data, **kwargs):
        # Custom rendering logic
        fig = super().render(data, **kwargs)
        # Add custom elements
        return fig
```

## Testing

```bash
# Run all visualization tests
pytest qubic/visualization/tests/ -v

# Run specific test file
pytest qubic/visualization/tests/test_adapters.py -v

# Run with coverage
pytest qubic/visualization/tests/ --cov=qubic.visualization
```

## Dependencies

### Core (Required)
- `numpy>=1.24.0`
- `matplotlib>=3.7.0`
- `click>=8.0.0` (for CLI)

### Optional (Install with `pip install -e ".[viz]"`)
- `imageio>=2.0.0` (video export)
- `imageio-ffmpeg>=0.4.0` (MP4 encoding)
- `plotly>=5.14.0` (interactive HTML)
- `torch>=2.0.0` (GPU acceleration)
- `websockets>=11.0` (streaming)

## Performance Considerations

### GPU Acceleration

GPU acceleration is automatically enabled when PyTorch with CUDA is available:

```python
from qubic.visualization.backends.gpu_backend import GPUBackend

# Automatic GPU detection
backend = GPUBackend()
print(f"Using GPU: {backend.use_gpu}")
```

### Memory Efficiency

For large meshes (>100k vertices):

1. Use headless backend for batch processing
2. Render frames individually for animations
3. Consider downsampling for preview renders

### Headless Rendering on Clusters

```python
import matplotlib
matplotlib.use('Agg')  # Set before importing pyplot

from qubic.visualization.backends.headless_backend import HeadlessBackend
backend = HeadlessBackend()
```

## Troubleshooting

### ImportError: plotly not found

Install plotly for interactive HTML export:
```bash
pip install plotly
```

### GPU not detected

Verify CUDA installation:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### Animation export fails

Install video dependencies:
```bash
pip install imageio imageio-ffmpeg
```

### Display not available (headless mode)

Use `HeadlessBackend` instead of `MatplotlibBackend`:
```python
pipeline = StaticPipeline(backend="headless")
```

## Examples

See `qubic/visualization/examples/` for complete working examples:

- `render_tire.py` - Tire thermal/stress/wear visualization
- `render_quantum.py` - Quantum amplitude and phase visualization
- `stream_simulation.py` - Real-time streaming server

Run examples via CLI:
```bash
qubic-viz example tire --output-dir ./output
qubic-viz example quantum --output-dir ./output
qubic-viz example streaming  # Starts WebSocket server
```

## License

Apache License 2.0 - See LICENSE file for details.

## Support

For issues, feature requests, or questions:
- GitHub Issues: https://github.com/robertringler/Qubic/issues
- Documentation: See this file and inline code documentation

## Compliance

This visualization subsystem is designed to support:
- DO-178C Level A aerospace certification posture
- NIST 800-53 Rev 5 security controls
- CMMC 2.0 Level 2 compliance
- Deterministic reproducibility for validation

All rendering outputs are deterministic when using fixed random seeds for synthetic data generation.
