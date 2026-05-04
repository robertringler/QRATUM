# QRATUM Aerospace Visualization Module

## Overview

The Aerospace Visualization Module extends the QUBIC visualization system with specialized rendering capabilities for aerospace simulation data. It maintains **DO-178C Level A** design patterns for deterministic, auditable rendering suitable for aerospace certification.

## Features

- **Flight Dynamics**: Aircraft trajectories with vapor trails and airflow streamlines
- **Structural Analysis**: FEA mesh visualization with stress fields and modal analysis
- **Thermal Systems**: Temperature distributions and heat flux visualization
- **Avionics Display**: Sensor FOV cones and radar cross-section polar plots
- **Compliance Support**: DO-178C Level A/B audit trails with SHA-256 frame hashing
- **Deterministic Rendering**: Seed-based reproducibility for certification

## Installation

The module is included with QRATUM. Ensure matplotlib is installed:

```bash
pip install matplotlib numpy
```

For enhanced thermal visualization with isotherms:

```bash
pip install scipy
```

## Quick Start

```python
from qubic.visualization.aerospace import AerospaceVisualizer, AerospaceVizConfig, ComplianceMode
import numpy as np

# Initialize with compliance mode
config = AerospaceVizConfig(
    compliance_mode=ComplianceMode.DO178C_LEVEL_A,
    seed=42,
    enable_audit_log=True,
)
viz = AerospaceVisualizer(config)

# Render flight trajectory
trajectory = np.random.randn(100, 3) * 1000  # 100 points, XYZ in meters
fig = viz.render_flight_trajectory(
    trajectory=trajectory,
    title="F-35 Flight Path",
    output_path="trajectory.png",
    show_vapor_trail=True,
)

# Export compliance audit trail
viz.export_audit_trail("audit_trail.json")
report = viz.generate_compliance_report()
print(f"Rendered {report['total_frames']} frames with compliance mode: {report['compliance_mode']}")
```

## API Reference

### Configuration

#### `AerospaceVizConfig`

Configuration dataclass extending `VizConfig` from QubicVisualizer.

**Fields:**
- `compliance_mode` (ComplianceMode): Compliance level (DO178C_LEVEL_A, DO178C_LEVEL_B, DEVELOPMENT)
- `render_backend` (RenderBackend): Rendering backend (matplotlib, threejs, headless)
- `seed` (int): Random seed for deterministic rendering (default: 42)
- `enable_audit_log` (bool): Enable audit trail recording (default: False)
- `resolution` (tuple[int, int]): Output resolution in pixels (default: 1920x1080)
- `target_fps` (int): Target frame rate for animations (default: 60)
- `show_velocity_vectors` (bool): Show velocity arrows (default: True)
- `show_pressure_gradients` (bool): Show pressure gradient visualizations (default: True)
- `show_streamlines` (bool): Show streamlines in flow fields (default: True)
- `streamline_density` (int): Number of streamlines to render (default: 100)
- `enable_hud` (bool): Enable heads-up display (default: True)
- `hud_opacity` (float): HUD transparency 0-1 (default: 0.8)
- `show_telemetry` (bool): Show telemetry data (default: True)

#### Configuration Presets

Load presets from `config/aerospace_presets.yaml`:

```python
import yaml
from pathlib import Path

preset_path = Path(__file__).parent / "config" / "aerospace_presets.yaml"
with open(preset_path) as f:
    presets = yaml.safe_load(f)

# Use production preset
config = AerospaceVizConfig(**presets["production"])
viz = AerospaceVisualizer(config)
```

Available presets:
- **draft**: Fast rendering for development (low DPI, reduced effects)
- **production**: Balanced quality/performance with DO-178C Level B compliance
- **cinematic**: Maximum quality for presentations (4K, high DPI)
- **compliance_do178c**: DO-178C Level A compliant settings with full audit
- **headless**: CI/cluster rendering without display

### Main Class

#### `AerospaceVisualizer`

Aerospace-grade visualizer extending `QubicVisualizer`.

```python
viz = AerospaceVisualizer(config: Optional[AerospaceVizConfig] = None)
```

### Flight Dynamics Methods

#### `render_flight_trajectory()`

Render aircraft flight trajectory with optional vapor trails.

```python
fig = viz.render_flight_trajectory(
    trajectory: np.ndarray,          # Nx3 array of XYZ positions (m)
    velocity: Optional[np.ndarray],  # Optional Nx3 velocity vectors (m/s)
    title: str = "Flight Trajectory",
    output_path: Optional[str] = None,
    show_vapor_trail: bool = False,
) -> Optional[Figure]
```

**Example:**

```python
import numpy as np

# Generate spiral trajectory
t = np.linspace(0, 4*np.pi, 200)
trajectory = np.column_stack([
    100 * np.cos(t),
    100 * np.sin(t),
    50 * t
])

# Velocity vectors (tangent to trajectory)
velocity = np.gradient(trajectory, axis=0) * 10

fig = viz.render_flight_trajectory(
    trajectory=trajectory,
    velocity=velocity,
    title="Spiral Climb Maneuver",
    output_path="spiral_trajectory.png",
    show_vapor_trail=True
)
```

#### `render_airflow_streamlines()`

Render airflow velocity field with streamlines.

```python
fig = viz.render_airflow_streamlines(
    velocity_field: np.ndarray,           # Flattened (N, 3) velocity field (m/s)
    grid_shape: tuple[int, int, int],     # Grid shape (nx, ny, nz)
    density: Optional[int] = None,        # Streamline density (uses config default)
    title: str = "Airflow Streamlines",
    output_path: Optional[str] = None,
) -> Optional[Figure]
```

**Example:**

```python
# Create synthetic velocity field
nx, ny, nz = 20, 20, 20
x = np.linspace(-1, 1, nx)
y = np.linspace(-1, 1, ny)
z = np.linspace(-1, 1, nz)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Vortex flow
vx = -Y.flatten()
vy = X.flatten()
vz = np.zeros_like(vx)
velocity_field = np.column_stack([vx, vy, vz])

fig = viz.render_airflow_streamlines(
    velocity_field=velocity_field,
    grid_shape=(nx, ny, nz),
    density=80,
    title="Wing Tip Vortex",
    output_path="airflow.png"
)
```

### Structural Analysis Methods

#### `render_fem_mesh()`

Render FEA mesh with stress field overlay.

```python
fig = viz.render_fem_mesh(
    nodes: np.ndarray,                     # Nx3 node coordinates
    elements: np.ndarray,                  # Mx4 element connectivity (tetrahedral)
    stress_tensor: Optional[np.ndarray],   # Optional Nx6 stress tensor
    title: str = "FEM Mesh",
    output_path: Optional[str] = None,
    show_wireframe: bool = True,
    colormap: Optional[str] = None,
) -> Optional[Figure]
```

**Example:**

```python
# Simple tetrahedral mesh
nodes = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
])

elements = np.array([
    [0, 1, 2, 3]  # Single tetrahedron
])

# Von Mises stress (xx, yy, zz, xy, xz, yz)
stress = np.array([
    [100e6, 50e6, 30e6, 10e6, 5e6, 5e6],
    [120e6, 60e6, 40e6, 15e6, 8e6, 8e6],
    [90e6, 45e6, 25e6, 8e6, 4e6, 4e6],
    [110e6, 55e6, 35e6, 12e6, 6e6, 6e6],
])

fig = viz.render_fem_mesh(
    nodes=nodes,
    elements=elements,
    stress_tensor=stress,
    title="Wing Spar Stress Analysis",
    output_path="fem_mesh.png"
)
```

#### `render_modal_analysis()`

Render modal analysis eigenmodes.

```python
fig = viz.render_modal_analysis(
    nodes: np.ndarray,                # Nx3 node coordinates
    eigenvectors: np.ndarray,         # NxMx3 mode shapes (M modes)
    eigenfrequencies: np.ndarray,     # M eigenfrequencies (Hz)
    mode_index: int = 0,
    amplitude_scale: float = 1.0,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
    animate: bool = False,
) -> Optional[Figure]
```

**Example:**

```python
# Simple beam nodes
nodes = np.array([[i, 0, 0] for i in range(10)])

# First bending mode shape
mode_shape = np.array([
    [0, np.sin(i * np.pi / 9), 0] for i in range(10)
])
eigenvectors = mode_shape[:, np.newaxis, :]  # Shape: (10, 1, 3)
eigenfrequencies = np.array([25.3])  # Hz

fig = viz.render_modal_analysis(
    nodes=nodes,
    eigenvectors=eigenvectors,
    eigenfrequencies=eigenfrequencies,
    mode_index=0,
    amplitude_scale=0.5,
    output_path="modal_analysis.png"
)
```

### Thermal Systems Methods

#### `render_thermal_field()`

Render temperature distribution on geometry.

```python
fig = viz.render_thermal_field(
    temperature: np.ndarray,          # N temperature values (K)
    geometry: np.ndarray,             # Nx3 point coordinates
    title: str = "Thermal Field",
    output_path: Optional[str] = None,
    colormap: Optional[str] = None,
    show_isotherms: bool = False,
    isotherm_levels: Optional[int] = None,
) -> Optional[Figure]
```

**Example:**

```python
# Create thermal field on surface
n_points = 500
x = np.random.uniform(-1, 1, n_points)
y = np.random.uniform(-1, 1, n_points)
z = np.random.uniform(-1, 1, n_points)
geometry = np.column_stack([x, y, z])

# Temperature distribution (K)
r = np.sqrt(x**2 + y**2 + z**2)
temperature = 300 + 200 * np.exp(-r**2)

fig = viz.render_thermal_field(
    temperature=temperature,
    geometry=geometry,
    title="Reentry Heating Profile",
    output_path="thermal.png",
    colormap="hot",
    show_isotherms=True,
    isotherm_levels=10
)
```

#### `render_heat_flux()`

Render heat flux vectors on surface.

```python
fig = viz.render_heat_flux(
    heat_flux: np.ndarray,            # Nx3 heat flux vectors (W/m²)
    surface_normals: np.ndarray,      # Nx3 surface normals
    geometry: np.ndarray,             # Nx3 surface points
    title: str = "Heat Flux",
    output_path: Optional[str] = None,
) -> Optional[Figure]
```

### Avionics Display Methods

#### `render_sensor_fov()`

Render sensor field of view cone.

```python
fig = viz.render_sensor_fov(
    sensor_position: np.ndarray,      # 3D sensor position (m)
    sensor_orientation: np.ndarray,   # 3D pointing direction
    fov_horizontal: float,            # Horizontal FOV (degrees)
    fov_vertical: float,              # Vertical FOV (degrees)
    range_m: float,                   # Sensor range (m)
    title: str = "Sensor Field of View",
    output_path: Optional[str] = None,
    cone_color: str = "cyan",
    cone_alpha: float = 0.3,
) -> Optional[Figure]
```

**Example:**

```python
# LIDAR sensor
sensor_pos = np.array([0, 0, 10])
sensor_dir = np.array([1, 0, -0.5])

fig = viz.render_sensor_fov(
    sensor_position=sensor_pos,
    sensor_orientation=sensor_dir,
    fov_horizontal=120,
    fov_vertical=30,
    range_m=500,
    title="Forward LIDAR Coverage",
    output_path="sensor_fov.png"
)
```

#### `render_radar_cross_section()`

Render radar cross section polar plot.

```python
fig = viz.render_radar_cross_section(
    geometry: np.ndarray,                       # Nx3 geometry points
    rcs_db: np.ndarray,                        # N RCS values (dBsm)
    frequency_ghz: float,                       # Radar frequency (GHz)
    azimuth_range: tuple[float, float] = (0, 360),
    elevation: float = 0.0,
    title: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Optional[Figure]
```

### Compliance Methods

#### `get_audit_trail()`

Return audit trail of all rendered frames.

```python
trail = viz.get_audit_trail()
# Returns: list[FrameAuditRecord]
```

#### `export_audit_trail()`

Export audit trail to JSON file.

```python
viz.export_audit_trail("audit_trail.json")
```

#### `generate_compliance_report()`

Generate compliance summary report.

```python
report = viz.generate_compliance_report()
# Returns: dict with compliance metrics
```

**Report structure:**

```python
{
    "compliance_mode": "DO178C_LEVEL_A",
    "seed": 42,
    "render_backend": "matplotlib",
    "config_hash": "a1b2c3...",
    "total_frames": 10,
    "total_warnings": 0,
    "render_time_stats": {
        "average_ms": 245.6,
        "min_ms": 198.3,
        "max_ms": 312.8
    },
    "audit_enabled": true
}
```

## Compliance Features

### DO-178C Level A

When using `ComplianceMode.DO178C_LEVEL_A`:

1. **Deterministic Rendering**: All randomness uses seeded RNG
2. **Frame Hashing**: SHA-256 hash computed for each rendered frame
3. **Audit Trail**: Complete record of all rendering operations
4. **Configuration Hash**: Immutable configuration fingerprint
5. **Timestamp**: Nanosecond-precision timestamps

### Audit Record Structure

```python
@dataclass
class FrameAuditRecord:
    frame_id: int                # Sequential frame counter
    timestamp_ns: int            # Nanosecond Unix timestamp
    seed: int                    # RNG seed used
    config_hash: str             # SHA-256 of configuration
    frame_hash: str              # SHA-256 of rendered frame
    render_time_ms: float        # Rendering time in milliseconds
    warnings: list[str]          # Any warnings during render
```

### Deterministic Reproducibility

Same seed + same config = identical frame hash:

```python
# First run
config1 = AerospaceVizConfig(seed=42, enable_audit_log=True)
viz1 = AerospaceVisualizer(config1)
fig1 = viz1.render_flight_trajectory(trajectory, output_path="run1.png")
trail1 = viz1.get_audit_trail()

# Second run (identical)
config2 = AerospaceVizConfig(seed=42, enable_audit_log=True)
viz2 = AerospaceVisualizer(config2)
fig2 = viz2.render_flight_trajectory(trajectory, output_path="run2.png")
trail2 = viz2.get_audit_trail()

# Verify determinism
assert trail1[0].frame_hash == trail2[0].frame_hash
```

## Integration Examples

### With QNX Substrate Results

```python
from qnx.core import SubstrateResult
from qubic.visualization.aerospace import AerospaceVisualizer

# Get results from QNX simulation
result = SubstrateResult.load("simulation_output.h5")

# Extract trajectory data
trajectory = result.get_trajectory()
velocity = result.get_velocity()

# Visualize
viz = AerospaceVisualizer()
fig = viz.render_flight_trajectory(
    trajectory=trajectory,
    velocity=velocity,
    title="QNX Trajectory Analysis"
)
```

### With Ansys FEA Adapter

```python
from sdk.ansys.quasim_ansys_adapter import QuasimAnsysAdapter
from qubic.visualization.aerospace import AerospaceVisualizer

# Load Ansys results
adapter = QuasimAnsysAdapter()
fem_data = adapter.get_fem_results("structural_analysis.rst")

# Extract mesh and stress
nodes = fem_data["nodes"]
elements = fem_data["elements"]
stress = fem_data["stress_tensor"]

# Visualize
viz = AerospaceVisualizer()
fig = viz.render_fem_mesh(
    nodes=nodes,
    elements=elements,
    stress_tensor=stress,
    title="Wing Structure Analysis"
)
```

## Testing

Run the test suite:

```bash
pytest qubic/visualization/aerospace/tests/ -v --cov=qubic.visualization.aerospace
```

Expected coverage: >80%

## Troubleshooting

### Matplotlib Not Available

```
ERROR: Matplotlib required for visualization
```

**Solution:**
```bash
pip install matplotlib
```

### SciPy Not Available (Isotherms)

```
WARNING: scipy not available, skipping isotherm contours
```

**Solution:**
```bash
pip install scipy
```

### Determinism Issues

If frame hashes differ between runs with same seed:

1. Verify matplotlib version is consistent
2. Check that no external randomness is introduced
3. Ensure same numpy version across runs

## Performance Tips

1. **Use draft preset** for development iteration
2. **Reduce streamline_density** for faster airflow rendering
3. **Disable audit logging** when not needed for compliance
4. **Subsample large meshes** before visualization
5. **Use headless backend** for CI/cluster rendering

## License

Apache 2.0 - See LICENSE file

## Contributing

See CONTRIBUTING.md for development guidelines.

## References

- DO-178C: Software Considerations in Airborne Systems and Equipment Certification
- QUBIC Visualization System: `qubic/visualization/qubic_viz.py`
- QRATUM Documentation: Root README.md
