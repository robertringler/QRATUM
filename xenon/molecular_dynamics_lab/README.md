# 🧬 Advanced Molecular Dynamics Lab

An interactive 3D molecular visualization and simulation platform with WebXR/VR support, built on 3Dmol.js and integrated with the XENON bioinformatics framework.

## Features

### 🔬 Molecular Visualization
- **3Dmol.js Integration**: Web-based molecular viewer similar to JSmol
- **Multiple Render Styles**: Cartoon, stick, sphere, line, surface
- **Color Schemes**: Chain, secondary structure, spectrum, element
- **Interactive Controls**: Rotate, zoom, pan with mouse/touch
- **Atom Selection**: Click atoms to view properties
- **Labels & Annotations**: Add custom labels and distance measurements

### 📂 PDB Loading
- **RCSB Integration**: Load structures directly from the Protein Data Bank
- **File Upload**: Support for local PDB files
- **Caching**: Downloaded structures are cached locally
- **Complete Parsing**: ATOM, HETATM, CONECT, HEADER records

### ⚗️ Molecular Docking
- **Interactive Docking**: Real-time ligand positioning
- **Binding Site Detection**: Automatic or manual pocket definition
- **Scoring Functions**: Lennard-Jones, electrostatic, H-bond
- **Multiple Poses**: Generate and rank binding conformations
- **Visualization**: Display interactions and contacts

### ⚡ Molecular Dynamics
- **Real-time Simulation**: Velocity Verlet integration
- **Force Field**: Lennard-Jones + electrostatic
- **Thermostat**: Berendsen temperature control
- **Energy Minimization**: Steepest descent optimizer
- **Trajectory Export**: Save simulation trajectories

### 🥽 WebXR VR Support
- **Immersive Mode**: Full VR headset support
- **Hand Tracking**: Natural hand gesture recognition
- **Controller Input**: Standard VR controller support
- **Molecule Manipulation**: Grab and rotate molecules in VR

### 📳 Haptic Feedback
- **Predefined Patterns**: Selection, hover, bond formation
- **Force Feedback**: Convert molecular forces to haptics
- **Dual Controllers**: Left/right hand support
- **Customizable Intensity**: Adjust feedback strength

## Installation

The module is part of the XENON bioinformatics framework:

```bash
# Install from the QRATUM repository
cd /workspaces/QRATUM
pip install -e .
```

### Dependencies

- `numpy` - Numerical computing
- `requests` - HTTP requests for RCSB
- Python 3.10+

Frontend dependencies (loaded via CDN):
- 3Dmol.js - Molecular visualization
- Three.js - WebXR support

## Quick Start

### Start the Web Server

```bash
# Run the demo script
python -m xenon.molecular_dynamics_lab.demo

# Or start directly with server
python -m xenon.molecular_dynamics_lab.demo --server

# Custom host/port
python -m xenon.molecular_dynamics_lab.demo --server --host 0.0.0.0 --port 9000
```

Open http://localhost:8080 in your browser.

### Programmatic Usage

```python
from xenon.molecular_dynamics_lab import (
    MolecularViewer,
    ViewerConfig,
    PDBLoader,
    DockingEngine,
    MDSimulator,
)

# Load a structure
loader = PDBLoader()
structure = loader.load_from_rcsb("1CRN")

# Create viewer
config = ViewerConfig(background_color="#000000", webxr_enabled=True)
viewer = MolecularViewer(config)

# Set structure and generate HTML
viewer.set_structure(structure)
html = viewer.generate_html()

# Run docking
engine = DockingEngine()
engine.set_receptor(receptor_structure)
engine.set_ligand(ligand_structure)
engine.auto_detect_binding_site()
result = engine.dock()

# Run MD simulation
simulator = MDSimulator()
simulator.load_structure(structure)
simulator.initialize_velocities()
trajectory = simulator.run(num_steps=1000)
```

## API Reference

### PDBLoader

```python
loader = PDBLoader(cache_dir=Path("~/.pdb_cache"))

# Load from RCSB
structure = loader.load_from_rcsb("1CRN")

# Load from file
structure = loader.load_from_file("protein.pdb")

# Load from string
structure = loader.load_from_string(pdb_content)
```

### MolecularViewer

```python
config = ViewerConfig(
    width=800,
    height=600,
    background_color="#000000",
    ambient_light=0.4,
    webxr_enabled=True,
)

viewer = MolecularViewer(config)
viewer.set_structure(structure)

# Add style
viewer.add_style(
    Selection(chain="A"),
    StyleSpec(style="cartoon", color="spectrum")
)

# Add labels
viewer.add_label(
    text="Active Site",
    position=(10.0, 20.0, 30.0)
)

# Generate output
html = viewer.generate_html()
js_code = viewer.generate_js()
```

### DockingEngine

```python
config = DockingConfig(
    search_exhaustiveness=8,
    num_poses=10,
    flexible_residues=["ARG45", "TYR67"],
)

engine = DockingEngine(config)
engine.set_receptor(receptor)
engine.set_ligand(ligand)

# Define binding site
engine.define_binding_site(
    center=(10.0, 20.0, 30.0),
    size=(20.0, 20.0, 20.0)
)

# Or auto-detect
engine.auto_detect_binding_site()

# Run docking
result = engine.dock()
print(f"Best score: {result.best_pose.score} kcal/mol")
```

### MDSimulator

```python
config = MDConfig(
    timestep=0.002,  # ps
    temperature=300.0,  # K
    thermostat="berendsen",
)

simulator = MDSimulator(config)
simulator.load_structure(structure)
simulator.initialize_velocities()

# Run simulation
trajectory = simulator.run(num_steps=10000, save_frequency=100)

# Access results
for frame in trajectory:
    print(f"Step {frame.step}: T={frame.temperature:.1f}K")
```

### VRController

```python
config = VRConfig(
    enable_hand_tracking=True,
    molecule_scale=0.1,
)

controller = VRController(config)

# Generate JavaScript for VR
session_js = controller.generate_session_init_js()
hand_js = controller.generate_hand_tracking_js()
ui_html = controller.generate_vr_controls_ui()
```

### HapticEngine

```python
config = HapticConfig(
    intensity=1.0,
    enabled=True,
)

engine = HapticEngine(config)

# Play predefined feedback
engine.play_pattern("atom_selected", "right")

# Force-based feedback
engine.feedback_from_force(force_magnitude=10.0, hand="right")

# Generate JavaScript
haptic_js = engine.generate_haptic_js()
```

## REST API

When running the server, these endpoints are available:

### `GET /`
Returns the main application HTML page.

### `GET /api/status`
Returns server status.

### `GET /api/structure/<pdb_id>`
Load a PDB structure from RCSB.

**Response:**
```json
{
  "pdb_id": "1CRN",
  "num_atoms": 327,
  "num_residues": 46,
  "chains": ["A"],
  "raw_content": "..."
}
```

### `POST /api/dock`
Run molecular docking.

**Request:**
```json
{
  "receptor_pdb": "3HTB",
  "ligand_pdb": "ATP",
  "binding_site": {
    "center": [10, 20, 30],
    "size": [20, 20, 20]
  }
}
```

### `POST /api/simulate`
Run MD simulation.

**Request:**
```json
{
  "pdb_id": "1CRN",
  "num_steps": 1000,
  "temperature": 300
}
```

## Architecture

```
molecular_dynamics_lab/
├── __init__.py          # Public API exports
├── demo.py              # Demo script
├── README.md            # This file
├── core/
│   ├── __init__.py
│   ├── pdb_loader.py    # PDB file loading
│   ├── molecular_viewer.py  # 3Dmol.js viewer
│   ├── docking_engine.py    # Docking simulation
│   └── md_simulator.py      # MD simulation
├── webxr/
│   ├── __init__.py
│   ├── vr_controller.py     # WebXR VR support
│   └── haptic_engine.py     # Haptic feedback
└── web/
    ├── __init__.py
    └── server.py            # HTTP/WS server
```

## Browser Compatibility

| Feature | Chrome | Firefox | Safari | Edge |
|---------|--------|---------|--------|------|
| 3Dmol.js | ✅ | ✅ | ✅ | ✅ |
| WebGL | ✅ | ✅ | ✅ | ✅ |
| WebXR | ✅ | ✅* | ❌ | ✅ |
| Hand Tracking | ✅ | ❌ | ❌ | ✅ |
| Haptics | ✅ | ❌ | ❌ | ✅ |

*Firefox requires WebXR flag to be enabled

## VR Headset Support

| Device | Supported | Notes |
|--------|-----------|-------|
| Meta Quest 2/3 | ✅ | Full support including hand tracking |
| Meta Quest Pro | ✅ | Full support |
| Valve Index | ✅ | Controller only |
| HTC Vive | ✅ | Controller only |
| Windows MR | ✅ | Controller only |

## Development

### Running Tests

```bash
pytest xenon/molecular_dynamics_lab/tests/
```

### Code Style

This module follows the XENON and QRATUM coding standards:
- PEP 8 compliant
- Type hints required
- Docstrings for all public APIs
- Maximum line length: 100 characters

## License

Part of the QRATUM project. See [LICENSE](../../../LICENSE) for details.

## References

- [3Dmol.js](https://3dmol.org/) - Molecular visualization library
- [RCSB PDB](https://www.rcsb.org/) - Protein Data Bank
- [WebXR API](https://immersive-web.github.io/webxr/) - VR/AR web standard
- [PDB Format](https://www.wwpdb.org/documentation/file-format) - PDB file format specification
