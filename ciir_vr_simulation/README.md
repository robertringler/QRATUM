# CIIR → QuASIM → QRATUM VR Simulation

Fully immersive VR and desktop-ready 3D simulation of the CIIR → QuASIM → QRATUM cognitive pipeline.

## Architecture

```
CIIR Module          QuASIM Module         QRATUM Module
┌──────────┐        ┌──────────────┐      ┌───────────────────┐
│CIIRState │──map──→│QuASIMTensor  │──exe──→│QRATUMHardware    │
│CIIRObserver│      │TensorRuntime │      │ExecutionScheduler │
│CIIRVisualizer│    │TensorVisualizer│    │PerformanceDashboard│
└──────────┘        └──────────────┘      └───────────────────┘
      ↓                    ↓                       ↓
                    Vulkan Renderer
                    ┌──────────────────┐
                    │VulkanContext     │
                    │VRContext (OpenXR)│
                    │Camera / Mesh    │
                    └──────────────────┘
```

## Mathematical Pipeline

Each frame:
1. **CIIR State**: `X ∈ R^{B×R×D} → ρ = X^T X / Tr(X^T X)`
2. **Loss**: `L(ρ) = Σ λ_i C_i(ρ)² + Σ μ_j |⟨O_j⟩ - y_j|² + γ·Ω(ρ)`
3. **Gradient Step**: `ρ_{t+1} = Π_C(ρ_t − η ∇L(ρ_t))`
4. **Render**: Manifold morphing (vertex shader) + tensor heatmaps (fragment shader)

## Building

```bash
mkdir build && cd build
cmake .. -DENABLE_VR=OFF -DENABLE_VALIDATION=ON
make -j$(nproc)
```

### With VR (requires OpenXR SDK):
```bash
cmake .. -DENABLE_VR=ON
```

## Running

```bash
# Desktop mode (200 steps, screenshots every 50)
./ciir_vr_sim --steps 200 --screenshot-interval 50 --output ciir_output

# High-resolution with VR
./ciir_vr_sim --vr --dim 16 --constraints 5 --observers 3

# Full options
./ciir_vr_sim --help
```

## Shaders

| Shader | Stage | Purpose |
|---|---|---|
| `manifold.vert` | Vertex | Dynamic CIIR manifold morphing based on constraint intensity |
| `manifold.frag` | Fragment | Cool-warm heatmap coloring, Blinn-Phong lighting, constraint glow |
| `heatmap.vert` | Vertex | Height-field displacement for loss landscape |
| `heatmap.frag` | Fragment | Inferno colormap with contour lines |
| `tensor_contraction.comp` | Compute | GPU batch `Tr(K_i · ρ)` tensor contraction |
| `gradient_update.comp` | Compute | GPU projected gradient `ρ ← ρ − η∇L` |

## Python Bridge

The simulation is also fully available in Python:

```python
from quasim.ciir.simulation import run_and_capture_simulation, SimulationConfig

engine = run_and_capture_simulation(
    config=SimulationConfig(
        n_steps=200,
        screenshot_interval=50,
        output_dir="ciir_output",
    ),
)
```

Or via CLI:
```bash
quasim-ciir simulate --steps 200 --screenshot-interval 50 --output ciir_output
```

## Output

```
ciir_output/
├── screenshots/        # PNG frames at intervals
│   ├── sim_000050.png
│   ├── sim_000100.png
│   └── ...
├── metrics.csv         # Per-step: loss, gradient, purity, entropy
├── tensor_log.json     # Tensor states and metadata
├── dashboard.png       # Metrics visualization
└── performance_dashboard.png
```

## Project Structure

```
ciir_vr_simulation/
├── CMakeLists.txt
├── include/
│   ├── simulation_engine.hpp      # Main engine + run_and_capture_simulation()
│   ├── ciir/
│   │   ├── ciir_state.hpp         # GPU-resident manifold state
│   │   ├── ciir_observer.hpp      # Non-commutative observers
│   │   └── ciir_visualizer.hpp    # 3D mesh generation
│   ├── quasim/
│   │   ├── quasim_tensor.hpp      # Multi-rank tensor (up to rank-8)
│   │   ├── tensor_runtime.hpp     # Gradient evolution engine
│   │   └── tensor_visualizer.hpp  # Loss landscape & gradient viz
│   ├── qratum/
│   │   ├── qratum_hardware.hpp    # Hardware abstraction (HCAL/qstack/qnx)
│   │   ├── execution_scheduler.hpp # Multi-threaded priority scheduler
│   │   └── performance_dashboard.hpp # Dear ImGui dashboard
│   └── renderer/
│       ├── vulkan_context.hpp     # Vulkan 1.3+ with compute pipeline
│       ├── vr_context.hpp         # OpenXR VR/AR integration
│       ├── mesh.hpp               # Vertex layout with constraint attributes
│       └── camera.hpp             # FPS / orbit / VR camera
├── src/                           # Full C++20 implementations
│   ├── main.cpp
│   ├── simulation_engine.cpp
│   ├── ciir/ quasim/ qratum/ renderer/
└── shaders/                       # GLSL 460 shaders
    ├── manifold.{vert,frag}
    ├── heatmap.{vert,frag}
    ├── tensor_contraction.comp
    └── gradient_update.comp
```
