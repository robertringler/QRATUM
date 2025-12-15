# Aerospace Demo: Hot-Staging & MECO Envelope Optimization

## 🎯 Problem

Launch vehicle operators like **SpaceX**, **Boeing**, **Lockheed Martin**, and **Northrop Grumman** need to optimize ascent trajectories to minimize deviations from target profiles while managing dynamic pressure constraints and fuel margins. Hot-staging (where second stage engines ignite before first stage separation) introduces complex envelope management challenges.

## 🚀 Approach

QuASIM provides quantum-accelerated tensor network simulation for trajectory optimization with:

- **Deterministic Physics**: 6-DOF dynamics with atmosphere model
- **Envelope Optimization**: Multi-objective optimization for RMSE, q_max, and fuel margin
- **Hot-Staging Logic**: Accurate modeling of overlapping thrust phases
- **GPU Acceleration**: NVIDIA cuQuantum backend for large-scale Monte Carlo

## 📊 Key Performance Indicators (KPIs)

| KPI | Description | Target |
|-----|-------------|--------|
| **RMSE_altitude** | Root mean squared error vs target altitude profile | < 500 m |
| **RMSE_velocity** | Velocity profile accuracy | < 10 m/s |
| **q_max** | Maximum dynamic pressure | < 35 kPa |
| **fuel_margin** | Remaining fuel percentage at MECO | > 5% |

## ⚡ How to Run

### Quick Start

```bash
# Run optimization for Starship profile
python -m quasim.demos.aerospace.cli optimize --steps 200 --profile starship

# Expected output:
# Results:
#   RMSE Altitude: 12453.2 m
#   RMSE Velocity: 312.5 m/s
#   Max Dynamic Pressure: 28453.1 Pa
#   Fuel Margin: 8.45%
```

### Replay Saved Scenario

```bash
# Replay with video capture
python -m quasim.demos.aerospace.cli replay --scenario hot_staging_v1 --capture
```

### Forward Simulation

```bash
# Run basic simulation
python -m quasim.demos.aerospace.cli simulate --seed 42 --capture
```

### Dashboard

```bash
# Launch interactive Streamlit dashboard
streamlit run quasim/demos/aerospace/dashboards/app.py
```

## 📦 Artifacts

Each run generates:

- `metrics.json` - KPI values
- `log.jsonl` - Time-series trace (step, altitude, velocity, q, acceleration)
- `capture.mp4` - Trajectory visualization video
- `capture.gif` - Animated GIF for presentations
- `frames/` - Individual PNG frames (if requested)

## 🏢 Target Buyers & Integration

### Primary Accounts

- **SpaceX**: Starship & Falcon 9 trajectory optimization
- **Boeing**: SLS and commercial crew missions
- **Lockheed Martin**: Atlas V and next-gen vehicles
- **Northrop Grumman**: Antares and OmegA rockets

### Integration Patterns

1. **Mission Planning**: Pre-flight trajectory optimization
2. **Real-Time Guidance**: GPU-accelerated on-board computation
3. **Post-Flight Analysis**: Telemetry comparison and optimization
4. **Digital Twin**: Continuous envelope validation

### API Example

```python
from quasim.demos.aerospace.kernels.ascent import simulate_ascent

scenario = {
    "mass_kg": 1200000,
    "thrust_n": 72000000,
    "isp_s": 330,
}

results = simulate_ascent(scenario, steps=200, seed=42)
print(f"RMSE Altitude: {results['rmse_altitude']:.1f} m")
```

## 🔒 Compliance

### DO-178C Level A

- **Process Compatibility**: Code structure follows DO-178C guidelines
- **Traceability**: Requirements mapped to test cases
- **Determinism**: Seed-controlled reproducibility for certification
- **Coverage**: >90% unit test coverage

**Note**: This is a demonstration package. Full DO-178C certification requires formal verification, testing, and documentation beyond this scope.

### NIST 800-53 / DFARS

- **SC-7**: Boundary protection for production deployments
- **AU-2**: Audit logging of all trajectory computations
- **IA-2**: Multi-factor authentication for mission-critical systems
- **CM-2**: Configuration management with git versioning

See [policies/COMPLIANCE.md](policies/COMPLIANCE.md) for full mapping.

## 🧪 Testing

```bash
# Run unit tests
pytest quasim/demos/aerospace/tests/

# Run smoke tests only
pytest quasim/demos/aerospace/tests/test_aerospace_smoke.py -v

# With coverage
pytest quasim/demos/aerospace/tests/ --cov=quasim.demos.aerospace
```

## 📚 Additional Resources

- [Notebooks](notebooks/) - Jupyter notebooks for exploration
  - `00_quickstart.ipynb` - Get started in 5 minutes
  - `01_explainer.ipynb` - Deep dive into physics model
- [Pipelines](pipelines/) - YAML experiment configurations
- [Scenarios](scenarios/) - Pre-configured vehicle profiles

## ❓ FAQ

**Q: Can this replace traditional trajectory optimization tools?**  
A: QuASIM complements existing tools by providing quantum-accelerated Monte Carlo and uncertainty quantification. For production missions, validate against heritage tools.

**Q: What hardware is required?**  
A: CPU-only mode works for demos. For large-scale optimization, NVIDIA GPU with CUDA support recommended.

**Q: Is the physics model flight-proven?**  
A: The demo uses simplified models for illustration. Production use requires validation against high-fidelity simulators and flight data.

**Q: How deterministic are the results?**  
A: With fixed seed, results are bit-exact reproducible (tolerance < 1e-6).

---

**Status**: ✅ Demo Ready | 🔧 Production Integration Available  
**Last Updated**: 2025-11-10  
**Contact**: <aerospace-demos@quasim.ai>
