# NGAD Defense Simulation Demo

This directory contains the `open_ngad_demo.py` script, which simulates Next Generation Air Dominance (NGAD) defensive engagement scenarios.

## Features

- **Configurable Parameters**: Control number of shots and threats via command-line
- **Multiple Threat Types**: Simulates cruise missiles, ballistic missiles, fighter aircraft, and drones
- **Observables**: Track engagement time, kill rates, fuel consumption, and g-loads
- **Visualization**: Generates plots including fuel histograms and g-load scatter plots
- **Reproducibility**: Deterministic results with seed control
- **Error Handling**: Robust exception handling for production use
- **JSON Output**: Results saved in structured format for analysis

## Quick Start

```bash
# Run with default parameters (20 shots, 10 threats)
python examples/open_ngad_demo.py

# Customize simulation
python examples/open_ngad_demo.py --shots 30 --threats 15 --seed 123

# Run without generating plots
python examples/open_ngad_demo.py --no-plots

# View help
python examples/open_ngad_demo.py --help
```

## Output

The simulation generates:

1. **Console Summary**: Real-time statistics including kill rates, engagement times, and fuel usage
2. **JSON Results**: Detailed shot-by-shot data in `ngad_results.json` (configurable)
3. **Plots** (in `plots/` directory by default):
   - `fuel_histogram.png`: Distribution of remaining fuel levels
   - `g_load_vs_fuel.png`: Scatter plot showing relationship between g-load and fuel
   - `fuel_depletion.png`: Time series of fuel consumption
   - `kill_rate_by_threat.png`: Success rate by threat type

## Reproducibility

Results are fully reproducible when using the same seed:

```bash
# First run
python examples/open_ngad_demo.py --seed 42 --output run1.json

# Second run (identical results)
python examples/open_ngad_demo.py --seed 42 --output run2.json

# Verify
diff run1.json run2.json  # Should show no differences
```

## Requirements

- Python 3.10+
- numpy
- matplotlib

Install with:
```bash
pip install numpy matplotlib
```

## Use Cases

This simulation is designed for:

- Defense mission planning and analysis
- Weapon system evaluation
- Training scenario generation
- Monte Carlo analysis of engagement outcomes
- Fuel and resource management studies

## Compliance

The script follows QuASIM coding standards:
- DO-178C compatible structure
- Deterministic reproducibility for certification
- Comprehensive error handling
- Full documentation and type hints

## Related

- See PR #202 for original requirements
- Part of the QuASIM aerospace/defense demo suite
