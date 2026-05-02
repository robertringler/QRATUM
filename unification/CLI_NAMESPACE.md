# QRATUM CLI Namespace
_Phase_: U1 Canonical Architecture  
_Generated_: 2026-04-30

## Canonical CLI Entry Points

All QRATUM commands are available via the unified namespace.

### Current `pyproject.toml` scripts

| Command | Entry point | Status |
|---------|------------|--------|
| `quasim-hcal` | `quasim.hcal.cli:main` | wired |
| `quasim-revultra` | `quasim.cli.revultra_cli:cli` | wired |
| `quasim-qgh` | `quasim.cli.qgh_cli:cli` | wired |
| `quasim-terc-obs` | `quasim.cli.terc_obs_cli:cli` | wired |
| `quasim-own` | `quasim.cli.quasim_own:cli` | wired |
| `quasim-tire` | `quasim.cli.tire_cli:cli` | wired |
| `qunimbus` | `quasim.qunimbus.cli:main` | wired |
| `qnx` | `qnx.cli:cli` | wired |
| `qstack` | `qstack.launch:main` | wired |
| `xenon` | `xenon.cli:cli` | wired |
| `qubic-viz` | `qubic.visualization.cli:cli` | wired |
| `quasim-ciir` | `quasim.ciir.cli:cli` | wired |

### New CLI Runners (GAP-CIIR-RUN-001/002)

| Script | Usage |
|--------|-------|
| `run_multi_qubit_ciir.py` | `python run_multi_qubit_ciir.py [--n-qubits N] [--steps S] [--seed SEED] [--json]` |
| `run_falsification.py` | `python run_falsification.py [--N N] [--n-steps S] [--N-scan 2 4 6] [--seed SEED] [--json]` |

### Planned: Unified `qratum <subcmd>` namespace

Per U1 architecture, all commands should eventually be accessible as:

```
qratum ric-step         # RIC pipeline step
qratum ciir-demo        # CIIR-CRS-RIC demo
qratum multi-qubit      # multi-qubit CIIR controller
qratum falsify          # CIIR falsification protocol
qratum plasma           # plasma reconnection control
qratum benchmark        # run benchmarks
qratum verify           # full-stack verification
```

**Status**: Top-level `qratum` dispatcher is not yet implemented. Individual runners and `quasim-*` scripts are the current canonical entry points.
