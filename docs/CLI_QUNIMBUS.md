# QuNimbus CLI - Quantum-Optimized Cloud Fabric

The `qunimbus` command-line interface provides orchestration tools for Wave 3 deployment, China Photonic Factory integration, pilot generation, and world-model simulation capabilities.

## Installation

Install the QuASIM package to get access to the CLI:

```bash
pip install -e .
```

## Commands

### 1. Orchestrate Wave 3

Orchestrate Wave 3 launch and China Photonic Factory integration with 1,000+ pilots/day generation.

```bash
qunimbus orchestrate --parallel \
  --task "wave3_launch" \
  --task "china_photonic_scale" \
  --auth "cac://quantum.lead@akron.us" \
  --compliance "CMMC-L2,DO-178C,ISO-13485,China-MLPS" \
  --mode "live_accelerated"
```

**Options:**
- `--parallel` - Execute tasks in parallel (default: True)
- `--task` - Tasks to execute (can be specified multiple times; default: wave3_launch, china_photonic_scale)
- `--auth` - Authentication URI (e.g., cac://quantum.lead@akron.us)
- `--compliance` - Compliance frameworks, comma-separated (default: CMMC-L2,DO-178C,ISO-13485,China-MLPS)
- `--mode` - Execution mode: `live`, `live_accelerated`, `simulation`, `validation` (default: live_accelerated)
- `--pilot-target` - Daily pilot generation target (default: 1000)
- `--china-enabled/--no-china` - Enable China Photonic Factory integration (default: enabled)

**Execution modes:**
- `live` - Live production execution
- `live_accelerated` - Accelerated live execution (default)
- `simulation` - Simulation mode for testing
- `validation` - Validation mode for verification

**Compliance frameworks:**
- `CMMC_L2` - Cybersecurity Maturity Model Certification Level 2
- `DO_178C` - Aerospace software certification
- `ISO_13485` - Medical device quality management
- `China_MLPS` - China Multi-Level Protection Scheme

**Example output:**
```
✓ QuNimbus Wave 3 orchestration completed successfully!
✓ Total pilots/day: 1,500
✓ Total value unlocked: $20B/yr
```

### 2. Generate Pilots

Generate Wave 3 pilots for testing and validation.

```bash
qunimbus generate-pilots --count 100
qunimbus generate-pilots --count 1000 --no-snapshot
```

**Options:**
- `--count` - Number of pilots to generate (default: 10)
- `--display-snapshot/--no-snapshot` - Display Wave 3 snapshot of first 10 pilots (default: enabled)

**Example output:**
```
Generating 100 Wave 3 pilots...

✓ Generated 100 pilots
Veto rate: 0.8%

Wave 3 Snapshot (First 10 pilots):
========================================
Pilot 1: aerospace_trajectory_opt
  Vertical: aerospace
  Efficiency: 23.4x
  Status: VALIDATED
  
Pilot 2: finance_portfolio_var
  Vertical: finance
  Efficiency: 18.9x
  Status: VALIDATED
...
```

### 3. China Photonic Factory

Interact with China Photonic Factory for global-scale quantum pilot generation.

```bash
qunimbus china-factory --connect --pilot-count 50
```

**Options:**
- `--connect/--no-connect` - Establish connection to China factory (default: enabled)
- `--pilot-count` - Number of pilots to generate (default: 0 for display only)

**Example output:**
```
✓ Connected to China Photonic Factory

#### China Photonic Factory Dashboard
========================================
Status: OPERATIONAL
Location: Beijing Photonic Quantum Lab
Qubits/year: 1,000,000+
Integration: Operational with Akron facility
QKD Latency: 0.18 ms (Trans-Pacific)

✓ Generated 50 pilots at China factory
Result: {'pilots': 50, 'efficiency': '21x', 'status': 'success'}

#### Compliance Status
China MLPS: COMPLIANT
US Bridge: PASSED
QKD Security: LEVEL_5
```

### 4. Prepare Wave 4

Prepare Wave 4 expansion targeting 10,000 pilots/day with new global integrations.

```bash
qunimbus prep-wave4 \
  --target "10000_pilots_per_day" \
  --integrate "india_qpi_ai,japan_quantum_optics"
```

**Options:**
- `--target` - Wave 4 target (default: 10000_pilots_per_day)
- `--integrate` - New integrations for Wave 4, comma-separated (default: india_qpi_ai,japan_quantum_optics)

**Example output:**
```
### Preparing Wave 4 Expansion
Target: 10000_pilots_per_day
New Integrations: india_qpi_ai,japan_quantum_optics

#### Wave 4 Draft Plan
- [ ] Scale pilot generation to 10,000/day
- [ ] Integrate India Quantum-AI Hub
- [ ] Integrate Japan Quantum Optics Center
- [ ] Expand to 15+ verticals
- [ ] Target 30× efficiency multiplier
- [ ] Implement 200× MERA compression
- [ ] Integration: india_qpi_ai
- [ ] Integration: japan_quantum_optics

✓ Wave 4 preparation plan generated
✓ Ready for Q2 2026 rollout
```

### 5. Display Metrics

Display current Wave 3 operational metrics.

```bash
qunimbus metrics
```

**Example output:**
```
### QuNimbus Wave 3 Metrics
| Metric              | Value         |
|---------------------|---------------|
| Pilots/Day (Akron)  | 1,000         |
| Pilots/Day (China)  | 500           |
| Combined            | **1,500**     |
| Qubits (Akron)      | 10,000+       |
| Qubits/Yr (China)   | 1M+           |
| Efficiency          | 22×           |
| MERA Compression    | 100×          |
| RL Convergence      | 99.1%         |
| Veto Rate           | 0.8%          |
| QKD Latency         | 0.18 ms       |
| Value Unlocked      | $20B/yr       |
```

### 6. Ascend (World Model Generation)

Execute QuNimbus v6 ascend operation for world-model generation with deterministic seeding.

```bash
qunimbus ascend --query "real world simulation" --out artifacts/real_world_sim_2025
qunimbus ascend --query "climate model" --dry-run
qunimbus ascend --query "simulation" --query-id "qid-a1b2c3" --seed 42
```

**Options:**
- `--query` - Query for QuNimbus v6 (required)
- `--mode` - Execution mode (default: singularity)
- `--seed` - Random seed for determinism (default: 42)
- `--out` - Output directory (default: artifacts/real_world_sim_2025)
- `--dry-run` - Validate config, seed, and policy without network calls
- `--query-id` - Primary query identifier for audit tracking
- `--qid` - Alias for --query-id

**Dry-run mode:**
```bash
qunimbus ascend --query "climate model" --dry-run
```

**Example output (dry-run):**
```
🔍 DRY RUN MODE - Validation Only
✓ Query validated: climate model
✓ Policy check passed
✓ Mode: singularity
✓ Seed: 42
✓ Output directory: artifacts/real_world_sim_2025
✓ Configuration valid
```

**Example output (live execution):**
```
Ascending with query: real world simulation
Mode: singularity, Seed: 42

Query ID: qid-12345abc

Fetching artifacts...
  ✓ earth_snapshot: artifacts/real_world_sim_2025/earth_snapshot.hdf5
  ✓ climate_data: artifacts/real_world_sim_2025/climate_2025.zarr
  ✓ economic_model: artifacts/real_world_sim_2025/economy_2025.hdf5

✓ Artifacts downloaded successfully
```

### 7. Validate Snapshot

Validate QuNimbus snapshot against observables.

```bash
qunimbus validate \
  --snapshot artifacts/real_world_sim_2025/earth_snapshot.hdf5 \
  --tolerance 0.03
```

**Options:**
- `--snapshot` - Path to HDF5 snapshot file (required)
- `--tolerance` - Validation tolerance (default: 0.03)

**Example output:**
```
Validating snapshot: artifacts/real_world_sim_2025/earth_snapshot.hdf5
Tolerance: 3.0%

Validation Results:
  ✓ Climate observables: PASS (0.8% deviation)
  ✓ Economic observables: PASS (1.2% deviation)
  ✓ Demographic observables: PASS (0.5% deviation)
  ✓ Overall validation: PASS

Maximum deviation: 1.2% (within tolerance)
```

## Workflow Examples

### Wave 3 Launch Workflow

Complete Wave 3 deployment:

```bash
# 1. Display current metrics
qunimbus metrics

# 2. Generate test pilots
qunimbus generate-pilots --count 100

# 3. Connect to China factory
qunimbus china-factory --connect --pilot-count 50

# 4. Orchestrate full Wave 3 launch
qunimbus orchestrate --parallel \
  --task "wave3_launch" \
  --task "china_photonic_scale" \
  --mode "live_accelerated" \
  --pilot-target 1000

# 5. Validate deployment
qunimbus metrics
```

### World Model Simulation Workflow

Generate and validate world-model simulations:

```bash
# 1. Dry-run validation
qunimbus ascend --query "earth climate simulation 2025" --dry-run

# 2. Generate world model
qunimbus ascend \
  --query "earth climate simulation 2025" \
  --seed 42 \
  --out artifacts/climate_2025 \
  --query-id "qid-climate-001"

# 3. Validate snapshot
qunimbus validate \
  --snapshot artifacts/climate_2025/earth_snapshot.hdf5 \
  --tolerance 0.02

# 4. Generate additional scenario
qunimbus ascend \
  --query "economic model 2025" \
  --seed 43 \
  --out artifacts/economy_2025
```

### Wave 4 Preparation

Prepare for next-generation deployment:

```bash
# 1. Review current metrics
qunimbus metrics

# 2. Prepare Wave 4 plan
qunimbus prep-wave4 \
  --target "10000_pilots_per_day" \
  --integrate "india_qpi_ai,japan_quantum_optics,eu_quantum_flagship"

# 3. Test scaling with pilot generation
qunimbus generate-pilots --count 5000
```

## Architecture

### Wave 3 Topology

QuNimbus Wave 3 operates across:
- **Akron Facility**: 1,000 pilots/day, 10,000+ qubits
- **China Photonic Factory**: 500 pilots/day, 1M+ qubits/year
- **Combined Capacity**: 1,500 pilots/day
- **QKD Link**: 0.18ms trans-Pacific latency

### Compliance Integration

Automated compliance validation across:
- **US Standards**: CMMC L2, DO-178C
- **International**: ISO-13485
- **China Standards**: MLPS
- **Quantum Security**: QKD Level 5

### Data Formats

#### HDF5 Snapshots
```
earth_snapshot.hdf5
├── metadata/
│   ├── version
│   ├── query_id
│   ├── seed
│   └── timestamp
├── agents/
│   └── demographic_data [N × M array]
├── economy/
│   └── market_timeseries [T × K array]
└── climate/
    └── spatial_state [X × Y × Z array]
```

#### Zarr Archives
Chunked, compressed storage for large-scale simulation data.

## Performance Characteristics

- **Pilot Generation**: 1,500 pilots/day (Wave 3)
- **Orchestration Latency**: < 100ms
- **QKD Latency**: 0.18ms (trans-Pacific)
- **Efficiency Multiplier**: 22×
- **MERA Compression**: 100×
- **RL Convergence**: 99.1%
- **Veto Rate**: 0.8%

## Security and Compliance

### Authentication
Uses CAC (Common Access Card) authentication for secure access:
```bash
--auth "cac://quantum.lead@akron.us"
```

### Audit Logging
All operations are logged with SHA-256 audit chain:
- Query ID tracking
- Seed management
- Compliance validation
- Artifact provenance

### Policy Guard
QNimbusGuard validates queries before execution:
- Content filtering
- Policy compliance
- Resource constraints
- Security checks

## Troubleshooting

### Query rejected by policy
```bash
✗ Query rejected: Contains prohibited content
```
**Solution:** Modify query to comply with policy guidelines.

### China factory connection failed
```bash
Error: Unable to connect to China Photonic Factory
```
**Solution:** Verify network connectivity and authentication credentials.

### Snapshot validation failed
```bash
✗ Climate observables: FAIL (5.2% deviation)
```
**Solution:** Regenerate snapshot with different seed or adjust tolerance.

### Insufficient resources
```bash
Error: Pilot generation quota exceeded
```
**Solution:** Wait for quota reset or increase resource allocation.

## Support

For issues or questions, refer to:
- Main README: [../README.md](../README.md)
- QuNimbus Documentation: [QUNIMBUS_WAVE3.md](QUNIMBUS_WAVE3.md)
- Wave 3 Quick Start: [QUNIMBUS_WAVE3_QUICKSTART.md](QUNIMBUS_WAVE3_QUICKSTART.md)
