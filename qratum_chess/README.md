# QRATUM-Chess Engine

A professional-strength chess engine powered by the QRATUM AI platform.

## Overview

QRATUM-Chess is a complete chess engine system implementing:

- **Stage I**: Core chess logic with bitboard representation
- **Stage II**: Neural evaluation networks (dual-head policy-value)
- **Stage III**: Tri-Modal Cognitive Core (Tactical, Strategic, Conceptual)
- **Stage IV**: Comprehensive benchmarking and load-testing protocol

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    QRATUM-Chess Engine                          │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  Tactical   │  │  Strategic  │  │ Conceptual  │             │
│  │   Cortex    │  │   Cortex    │  │   Cortex    │             │
│  │  (NNUE)     │  │  (RL Net)   │  │(Transformer)│             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│              ┌─────────────────────┐                            │
│              │    Fusion Layer     │                            │
│              │  M = Σ wᵢQᵢ + Ω    │                            │
│              └──────────┬──────────┘                            │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │   Search Engine     │                            │
│              │ (AAS/AB/MCTS)       │                            │
│              └──────────┬──────────┘                            │
│                         ▼                                       │
│              ┌─────────────────────┐                            │
│              │   UCI Interface     │                            │
│              └─────────────────────┘                            │
└─────────────────────────────────────────────────────────────────┘
```

## Components

### Core (`qratum_chess/core/`)
- **BitBoard**: 64-bit integer bitboard representation
- **Position**: Complete chess position state
- **Move**: Move representation with UCI notation

### Neural (`qratum_chess/neural/`)
- **PolicyValueNetwork**: Dual-head residual network
- **PositionEncoder**: 28-channel tensor encoding
- **TriModalCore**: Three co-evolving cognition stacks

### Search (`qratum_chess/search/`)
- **AlphaBetaSearch**: Alpha-beta with pruning optimizations
- **MCTSSearch**: Monte Carlo Tree Search with neural guidance
- **AsymmetricAdaptiveSearch**: Phase-aware adaptive search

### Agents (`qratum_chess/agents/`)
- **BoardManagerAgent**: Position state management
- **EvaluationAgent**: Position assessment
- **MoveProposalAgent**: Move candidate generation
- **RuleValidatorAgent**: Move legality enforcement
- **MetaStrategyDirector**: High-level strategy coordination
- **AgentOrchestrator**: Multi-agent pipeline coordination

### Benchmarks (`qratum_chess/benchmarks/`)
- **PerformanceMetrics**: Core KPI measurement
- **StrategicTortureSuite**: Pathological position testing
- **AdversarialGauntlet**: Engine-vs-engine matches
- **EloCertification**: Rating calculation and validation
- **ResilienceTest**: Failure injection and recovery
- **TelemetryOutput**: Diagnostic data generation
- **AutoBenchmark**: Fully automated benchmarking orchestration
- **MotifExtractor**: Novel chess pattern discovery and classification

### Protocols (`qratum_chess/protocols/`)
- **UCIEngine**: Universal Chess Interface implementation

### Training (`qratum_chess/training/`)
- **SelfPlayGenerator**: Self-play data generation
- **NetworkTrainer**: Neural network training loop
- **KnowledgeDistillation**: Cross-generation learning

## Quick Start

```python
from qratum_chess.core.position import Position
from qratum_chess.search.aas import AsymmetricAdaptiveSearch

# Create starting position
pos = Position.starting()

# Initialize search engine
search = AsymmetricAdaptiveSearch()

# Find best move
best_move, value, stats = search.search(pos, depth=10)
print(f"Best move: {best_move.to_uci()}, Eval: {value:.2f}")
```

## UCI Usage

Run as a UCI engine:

```bash
python -m qratum_chess.protocols.uci
```

Then connect with any UCI-compatible chess GUI.

## Benchmarking

### Manual Benchmarking

```python
from qratum_chess.benchmarks.runner import BenchmarkRunner, BenchmarkConfig
from qratum_chess.search.aas import AsymmetricAdaptiveSearch

config = BenchmarkConfig(
    run_performance=True,
    run_torture=True,
    run_elo=True,
    run_resilience=True
)

runner = BenchmarkRunner(config)
engine = AsymmetricAdaptiveSearch()

summary = runner.run(engine)
runner.print_summary(summary)
```

### Automated Benchmarking with Motif Extraction

Run the complete automated benchmark suite:

```bash
# Full benchmark with certification and motif extraction
python run_full_benchmark.py --certify --extract-motifs

# Quick mode for faster iteration
python run_full_benchmark.py --quick

# Custom configuration
python run_full_benchmark.py \
  --certify \
  --output-dir ./my_results \
  --gpu \
  --torture-depth 20
```

Features:
- **Automated execution**: Complete pipeline from start to finish
- **Environment verification**: Python, dependencies, GPU detection
- **Motif extraction**: Discover novel chess patterns automatically
- **Comprehensive reports**: JSON, CSV, HTML, PGN outputs
- **Stage III certification**: Automatic verification against promotion criteria

See [`qratum_chess/benchmarks/README_AUTOMATION.md`](benchmarks/README_AUTOMATION.md) for detailed usage.

See [`docs/MOTIF_EXTRACTION.md`](../docs/MOTIF_EXTRACTION.md) for motif classification details.

## Performance Targets (Stage IV)

| Metric | Target |
|--------|--------|
| Nodes/sec (single core) | ≥ 70M |
| Nodes/sec (32 threads) | ≥ 1.9B |
| MCTS rollouts/sec | ≥ 500k |
| NN eval latency | ≤ 0.15 ms |
| Hash table hit rate | ≥ 93% |
| Eval volatility | ≤ 0.05 |
| Blunder rate | ≤ 0.01% |

## Certification Criteria

### Stage III Promotion Gate
- ≥ 75% winrate vs Stockfish-NNUE baseline
- ≥ 70% winrate vs Lc0-class nets
- Novel motif emergence confirmed

### Elo Certification
- ELO_QRATUM - ELO_SF17 ≥ +250 for promotion
- Any regression ≥ 10 Elo triggers rollback

## Training

### Self-Play Generation

```python
from qratum_chess.training.selfplay import SelfPlayGenerator, SelfPlayConfig

config = SelfPlayConfig(
    num_games=1000,
    mcts_simulations=800,
    temperature=1.0
)

generator = SelfPlayGenerator(config)
games = generator.generate_games()
```

### Network Training

```python
from qratum_chess.training.trainer import NetworkTrainer, TrainingConfig
from qratum_chess.neural.network import PolicyValueNetwork
from qratum_chess.neural.encoding import PositionEncoder

network = PolicyValueNetwork()
encoder = PositionEncoder()
trainer = NetworkTrainer(network)

samples = generator.get_training_batch(1024)
metrics = trainer.train(samples, encoder, epochs=100)
```

## License

Apache 2.0 License - See main repository LICENSE file.

## Contributing

See main repository CONTRIBUTING.md for contribution guidelines.
