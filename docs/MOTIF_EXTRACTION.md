# Motif Extraction and Classification System

Comprehensive guide to the QRATUM-Chess motif discovery and classification system.

## Overview

QRATUM-Chess "Bob" can discover and classify novel chess patterns (motifs) that diverge from standard engine play. The motif extraction system analyzes telemetry data from benchmark runs to identify these patterns and categorize them by type and novelty.

## What is a Motif?

A **motif** is a novel chess pattern or move sequence that:

1. **Diverges from engine databases** (Stockfish, Lc0)
2. **Shows high novelty pressure** Ω(a)
3. **Activates specific cortex modules** (tactical/strategic/conceptual)
4. **Demonstrates unique characteristics** not found in standard play

## Motif Types

### 1. Tactical Motifs

**Definition**: Short-term combinations involving direct threats and forcing sequences.

**Characteristics**:
- High tactical cortex activation
- Concrete calculation required
- Usually 2-5 moves deep
- Clear material or positional gain

**Examples**:
- Novel pin/fork combinations
- Unexpected deflection sequences
- Creative back-rank tactics
- Unconventional piece sacrifices

**Detection Criteria**:
- Tactical cortex weight > 0.5
- High divergence from engine moves
- Short move sequences (2-5 moves)

### 2. Strategic Motifs

**Definition**: Long-term positional plans involving piece coordination and pawn structures.

**Characteristics**:
- High strategic cortex activation
- Positional evaluation changes
- Planning horizon 5-15 moves
- Subtle improvements

**Examples**:
- Novel pawn structures
- Unconventional piece placements
- Prophylactic maneuvers
- Space control patterns

**Detection Criteria**:
- Strategic cortex weight > 0.5
- Medium divergence from engines
- Medium-length sequences (5-10 moves)

### 3. Opening Motifs

**Definition**: Novel patterns in the opening phase (moves 1-15).

**Characteristics**:
- Discovered in opening positions
- Piece count > 28
- Creates new theoretical lines
- Often strategic in nature

**Examples**:
- New move orders
- Sideline innovations
- Transposition tricks
- Preparation novelties

**Detection Criteria**:
- Game phase = OPENING
- Total pieces > 28
- Position from early game

### 4. Endgame Motifs

**Definition**: Novel patterns in simplified positions (≤6 pieces excluding pawns).

**Characteristics**:
- Discovered in endgame positions
- Low piece count
- Precise calculation required
- Often decisive

**Examples**:
- Zugzwang formations
- Novel conversion techniques
- Opposition tricks
- Fortress breakthroughs

**Detection Criteria**:
- Game phase = ENDGAME
- Piece count ≤ 6 (excluding kings/pawns)
- Late-game position

### 5. Conceptual Motifs

**Definition**: Abstract patterns requiring high-level understanding.

**Characteristics**:
- High conceptual cortex activation
- Non-obvious evaluation
- Requires pattern recognition
- Hard to explain tactically

**Examples**:
- Positional sacrifices
- Compensation schemes
- Dynamic imbalances
- Abstract plans

**Detection Criteria**:
- Conceptual cortex weight > 0.5
- Very high divergence (> 0.7)
- Complex positions

## Novelty Scoring

Each motif receives a novelty score from 0.0 to 1.0 based on:

### Scoring Components

1. **Divergence from Engines** (40%)
   - How different is the move from Stockfish/Lc0?
   - Measured by move difference and evaluation gap
   - Score: 0.0 (identical) to 1.0 (completely different)

2. **Novelty Pressure Ω(a)** (30%)
   - Functional measuring pattern invention pressure
   - Higher for positions requiring creative solutions
   - Calculated during search

3. **Cortex Activation** (20%)
   - Which cortex modules activated?
   - Conceptual activation suggests novel thinking
   - Balanced activation indicates complex patterns

4. **Historical Rarity** (10%)
   - Is pattern seen in previous games?
   - Database lookup (if available)
   - Unseen patterns score higher

### Novelty Levels

- **0.8 - 1.0**: Breakthrough novelty (completely new)
- **0.6 - 0.8**: High novelty (significant divergence)
- **0.4 - 0.6**: Moderate novelty (interesting variation)
- **0.2 - 0.4**: Low novelty (minor improvement)
- **0.0 - 0.2**: Minimal novelty (near-standard play)

## Extraction Process

### 1. Telemetry Capture

During benchmarking, telemetry records:

```python
# Per-move data
telemetry.record_cortex_activation(
    tactical=0.3,
    strategic=0.5,
    conceptual=0.2
)

telemetry.record_novelty_pressure(0.75)

telemetry.record_move_divergence(
    position_fen="...",
    move_uci="e4",
    engine_move="d4",
    divergence_score=0.85
)
```

### 2. Pattern Detection

The `MotifExtractor` analyzes telemetry:

```python
from qratum_chess.benchmarks.motif_extractor import MotifExtractor

extractor = MotifExtractor(
    novelty_threshold=0.6,      # Minimum novelty score
    divergence_threshold=0.5,    # Minimum divergence
    min_cortex_activation=0.3    # Minimum cortex weight
)

motifs = extractor.extract_from_telemetry(telemetry_data)
```

### 3. Classification

Each motif is classified by:

- **Type**: Tactical/Strategic/Opening/Endgame/Conceptual
- **Phase**: Opening/Middlegame/Endgame
- **Novelty**: Score 0.0-1.0
- **Cortex**: Activation weights

### 4. Ranking and Filtering

```python
# Get top motifs by novelty
top_motifs = extractor.get_top_motifs(n=10, sort_by="novelty")

# Filter by type
tactical = extractor.filter_by_type(MotifType.TACTICAL)

# Filter by phase
endgame = extractor.filter_by_phase(GamePhase.ENDGAME)
```

## Motif Data Structure

Each motif contains:

```python
{
    "motif_id": "MOTIF_0042",
    "motif_type": "tactical",
    "game_phase": "middlegame",
    "position_fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4",
    "move_sequence": ["Nxe5", "Nxe5", "d4"],
    "evaluation": 0.45,
    "novelty_score": 0.82,
    "cortex_weights": {
        "tactical": 0.7,
        "strategic": 0.2,
        "conceptual": 0.1
    },
    "novelty_pressure": 0.68,
    "engine_comparison": {
        "engine_move": "d4",
        "selected_move": "Nxe5",
        "divergence": 0.82
    },
    "discovery_timestamp": 1735562422.0,
    "description": "Novel move diverging from engine baseline"
}
```

## Output Formats

### JSON Catalog

Complete motif database:

```json
{
    "total_motifs": 42,
    "extraction_timestamp": 1735562422.0,
    "motifs_by_type": {
        "tactical": 15,
        "strategic": 12,
        "opening": 8,
        "endgame": 5,
        "conceptual": 2
    },
    "motifs": [...]
}
```

### CSV Summary

Tabular format:

```csv
motif_id,type,phase,novelty_score,evaluation,novelty_pressure,move_count,position_fen
MOTIF_0001,tactical,middlegame,0.8200,0.45,0.68,3,"r1bqkb1r/..."
MOTIF_0002,strategic,opening,0.7500,0.12,0.55,5,"rnbqkbnr/..."
```

### PGN Format

Chess game format:

```pgn
[Event "QRATUM-Chess Motif Discovery"]
[Site "Benchmark Suite"]
[Date "2025.12.30"]
[Round "MOTIF_0001"]
[White "Bob (QRATUM)"]
[Black "Analysis"]
[Result "*"]
[FEN "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 4"]
[MotifType "tactical"]
[GamePhase "middlegame"]
[NoveltyScore "0.8200"]
[Evaluation "0.45"]

Nxe5 Nxe5 d4 *
```

### HTML Report

Visual catalog with board diagrams and statistics.

## Analysis Workflow

### 1. Run Benchmark with Motif Extraction

```bash
python run_full_benchmark.py --certify --extract-motifs
```

### 2. Review HTML Report

Open `motifs/motifs_report.html` in browser to see:
- Summary statistics
- Top motifs by novelty
- Distribution by type
- Cortex activation patterns

### 3. Analyze in Chess Software

Import PGN files into chess GUI:

```bash
# All tactical motifs
chess-software tactical_motifs.pgn

# All endgame motifs
chess-software endgame_motifs.pgn
```

### 4. Programmatic Analysis

```python
import json

# Load catalog
with open('motifs/motif_catalog.json') as f:
    catalog = json.load(f)

# Find high-novelty conceptual motifs
conceptual = [
    m for m in catalog['motifs']
    if m['motif_type'] == 'conceptual'
    and m['novelty_score'] > 0.8
]

# Analyze cortex patterns
for motif in conceptual:
    print(f"{motif['motif_id']}: {motif['cortex_weights']}")
```

## Validation

### Comparison to Engine Databases

Motifs are compared against:

1. **Stockfish-NNUE**: Classical engine baseline
2. **Lc0**: Neural network engine
3. **Opening Books**: Theoretical databases
4. **Endgame Tablebases**: Perfect play databases

### Evaluation Verification

- Position evaluation from search
- Comparison to external engines
- Move quality assessment
- Blunder detection

### Novelty Confirmation

- Historical database lookup
- Pattern frequency analysis
- Expert review (optional)
- Peer comparison

## Practical Applications

### 1. Training Data

Novel motifs can be:
- Added to training datasets
- Used for knowledge distillation
- Shared across generations

### 2. Opening Preparation

Opening motifs provide:
- New theoretical lines
- Surprise weapons
- Sideline improvements

### 3. Study Material

Endgame motifs help:
- Learn conversion techniques
- Understand zugzwang
- Master technical positions

### 4. Research

Motif analysis enables:
- Understanding engine creativity
- Studying novel patterns
- Advancing chess theory

## Configuration

### Extraction Thresholds

Adjust in code:

```python
extractor = MotifExtractor(
    novelty_threshold=0.7,      # Higher = fewer, more novel
    divergence_threshold=0.6,    # Higher = more divergent only
    min_cortex_activation=0.4    # Higher = stronger activation
)
```

### Classification Heuristics

Game phase determination:

```python
# Opening: total_pieces >= 28
# Endgame: piece_count <= 6 (excluding kings/pawns)
# Middlegame: everything else
```

Motif type from cortex:

```python
# Dominant cortex determines type
# tactical > 0.5 → TACTICAL
# strategic > 0.5 → STRATEGIC
# conceptual > 0.5 → CONCEPTUAL
```

## Limitations

### Current System

1. **Evaluation**: Uses placeholder evaluation (0.0)
   - Future: Integrate neural evaluator
   
2. **Database Comparison**: Limited engine comparison
   - Future: Add Stockfish/Lc0 API calls
   
3. **Historical Analysis**: No game database lookup
   - Future: Connect to opening/endgame databases
   
4. **Board Visualization**: No inline board diagrams
   - Future: Add SVG board rendering

### Known Issues

- FEN parsing doesn't validate legality
- Cortex activations may be simulated
- Novelty pressure may need calibration
- Classification heuristics are basic

## Future Enhancements

### Planned Features

1. **Deep Analysis**
   - Engine evaluation integration
   - Multi-depth analysis
   - Variation tree exploration

2. **Visualization**
   - Interactive board diagrams
   - Cortex activation heatmaps
   - Search tree visualizations

3. **Machine Learning**
   - Learned classification
   - Novelty prediction
   - Pattern clustering

4. **Database Integration**
   - Opening book queries
   - Tablebase lookups
   - Historical game search

## References

- QRATUM-Chess Architecture: `qratum_chess/README.md`
- Benchmarking System: `qratum_chess/benchmarks/README_AUTOMATION.md`
- Telemetry System: `qratum_chess/benchmarks/telemetry.py`
- Motif Extractor: `qratum_chess/benchmarks/motif_extractor.py`

## Examples

See `benchmarks/auto_run/` for example outputs after running:

```bash
python run_full_benchmark.py --certify --extract-motifs
```
