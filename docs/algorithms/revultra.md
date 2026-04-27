# REVULTRA Algorithms

## Overview

REVULTRA (Quantum-Inspired Cryptanalytic Algorithms) provides a suite of advanced methods for analyzing ciphertexts using holographic entropy, temporal embeddings, spectral analysis, and pattern mining.

## Core Algorithms

### 1. Quantum Information Topology

Maps character probabilities to complex quantum amplitudes using holographic encoding.

**Math**: For character `c` with probability `p`:

- Amplitude: `α = √p`
- Phase: `φ = exp(i * qcorr * log(p + ε))`
- Quantum amplitude: `ψ_c = α * φ`

**Use Case**: Identify character distribution patterns in encrypted text.

### 2. Holographic Entropy Transform

Transforms text into 2D grid and computes holographic entropy via FFT.

**Math**:

1. Reshape text into square grid
2. Apply 2D FFT: `F(u,v) = FFT2D(grid)`
3. Compute entropy: `H = -Σ p_i log(p_i)` where `p_i = |F_i| / Σ|F_j|`

**Use Case**: Detect structural patterns in polyalphabetic ciphers.

### 3. Temporal Embedding Sequence

Generates time-series embeddings with sliding windows.

**Math**: For window size `w`:

- Embedding at position `t`: `E_t = [c_t, c_{t+1}, ..., c_{t+w-1}] * W_temporal`
- Temporal weights: `W_i = exp(-teq * i / w)`

**Use Case**: Analyze character transition patterns.

### 4. Memory Recursive Pattern Mining

Identifies repeating patterns using recursive search.

**Algorithm**:

1. For each length `l` in [min_len, max_len]
2. Extract all substrings of length `l`
3. Count occurrences
4. Return patterns with count ≥ 2

**Use Case**: Find repeated plaintext or key patterns.

### 5. Index of Coincidence (IoC) Tensor

Computes IoC for different period lengths to detect polyalphabetic periodicities.

**Math**: For period `p`:

- Split text into `p` cosets
- For each coset: `IoC = Σ n_i(n_i-1) / (N(N-1))`
- Average across cosets

**Use Case**: Determine Vigenère or similar cipher periods.

### 6. Spectral Autocorrelation

FFT-based autocorrelation for periodicity detection.

**Math**:

1. Normalize sequence: `x' = x - mean(x)`
2. FFT: `X = FFT(x', 2N)`
3. Power spectrum: `P = X * conj(X)`
4. Autocorr: `R = IFFT(P)`

**Use Case**: Identify periodic structures in frequency domain.

### 7. Emergent Complexity Score

Combines multiple complexity measures.

**Components**:

- Character entropy: `H = -Σ p_i log(p_i)`
- Pattern density: `ρ = patterns / length`
- IoC variance: `σ²(IoC)`
- **Score**: `(H*10 + ρ*50 + σ²*100) / 3`

**Use Case**: Overall cipher complexity assessment.

### 8. Frequency Analysis

Basic character frequency counting.

**Output**: Percentage distribution of each character.

### 9. Chi-Squared Test

Tests deviation from expected frequency distribution.

**Math**: `χ² = Σ (observed - expected)² / expected`

**Use Case**: Compare against English or other language frequencies.

### 10. Kryptos K4 Analysis

Complete analysis pipeline combining all methods.

## CLI Usage

### Analyze Ciphertext

```bash
quasim-revultra analyze --ciphertext "OBKRUOXOGHULBSOLIFBBWFLRVQQPRNGKSSOTWTQSJQSSEKZZWATJKLUDIAWINFBNYPVTTMZFPK" \
  --export results.json \
  --max-period 20
```

### Frequency Analysis

```bash
quasim-revultra frequency --ciphertext "ATTACKATDAWN"
```

### Run Demos

```bash
quasim-revultra demo --section kryptos --export demo.json
```

## Python API

```python
from quasim.revultra import REVULTRAAlgorithms

# Initialize with deterministic RNG
rev = REVULTRAAlgorithms(rng=np.random.default_rng(42))

# Analyze ciphertext
text = "CRYPTOTEXT" * 10

# Get complexity score
complexity = rev.emergent_complexity_score(text)
print(f"Complexity: {complexity['score']:.2f}")

# Detect periods
ioc = rev.index_of_coincidence_tensor(text, max_period=20)
print(f"IoC values: {ioc}")

# Find patterns
patterns = rev.memory_recursive_pattern_mining(text)
print(f"Found {len(patterns)} patterns")
```

## JSON Output Schema

```json
{
  "input_length": 100,
  "frequency_analysis": {"A": 12.5, "B": 8.3, ...},
  "complexity": {
    "score": 45.2,
    "entropy": 3.14,
    "pattern_density": 0.15,
    "ioc_variance": 0.023
  },
  "ioc_tensor": [0.065, 0.042, 0.38, ...],
  "ioc_peaks": [3, 6, 9],
  "patterns": {"ABC": 5, "XYZ": 3},
  "chi_squared": 125.4
}
```

## TERC Integration

REVULTRA metrics feed into TERC validation:

- **Beta entropy**: Normalized character entropy → Tier-1 state coherence
- **Beta complexity**: Overall complexity → Tier-5 robustness metric
- **Beta coherence**: Inverse IoC variance → System stability
- **Beta periodicity**: Autocorrelation peaks → Synchronization quality

See `docs/terc_bridge.md` for details.

## References

- Kryptos K4: <https://en.wikipedia.org/wiki/Kryptos>
- Index of Coincidence: Friedman, W.F. (1920)
- Holographic Entropy: Information-theoretic cryptanalysis
