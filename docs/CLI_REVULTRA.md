# QuASIM REVULTRA CLI - Cryptanalytic Analysis Tools

The `quasim-revultra` command-line interface provides quantum-inspired algorithms for ciphertext analysis including holographic entropy, temporal embeddings, spectral analysis, and pattern mining.

## Installation

Install the QuASIM package to get access to the CLI:

```bash
pip install -e .
```

## Commands

### 1. Analyze Ciphertext

Run comprehensive REVULTRA analysis on ciphertext.

```bash
quasim-revultra analyze --ciphertext "ATTACKATDAWN" --export results.json
quasim-revultra analyze --file ciphertext.txt --max-period 30 --export analysis.json
```

**Options:**
- `--ciphertext` - Ciphertext string to analyze (either this or `--file` required)
- `--file` - Path to ciphertext file (either this or `--ciphertext` required)
- `--plot` - Generate plots (requires matplotlib)
- `--export` - Export results to JSON file (optional)
- `--max-period` - Maximum period for IoC analysis (default: 20)

**Analysis includes:**
- Quantum information topology
- Holographic entropy transform
- Temporal embeddings
- Pattern mining
- Index of Coincidence (IoC) analysis
- Spectral autocorrelation
- Emergent complexity scoring

**Example output:**
```
Analyzing ciphertext (length: 97)...

--- Summary ---
Complexity Score: 3.45
Chi-squared: 245.67
IoC Peaks at periods: [5, 10, 15]
Pattern count: 23

Results exported to: results.json
```

### 2. Run Demonstrations

Run REVULTRA demonstration examples.

```bash
quasim-revultra demo --section kryptos
quasim-revultra demo --section all --export demo_results.json
```

**Options:**
- `--section` - Demo section to run: `kryptos`, `simple`, `all` (default: `kryptos`)
- `--export` - Export demo results to JSON file (optional)

**Available demos:**

#### Kryptos K4 Analysis
Analysis of the famous unsolved Kryptos K4 cipher sculpture.

```bash
quasim-revultra demo --section kryptos --export kryptos_analysis.json
```

**Example output:**
```
======================================================================
REVULTRA Kryptos K4 Analysis Demo
======================================================================
Analyzing: OBKRUOXOGHULBSOLIFBBWFLRVQQPRNGKSSOT...

Complexity Score: 4.23
Entropy: 3.8456
IoC Peaks: [7, 14]
Patterns detected: 45

Results exported to: kryptos_analysis.json
```

### 3. Frequency Analysis

Perform frequency analysis on ciphertext.

```bash
quasim-revultra frequency --ciphertext "HELLO WORLD"
quasim-revultra frequency --file message.txt
```

**Options:**
- `--ciphertext` - Ciphertext to analyze (either this or `--file` required)
- `--file` - Path to ciphertext file (either this or `--ciphertext` required)

**Example output:**
```
Character Frequencies:
----------------------------------------
A:   8.50%
B:   1.20%
C:   3.40%
D:   4.10%
E:  12.70%
...
```

## Workflow Example

Complete cryptanalysis workflow:

```bash
# 1. Start with frequency analysis
quasim-revultra frequency --file cipher.txt

# 2. Run comprehensive analysis
quasim-revultra analyze --file cipher.txt --max-period 25 --export full_analysis.json

# 3. Run demo on known cipher for comparison
quasim-revultra demo --section kryptos --export kryptos_baseline.json

# 4. Compare complexity scores and patterns
# Review exported JSON files for insights
```

## Algorithms Overview

### Quantum Information Topology

Analyzes the topological structure of quantum information in ciphertext.

**Metrics:**
- Topological features
- Quantum state representations
- Information flow patterns

### Holographic Entropy Transform

Applies holographic principles to analyze entropy distribution.

**Features:**
- Surface-to-volume entropy ratio
- Holographic bounds
- Information density maps

### Temporal Embeddings

Creates temporal embeddings of ciphertext for pattern detection.

**Capabilities:**
- Time-series analysis
- Periodicity detection
- Temporal correlation

### Pattern Mining

Discovers recursive and emergent patterns in ciphertext.

**Techniques:**
- Memory-recursive pattern mining
- N-gram analysis
- Structural pattern detection

### Index of Coincidence (IoC)

Computes IoC tensor across multiple period lengths to detect periodicity.

**Analysis:**
- Period detection
- Key length estimation
- Peak identification

### Spectral Autocorrelation

Analyzes spectral properties and autocorrelation of ciphertext.

**Metrics:**
- Autocorrelation function
- Spectral peaks
- Frequency domain analysis

### Emergent Complexity Score

Quantifies the complexity and randomness of ciphertext.

**Components:**
- Shannon entropy
- Chi-squared statistic
- Pattern complexity
- Information theoretic measures

## Analysis Output Format

The JSON export contains comprehensive analysis results:

```json
{
  "input_length": 97,
  "frequency_analysis": {
    "A": 8.5,
    "B": 1.2,
    ...
  },
  "complexity": {
    "score": 3.45,
    "entropy": 3.84,
    "chi_squared": 245.67
  },
  "ioc_tensor": [0.045, 0.052, 0.067, ...],
  "ioc_peaks": [5, 10, 15],
  "autocorrelation": [1.0, 0.23, 0.15, ...],
  "autocorr_peaks": [7],
  "holographic_entropy": 3.21,
  "surface_shape": [10, 10],
  "topology_size": 156,
  "patterns": [
    {"pattern": "AB", "count": 5, "positions": [2, 15, 23, ...]},
    ...
  ]
}
```

## Use Cases

### Cryptanalysis
Analyze unknown ciphers and encrypted messages using quantum-inspired algorithms.

### Cipher Classification
Classify ciphers by type (substitution, transposition, etc.) based on complexity metrics.

### Key Length Detection
Detect key lengths in polyalphabetic ciphers using IoC peaks.

### Pattern Discovery
Discover hidden patterns and structures in encrypted data.

### Complexity Assessment
Assess the strength and complexity of encryption schemes.

## Performance Characteristics

- **Analysis Time**: O(n²) for length n ciphertext
- **Memory**: O(n) for typical analyses
- **IoC Computation**: O(n × p) for max period p
- **Pattern Mining**: O(n log n) average case

## Advanced Usage

### Custom Period Range

For analyzing ciphers with suspected long key lengths:

```bash
quasim-revultra analyze --ciphertext "$CIPHER" --max-period 50 --export long_key.json
```

### Batch Analysis

Analyze multiple ciphertexts:

```bash
for file in ciphers/*.txt; do
  quasim-revultra analyze --file "$file" --export "results/$(basename $file .txt).json"
done
```

### Integration with Other Tools

Combine with external tools:

```bash
# Preprocess with external tool
cat message.enc | decode_tool > preprocessed.txt

# Analyze with REVULTRA
quasim-revultra analyze --file preprocessed.txt --export analysis.json

# Post-process results
python postprocess.py analysis.json
```

## Troubleshooting

### Must specify ciphertext or file
```bash
Error: Must specify either --ciphertext or --file
```
**Solution:** Provide either `--ciphertext "TEXT"` or `--file path/to/file.txt`.

### File not found
```bash
Error: File not found: cipher.txt
```
**Solution:** Verify the file path and ensure the file exists.

### Plotting not implemented
```bash
Note: Plotting requires matplotlib (not implemented in this version)
```
**Solution:** Use `--export` to save results and visualize with external tools.

### Memory issues with large ciphertexts
```bash
MemoryError: Unable to allocate tensor
```
**Solution:** Reduce `--max-period` or process the ciphertext in chunks.

## References

### Kryptos K4
The Kryptos sculpture at CIA headquarters contains four encrypted messages. Three have been solved, but K4 remains unsolved since 1990.

**REVULTRA Analysis:**
- Complexity Score: 4.23 (high)
- Suspected period: 7 or 14
- Multiple pattern classes detected

### Research Applications
REVULTRA algorithms are used in:
- Academic cryptanalysis research
- Historical cipher analysis
- Encryption strength assessment
- Information theory studies

## Support

For issues or questions, refer to:
- Main README: [../README.md](../README.md)
- REVULTRA Algorithm Documentation: [algorithms/revultra.md](algorithms/revultra.md)
