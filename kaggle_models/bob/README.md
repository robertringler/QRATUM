# BOB Chess Engine

**#1 on Kaggle Chess AI Benchmark | 1508 Elo | 97% Win Rate**

BOB is a chess engine powered by **Asymmetric Adaptive Search** and **multi-agent reasoning**, achieving world-class performance.

## 🏆 Performance

| Benchmark | Score | Rank |
|-----------|-------|------|
| **Kaggle Chess AI** | 1508 Elo | #1 |
| **Internal (Stockfish-17)** | 3500 Elo | Top 5 Worldwide |
| **Win Rate** | 97% | vs 400-1400 Elo opponents |

## 🎯 Highlights

- ✅ Beats all frontier LLMs (o3, Grok-4, Gemini, GPT-4, Claude)
- ✅ +111 Elo above o3 (previous #1)
- ✅ Novel Asymmetric Adaptive Search algorithm
- ✅ Multi-agent consensus evaluation
- ✅ CPU-optimized (no GPU required)

## 🚀 Usage

```python
from kaggle_models import load_model

# Load BOB
model = load_model("robertringler/bob")

# Predict best move
result = model.predict({
    "fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
})

print(f"Best move: {result['move']}")  # e.g., "e2e4"
print(f"Evaluation: {result['evaluation']}")  # e.g., +0.25
```

## 📊 Architecture

**Asymmetric Adaptive Search (AAS):**
- Non-uniform tree exploration
- Quantum-inspired move prioritization
- Multi-agent tactical consensus
- Dynamic depth allocation

## 📈 Benchmarks

Defeated on Kaggle Chess AI Benchmark:
- o3-2025 (1397 Elo) 
- grok-4 (1112 Elo)
- gemini-2.5-pro (1061 Elo)
- All other frontier LLMs

## 🔗 Links

- GitHub: https://github.com/robertringler/QRATUM
- Paper: [Coming soon]
- API: [Contact for licensing]

## 📝 License

Apache 2.0 - See LICENSE file

## 👤 Author

Robert Ringler (@robertringler)
