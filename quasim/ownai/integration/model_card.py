"""Model card auto-generator for QuASIM-Own models."""

from datetime import datetime
from pathlib import Path

from quasim.ownai.eval.benchmark import BenchmarkResult


def generate_model_card(
    model_name: str,
    results: list[BenchmarkResult],
    output_path: Path,
    description: str = "",
) -> None:
    """Generate a model card for a QuASIM-Own model.

    Parameters
    ----------
    model_name : str
        Model name
    results : list[BenchmarkResult]
        Benchmark results for this model
    output_path : Path
        Output path for model card
    description : str
        Model description
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter results for this model
    model_results = [r for r in results if r.model_name == model_name]

    if not model_results:
        return

    # Compute aggregated metrics
    primary_scores = [r.primary_metric for r in model_results]
    latencies = [r.latency_p50 for r in model_results]

    avg_primary = sum(primary_scores) / len(primary_scores)
    avg_latency = sum(latencies) / len(latencies)

    # Check determinism
    hashes = [r.prediction_hash for r in model_results]
    deterministic = len(set(hashes)) == len(set((r.task, r.dataset, r.seed) for r in model_results))

    # Generate card
    card = f"""# Model Card: {model_name}

## Model Details

**Name:** {model_name}  
**Version:** 0.1.0  
**Type:** Symbolic-Latent Transformer (QuASIM-Own)  
**Framework:** QuASIM-Own / scikit-learn  
**Generated:** {datetime.now().isoformat()}  

{description}

## Intended Use

This model is designed for deterministic, auditable AI tasks including:
- Tabular classification and regression
- Text classification
- Vision classification
- Time series forecasting

## Performance Summary

**Average Primary Metric:** {avg_primary:.4f}  
**Average Inference Latency (p50):** {avg_latency:.2f} ms  
**Deterministic:** {"✅ Yes" if deterministic else "❌ No"}  

## Evaluation Results

### Tasks Evaluated

"""

    # Group by task
    task_groups = {}
    for r in model_results:
        task_key = f"{r.task}/{r.dataset}"
        if task_key not in task_groups:
            task_groups[task_key] = []
        task_groups[task_key].append(r)

    for task_key, group in sorted(task_groups.items()):
        card += f"\n#### {task_key}\n\n"
        card += "| Metric | Value |\n"
        card += "|--------|-------|\n"

        primary_mean = sum(r.primary_metric for r in group) / len(group)
        secondary_mean = sum(r.secondary_metric for r in group) / len(group)
        latency_mean = sum(r.latency_p50 for r in group) / len(group)

        card += f"| Primary Metric | {primary_mean:.4f} |\n"
        card += f"| Secondary Metric | {secondary_mean:.4f} |\n"
        card += f"| Latency (p50) | {latency_mean:.2f} ms |\n"
        card += f"| Runs | {len(group)} |\n"

    card += """

## Model Architecture

The model uses the Symbolic-Latent Transformer architecture, which combines:

1. **REVULTRA Symbolic Features:**
   - Index of Coincidence (IoC) tensor
   - Spectral autocorrelation
   - Entropy motifs

2. **Learned Representations:**
   - Task-specific embeddings
   - Ensemble learning (Random Forest baseline)

3. **QGH Causal Tracking:**
   - Append-only ledger for run history
   - Prediction hash tracking for consensus

4. **TERC Observables:**
   - Stability margin monitoring
   - Emergent complexity tracking

## Training

All models are trained with:
- Deterministic random seeding
- Fixed data shuffling
- No dropout or stochastic components
- Reproducible initialization

## Ethical Considerations

This model is designed for transparent, auditable AI applications. Users should:
- Validate predictions on their specific data
- Monitor for distribution shift
- Ensure appropriate use cases
- Consider fairness implications

## Limitations

- Performance may vary on out-of-distribution data
- Computational requirements depend on task complexity
- Not optimized for real-time streaming applications

## References

- QuASIM-Own: Deterministic AI with Symbolic-Latent Transformers
- REVULTRA: Cryptanalysis algorithms for symbolic feature extraction
- QGH: Quantum-inspired causal history tracking
- TERC: Tier-based observability framework
"""

    with open(output_path, "w") as f:
        f.write(card)
