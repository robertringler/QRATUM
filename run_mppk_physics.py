#!/usr/bin/env python
r"""Run the Minimal Programmable Physics Kernel (MPPK) Law-Discovery Engine.

Executes the full 6-step evolution loop:
  1. Simulate — run kernel for T timesteps, store full trajectory
  2. Extract structure — detect fixed points, limit cycles, chaos, clusters
  3. Mine invariants — conserved quantities, symmetries, spectral stability
  4. Evaluate fitness — F = α·stability + β·structure + γ·boundedness + δ·compress
  5. Mutate rules — modify interaction fn, topology, stochasticity, coupling
  6. Select — retain top-k rule systems by fitness

Then infers emergent laws via the meta-learning layer.

Usage:
    python run_mppk_physics.py [--generations N] [--nodes N] [--dim D] [--timesteps T]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from quasim.ciir.swarm.mppk import (GenerationResult, MetaLearningReport,
                                    MPPKEngine)
from quasim.ciir.swarm.orchestrator import Orchestrator


def format_generation(g: GenerationResult) -> dict:
    """Convert a GenerationResult to a JSON-serializable dict."""
    return {
        "generation": g.generation,
        "rule_name": g.rule.name,
        "interaction_type": g.rule.interaction_type.name,
        "noise_sigma": g.rule.noise_sigma,
        "has_coupling_matrix": g.rule.coupling_matrix is not None,
        "trajectory_timesteps": g.trajectory.T,
        "final_energy": g.trajectory.energies[-1] if g.trajectory.energies else None,
        "energy_range": [
            float(min(g.trajectory.energies)),
            float(max(g.trajectory.energies)),
        ] if g.trajectory.energies else None,
        "final_momentum_norm": float(np.linalg.norm(g.trajectory.momenta[-1]))
            if g.trajectory.momenta else None,
        "structure": {
            "has_fixed_point": g.structure.has_fixed_point,
            "has_limit_cycle": g.structure.has_limit_cycle,
            "is_chaotic": g.structure.is_chaotic,
            "n_clusters": g.structure.n_clusters,
            "symmetry_breaking": g.structure.symmetry_breaking,
            "lyapunov_proxy": round(g.structure.lyapunov_proxy, 6),
        },
        "invariants": {
            "energy_conserved": g.invariants.energy_conserved,
            "energy_rel_var": round(g.invariants.energy_rel_var, 6),
            "momentum_conserved": g.invariants.momentum_conserved,
            "momentum_rel_var": round(g.invariants.momentum_rel_var, 6),
            "norm_conserved": g.invariants.norm_conserved,
            "norm_rel_var": round(g.invariants.norm_rel_var, 6),
            "spectral_gap": round(g.invariants.spectral_gap, 6),
            "spectral_stable": g.invariants.spectral_stable,
        },
        "fitness": {
            "stability": round(g.fitness.stability, 4),
            "structure_formation": round(g.fitness.structure_formation, 4),
            "boundedness": round(g.fitness.boundedness, 4),
            "compressibility": round(g.fitness.compressibility, 4),
            "total": round(g.fitness.total, 4),
        },
    }


def format_meta(meta: MetaLearningReport) -> dict:
    """Convert MetaLearningReport to a JSON-serializable dict."""
    return {
        "effective_force_law": meta.effective_force_law,
        "stable_invariants": meta.stable_invariants,
        "embedding_dim": meta.embedding_dim,
        "explained_variance_ratio": round(meta.explained_variance_ratio, 4),
        "candidate_laws": meta.candidate_laws,
    }


def run_mppk(
    generations: int = 10,
    n_nodes: int = 12,
    dim: int = 4,
    timesteps: int = 100,
    n_candidates: int = 5,
    seed: int = 42,
) -> dict:
    """Run the MPPK engine and return structured results."""
    print("=" * 72)
    print("  MINIMAL PROGRAMMABLE PHYSICS KERNEL — Law-Discovery Engine")
    print("=" * 72)
    print("\n  Configuration:")
    print(f"    Nodes:       {n_nodes}")
    print(f"    Dimensions:  {dim}")
    print(f"    Timesteps:   {timesteps}")
    print(f"    Generations: {generations}")
    print(f"    Candidates:  {n_candidates}/generation")
    print(f"    Seed:        {seed}")
    print()

    engine = MPPKEngine(
        n_nodes=n_nodes,
        dim=dim,
        seed=seed,
    )

    t0 = time.time()
    results, meta = engine.evolve(
        generations=generations,
        T=timesteps,
        n_candidates=n_candidates,
    )
    elapsed = time.time() - t0

    # Print generation-by-generation results
    print("-" * 72)
    print("  EVOLUTION LOG")
    print("-" * 72)
    for g in results:
        fp = "FP" if g.structure.has_fixed_point else "  "
        lc = "LC" if g.structure.has_limit_cycle else "  "
        ch = "CH" if g.structure.is_chaotic else "  "
        sb = "SB" if g.structure.symmetry_breaking else "  "
        ec = "EC" if g.invariants.energy_conserved else "  "
        mc = "MC" if g.invariants.momentum_conserved else "  "
        ss = "SS" if g.invariants.spectral_stable else "  "
        print(
            f"  Gen {g.generation:3d} | "
            f"Rule: {g.rule.interaction_type.name:12s} | "
            f"F={g.fitness.total:.4f} | "
            f"E={g.trajectory.energies[-1]:8.3f} | "
            f"{fp} {lc} {ch} {sb} | "
            f"{ec} {mc} {ss} | "
            f"Clust={g.structure.n_clusters}"
        )

    # Print meta-learning results
    print()
    print("-" * 72)
    print("  META-LEARNING REPORT")
    print("-" * 72)
    print(f"  Effective force law:       {meta.effective_force_law}")
    print(f"  Stable invariants:         {meta.stable_invariants}")
    print(f"  Embedding dimension:       {meta.embedding_dim}")
    print(f"  Explained variance ratio:  {meta.explained_variance_ratio:.4f}")
    print("  Candidate physical laws:")
    for law in meta.candidate_laws:
        print(f"    • {law}")

    # Best generation
    best = max(results, key=lambda g: g.fitness.total)
    print()
    print("-" * 72)
    print("  BEST RULE SYSTEM")
    print("-" * 72)
    print(f"  Rule:                {best.rule.name}")
    print(f"  Interaction type:    {best.rule.interaction_type.name}")
    print(f"  Noise σ:             {best.rule.noise_sigma}")
    print(f"  Coupling matrix:     {'Yes' if best.rule.coupling_matrix is not None else 'No'}")
    print(f"  Total fitness:       {best.fitness.total:.4f}")
    print(f"    Stability:         {best.fitness.stability:.4f}")
    print(f"    Structure:         {best.fitness.structure_formation:.4f}")
    print(f"    Boundedness:       {best.fitness.boundedness:.4f}")
    print(f"    Compressibility:   {best.fitness.compressibility:.4f}")

    print()
    print(f"  Elapsed time: {elapsed:.2f}s")
    print("=" * 72)

    return {
        "config": {
            "n_nodes": n_nodes,
            "dim": dim,
            "timesteps": timesteps,
            "generations": generations,
            "n_candidates": n_candidates,
            "seed": seed,
        },
        "generations": [format_generation(g) for g in results],
        "meta_learning": format_meta(meta),
        "best_generation": format_generation(best),
        "elapsed_seconds": round(elapsed, 2),
    }


def run_orchestrator(
    generations: int = 3,
    steps_per_sim: int = 30,
    population_size: int = 3,
    dim: int = 4,
    seed: int = 42,
) -> dict:
    """Run the multi-agent swarm Orchestrator."""
    print()
    print("=" * 72)
    print("  MULTI-AGENT SWARM ORCHESTRATOR")
    print("=" * 72)
    print(f"    Generations:    {generations}")
    print(f"    Steps/sim:      {steps_per_sim}")
    print(f"    Population:     {population_size}")
    print(f"    Dimensions:     {dim}")
    print()

    orc = Orchestrator(dim=dim, seed=seed)
    t0 = time.time()
    results = orc.run_evolution(
        generations=generations,
        steps_per_sim=steps_per_sim,
        population_size=population_size,
    )
    elapsed = time.time() - t0

    print("-" * 72)
    print("  SWARM EVOLUTION LOG")
    print("-" * 72)
    for r in results:
        print(
            f"  Gen {r.generation:3d} | "
            f"Fitness: C={r.best_fitness.consistency:.2f} "
            f"E={r.best_fitness.empirical_fit:.2f} "
            f"S={r.best_fitness.simplicity:.2f} "
            f"N={r.best_fitness.novelty:.2f} "
            f"→ T={r.best_fitness.total:.4f} | "
            f"Pop={r.population_size} | "
            f"Mem={r.memory_size}"
        )

    best = max(results, key=lambda r: r.best_fitness.total) if results else None
    if best:
        print()
        print(f"  Best generation: {best.generation}")
        print(f"  Best total fitness: {best.best_fitness.total:.4f}")
        print(f"  Knowledge graph size: {best.memory_size} nodes")

    print(f"\n  Elapsed time: {elapsed:.2f}s")
    print("=" * 72)

    return {
        "config": {
            "generations": generations,
            "steps_per_sim": steps_per_sim,
            "population_size": population_size,
            "dim": dim,
        },
        "results": [
            {
                "generation": r.generation,
                "fitness": {
                    "consistency": round(r.best_fitness.consistency, 4),
                    "empirical_fit": round(r.best_fitness.empirical_fit, 4),
                    "simplicity": round(r.best_fitness.simplicity, 4),
                    "novelty": round(r.best_fitness.novelty, 4),
                    "total": round(r.best_fitness.total, 4),
                },
                "population_size": r.population_size,
                "memory_size": r.memory_size,
            }
            for r in results
        ],
        "best_generation": best.generation if best else None,
        "best_fitness": round(best.best_fitness.total, 4) if best else None,
        "elapsed_seconds": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run MPPK Law-Discovery Engine and Swarm Orchestrator",
    )
    parser.add_argument("--generations", type=int, default=10,
                        help="Number of evolution generations (default: 10)")
    parser.add_argument("--nodes", type=int, default=12,
                        help="Number of graph nodes (default: 12)")
    parser.add_argument("--dim", type=int, default=4,
                        help="State vector dimension (default: 4)")
    parser.add_argument("--timesteps", type=int, default=100,
                        help="Simulation timesteps per generation (default: 100)")
    parser.add_argument("--candidates", type=int, default=5,
                        help="Candidate mutations per generation (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path")
    parser.add_argument("--skip-orchestrator", action="store_true",
                        help="Skip the swarm orchestrator run")
    args = parser.parse_args()

    # Run MPPK engine
    mppk_results = run_mppk(
        generations=args.generations,
        n_nodes=args.nodes,
        dim=args.dim,
        timesteps=args.timesteps,
        n_candidates=args.candidates,
        seed=args.seed,
    )

    # Run swarm orchestrator
    orc_results = None
    if not args.skip_orchestrator:
        orc_results = run_orchestrator(
            generations=min(args.generations, 5),
            steps_per_sim=min(args.timesteps, 50),
            population_size=min(args.candidates, 3),
            dim=args.dim,
            seed=args.seed,
        )

    # Combine and save
    output = {
        "mppk": mppk_results,
        "orchestrator": orc_results,
    }

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        class _NumpyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return super().default(obj)

        out_path.write_text(json.dumps(output, indent=2, cls=_NumpyEncoder))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
