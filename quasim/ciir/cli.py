"""CIIR CLI entry point: ``quasim-ciir``.

Commands
--------
run        Run a CIIR evolution
benchmark  Run the full validation suite
config     Generate a default configuration file
info       Display theory tuple summary
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
import numpy as np

from quasim.ciir.config import CIIRConfig


@click.group()
@click.version_option(version="0.1.0", prog_name="quasim-ciir")
def cli():
    """QuASIM CIIR — Constraint-Induced Interface Realism engine.

    Executable bridge between CIIR formal theory and QuASIM tensor runtime.
    """


@cli.command()
@click.option("--config", "config_path", type=click.Path(), default=None, help="YAML config file")
@click.option("--steps", default=200, help="Number of evolution steps")
@click.option("--lr", default=0.01, help="Learning rate / time step")
@click.option("--batch", default=4, help="Batch size")
@click.option("--seed", default=42, help="Random seed")
@click.option("--json-output", is_flag=True, help="Output results as JSON")
def run(config_path, steps, lr, batch, seed, json_output):
    """Run a CIIR state evolution."""
    from quasim.ciir.evolution import IntegrationMethod, evolve
    from quasim.ciir.loss import CIIRLoss
    from quasim.ciir.theory import build_default_theory

    if config_path:
        cfg = CIIRConfig.from_yaml(config_path)
    else:
        cfg = CIIRConfig(n_steps=steps, lr=lr, batch_size=batch, seed=seed)

    theory = build_default_theory(
        rank=cfg.rank,
        rep_dim=cfg.rep_dim,
        n_constraints=cfg.n_constraints,
        n_observers=cfg.n_observers,
        seed=cfg.seed,
    )

    rng = np.random.default_rng(cfg.seed)
    rho = theory.manifold.density_matrix(
        theory.manifold.random_state(batch=cfg.batch_size, rng=rng)
    )

    loss_fn = CIIRLoss(
        theory=theory,
        entropy_weight=cfg.entropy_weight,
    )

    method_map = {
        "gradient": IntegrationMethod.GRADIENT,
        "projected_gradient": IntegrationMethod.PROJECTED_GRADIENT,
        "strang_splitting": IntegrationMethod.STRANG_SPLITTING,
    }
    method = method_map.get(cfg.method, IntegrationMethod.PROJECTED_GRADIENT)

    t0 = time.time()
    result = evolve(
        loss_fn,
        rho,
        n_steps=cfg.n_steps,
        lr=cfg.lr,
        method=method,
        tol=cfg.tol,
        record_every=cfg.record_every,
    )
    elapsed = time.time() - t0

    output = {
        "converged": result.converged,
        "n_steps": result.n_steps,
        "initial_loss": result.losses[0] if result.losses else None,
        "final_loss": result.losses[-1] if result.losses else None,
        "elapsed_s": round(elapsed, 3),
        "config": cfg.to_dict(),
    }

    if json_output:
        click.echo(json.dumps(output, indent=2, default=str))
    else:
        click.echo("CIIR Evolution Complete")
        click.echo("=" * 50)
        click.echo(f"  Steps:      {result.n_steps}")
        click.echo(f"  Converged:  {result.converged}")
        click.echo(f"  Loss:       {output['initial_loss']:.4f} → {output['final_loss']:.4f}")
        click.echo(f"  Elapsed:    {elapsed:.2f}s")
        if result.loss_components:
            final = result.loss_components[-1]
            click.echo(f"  Components: constraint={final['constraint']:.4f} "
                       f"observer={final['observer']:.4f} "
                       f"entropy={final['entropy']:.4f}")


@cli.command()
@click.option("--json-output", is_flag=True, help="Output results as JSON")
def benchmark(json_output):
    """Run the full CIIR validation suite."""
    from quasim.ciir.benchmarks import run_all_benchmarks

    if not json_output:
        click.echo("CIIR Validation Suite")
        click.echo("=" * 50)

    results = run_all_benchmarks(verbose=not json_output)

    if json_output:
        output = [
            {
                "name": r.name,
                "passed": r.passed,
                "metrics": r.metrics,
                "elapsed_s": round(r.elapsed_s, 3),
            }
            for r in results
        ]
        click.echo(json.dumps(output, indent=2, default=str))
    else:
        n_pass = sum(1 for r in results if r.passed)
        click.echo(f"\n{n_pass}/{len(results)} benchmarks passed")


@cli.command("config")
@click.option("--output", "-o", default="configs/ciir_default.yaml", help="Output path")
def generate_config(output):
    """Generate a default CIIR configuration file."""
    cfg = CIIRConfig()
    cfg.to_yaml(output)
    click.echo(f"Default configuration written to {output}")


@cli.command()
def info():
    """Display CIIR theory tuple summary."""
    from quasim.ciir.theory import build_default_theory

    theory = build_default_theory()
    checks = theory.validate()

    click.echo("CIIR Theory Tuple T_CIIR = (M, {C_i}, {O_j}, Π)")
    click.echo("=" * 50)
    click.echo(f"  State manifold:  R={theory.manifold.rank}, D={theory.manifold.rep_dim}")
    click.echo(f"  Constraints:     {len(theory.constraints)}")
    click.echo(f"  Observers:       {len(theory.observers)}")
    click.echo(f"  Interface map:   {'defined' if theory.interface else 'none'}")
    if theory.interface:
        click.echo(f"  K_min:           {theory.interface.kappa_min}")
        click.echo(f"  Diameter:        {theory.interface.diameter}")
        click.echo(f"  Contraction:     {theory.interface.contraction_factor:.6f}")

    # Commutation check
    if len(theory.observers) >= 2:
        comm = theory.observers[0].commutator(theory.observers[1])
        comm_norm = float(np.linalg.norm(comm))
        click.echo(f"  [O_0, O_1] norm: {comm_norm:.6f} "
                    f"({'non-commuting' if comm_norm > 1e-10 else 'commuting'})")

    click.echo("\nAxiom Validation:")
    for k, v in checks.items():
        status = "✓" if v else "✗"
        click.echo(f"  [{status}] {k}")


if __name__ == "__main__":
    cli()
