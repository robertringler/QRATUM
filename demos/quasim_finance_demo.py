#!/usr/bin/env python3
"""

QuASIM × Finance - Portfolio Risk (VaR) Demo

Monte Carlo Value-at-Risk optimization for multi-asset portfolios.
Deterministic, CPU-only, < 60s runtime per profile.

Usage:
    python demos/quasim_finance_demo.py --profile configs/vertical_profiles/finance_var.json
"""

import argparse
import base64
import io
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from quasim.common import (calculate_fidelity, evolutionary_optimization,
                           generate_report, load_profile)


def simulate_portfolio(alpha: float, profile: dict, seed: int = 42) -> dict:
    """

    Simulate portfolio returns using Monte Carlo.

    Alpha controls risk-return trade-off (higher alpha = more aggressive).
    """

    np.random.seed(seed)

    n_simulations = 1000
    n_days = 252  # Trading days in a year

    # Alpha affects portfolio composition
    equity_weight = 0.4 + 0.3 * alpha  # 40-70% equities
    bond_weight = 1.0 - equity_weight

    # Simulate daily returns
    equity_mean = 0.10 / n_days  # 10% annual return
    equity_vol = 0.20 / np.sqrt(n_days)  # 20% annual volatility
    bond_mean = 0.04 / n_days  # 4% annual return
    bond_vol = 0.05 / np.sqrt(n_days)  # 5% annual volatility

    # Run Monte Carlo simulations
    portfolio_returns = []
    for _ in range(n_simulations):
        equity_returns = np.random.normal(equity_mean, equity_vol, n_days)
        bond_returns = np.random.normal(bond_mean, bond_vol, n_days)

        daily_returns = equity_weight * equity_returns + bond_weight * bond_returns
        portfolio_returns.append(np.sum(daily_returns))

    portfolio_returns = np.array(portfolio_returns)

    # Calculate metrics
    annual_return = np.mean(portfolio_returns) * 100  # As percentage
    var_95 = -np.percentile(portfolio_returns, 5) * 100  # 95% VaR
    sharpe = annual_return / (np.std(portfolio_returns) * 100 * np.sqrt(n_days))
    max_drawdown = np.abs(np.min(portfolio_returns)) * 100

    return {
        "returns_distribution": portfolio_returns.tolist(),
        "target_var_pct": float(var_95),
        "target_return_pct": float(annual_return),
        "target_sharpe_ratio": float(sharpe),
        "target_max_drawdown_pct": float(max_drawdown),
        "equity_weight": float(equity_weight),
        "bond_weight": float(bond_weight),
    }


def evaluate_fitness(metrics: dict, profile: dict) -> float:
    """Evaluate portfolio fitness against risk-return targets."""

    targets = profile["targets"]
    tolerances = profile["tolerances"]
    weights = profile["weights"]

    var_error = (
        abs(metrics["target_var_pct"] - targets["target_var_pct"]) / tolerances["var_tolerance_pct"]
    ) * weights["var"]

    return_error = (
        abs(metrics["target_return_pct"] - targets["target_return_pct"])
        / tolerances["return_tolerance_pct"]
    ) * weights["return"]

    sharpe_error = (
        abs(metrics["target_sharpe_ratio"] - targets["target_sharpe_ratio"])
        / tolerances["sharpe_tolerance"]
    ) * weights["sharpe"]

    drawdown_error = (
        abs(metrics["target_max_drawdown_pct"] - targets["target_max_drawdown_pct"])
        / tolerances["drawdown_tolerance_pct"]
    ) * weights["drawdown"]

    fitness = np.sqrt(var_error**2 + return_error**2 + sharpe_error**2 + drawdown_error**2)
    return float(fitness)


def create_visualization(metrics: dict, profile: dict) -> str:
    """Create portfolio performance visualization."""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

    returns = np.array(metrics["returns_distribution"]) * 100

    # Returns distribution
    ax1.hist(returns, bins=50, alpha=0.7, color="blue", edgecolor="black")
    ax1.axvline(x=0, color="r", linestyle="--", label="Break-even")
    ax1.set_xlabel("Annual Return (%)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Portfolio Returns Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # VaR visualization
    var_value = metrics["target_var_pct"]
    ax2.hist(returns, bins=50, alpha=0.7, color="green", edgecolor="black")
    ax2.axvline(
        x=-var_value, color="r", linestyle="--", linewidth=2, label=f"VaR 95%: {var_value:.2f}%"
    )
    ax2.set_xlabel("Annual Return (%)")
    ax2.set_ylabel("Frequency")
    ax2.set_title("Value at Risk (95% confidence)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Risk-Return scatter
    ax3.scatter([np.std(returns)], [np.mean(returns)], s=200, c="red", marker="o")
    ax3.scatter(
        [profile["targets"]["target_var_pct"] * 0.3],
        [profile["targets"]["target_return_pct"]],
        s=200,
        c="green",
        marker="x",
        label="Target",
    )
    ax3.set_xlabel("Risk (Volatility %)")
    ax3.set_ylabel("Return (%)")
    ax3.set_title("Risk-Return Profile")
    ax3.legend(["Actual", "Target"])
    ax3.grid(True, alpha=0.3)

    # Portfolio allocation
    allocations = [metrics["equity_weight"] * 100, metrics["bond_weight"] * 100]
    labels = [f"Equities\n{allocations[0]:.1f}%", f"Bonds\n{allocations[1]:.1f}%"]
    colors = ["#ff9999", "#66b3ff"]
    ax4.pie(allocations, labels=labels, colors=colors, autopct="", startangle=90)
    ax4.set_title("Portfolio Allocation")

    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=100)
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    return image_base64


def main():
    parser = argparse.ArgumentParser(description="QuASIM × Finance Demo")
    parser.add_argument("--profile", type=str, required=True, help="Path to finance profile JSON")
    parser.add_argument("--generations", type=int, default=50, help="Number of generations")
    parser.add_argument("--pop", type=int, default=20, help="Population size")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    print(f"Loading profile: {args.profile}")
    profile = load_profile(args.profile)
    print(f"Profile: {profile['name']}")

    print("\nRunning evolutionary optimization...")
    best_alpha, best_metrics, best_fitness, history = evolutionary_optimization(
        simulate_portfolio,
        evaluate_fitness,
        profile,
        generations=args.generations,
        population_size=args.pop,
        seed=args.seed,
    )

    print("\nOptimization complete!")
    print(f"  Best alpha: {best_alpha:.6f}")
    print(f"  VaR (95%): {best_metrics['target_var_pct']:.2f}%")
    print(f"  Expected Return: {best_metrics['target_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {best_metrics['target_sharpe_ratio']:.4f}")

    viz_base64 = create_visualization(best_metrics, profile)

    fidelity = calculate_fidelity(
        best_metrics,
        profile["targets"],
        ["target_var_pct", "target_return_pct", "target_sharpe_ratio"],
    )

    validation_metrics = {"fidelity": fidelity, "fitness_rmse": best_fitness}

    output_file = f"{Path(args.profile).stem}_demo_report.json"

    generate_report(
        profile,
        best_alpha,
        best_fitness,
        best_metrics,
        history,
        {"generations": args.generations, "population_size": args.pop, "seed": args.seed},
        viz_base64,
        validation_metrics,
        profile.get("compliance_tags", []),
        output_file,
    )

    print(f"\nReport saved: {output_file}")
    print(f"  Fidelity: {fidelity:.4f}")
    print("\nDemo complete! ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
