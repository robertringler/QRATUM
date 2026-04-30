"""Main simulation runner."""

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from .audit.causality_audit import CausalityAuditor
from .diagnostics.chi_effective import compute_chi_effective, compute_chi_statistics
from .diagnostics.s_lab import compute_s_lab, check_s_lab_significance
from .diagnostics.validity import check_validity
from .io.artifacts import ArtifactManager
from .models.cptp_process_tensor import CPTPProcessTensorModel
from .models.lattice_field import LatticeFieldModel
from .sensitivity.anti_holographic import analyze_bulk_vs_boundary_sensitivity
from .utils.seeding import generate_seed_sequence


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def generate_parameter_grid(grid_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate parameter grid from configuration.
    
    Args:
        grid_config: Grid configuration
        
    Returns:
        List of parameter dictionaries
    """
    parameters = grid_config.get('parameters', {})
    fixed = grid_config.get('fixed', {})
    
    # Get all parameter combinations
    param_names = list(parameters.keys())
    param_values = [parameters[name] for name in param_names]
    
    grid = []
    for combo in itertools.product(*param_values):
        config = dict(zip(param_names, combo))
        config.update(fixed)
        grid.append(config)
    
    return grid


def run_single_simulation(
    model_type: str,
    config: Dict[str, Any],
    seed: int,
    main_config: Dict[str, Any],
    auditor: CausalityAuditor
) -> Dict[str, Any]:
    """
    Run a single simulation with given configuration.
    
    Args:
        model_type: 'cptp' or 'lattice'
        config: Model configuration
        seed: Random seed
        main_config: Main configuration
        auditor: Causality auditor
        
    Returns:
        Results dictionary
    """
    # Create model
    if model_type == 'cptp':
        model = CPTPProcessTensorModel(config, seed)
    elif model_type == 'lattice':
        model = LatticeFieldModel(config, seed)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Compute S_lab diagnostic
    n_samples = main_config.get('n_samples_s_lab', 100)
    s_lab_result = compute_s_lab(model, n_samples=n_samples)
    
    # Run twin simulations for audit
    results_I0, results_I1 = model.run_twin_simulations()
    
    # Verify causality
    causality_passed = auditor.verify_causality(model, results_I0, results_I1)
    
    # Compute chi (only for lattice models)
    chi_result = None
    validity_result = None
    
    if model_type == 'lattice':
        chi_result = compute_chi_effective(model, results_I0)
        
        # Check validity
        metric_data = model.effective_metric_data
        thresholds = main_config.get('validity_thresholds', {})
        validity_result = check_validity(metric_data, thresholds)
    
    # Check S_lab significance
    threshold_sigma = main_config.get('statistical_thresholds', {}).get('s_lab_significance_sigma', 5.0)
    s_lab_significant = check_s_lab_significance(s_lab_result, threshold_sigma)
    
    # Compile results
    result = {
        'model_type': model_type,
        'config': config,
        'seed': seed,
        's_lab': s_lab_result['s_lab'],
        's_lab_stderr': s_lab_result['s_lab_stderr'],
        's_lab_p_value': s_lab_result['p_value'],
        's_lab_significant': s_lab_significant,
        'causality_passed': causality_passed,
    }
    
    # Add config parameters to result for easy filtering
    result.update(config)
    
    # Add chi results if available
    if chi_result is not None:
        result['chi'] = chi_result.get('chi')
        result['flow_velocity'] = chi_result.get('flow_velocity')
        result['c_eff'] = chi_result.get('c_eff')
    else:
        result['chi'] = None
    
    # Add validity results
    if validity_result is not None:
        result['validity_passed'] = validity_result['all_pass']
        result['dispersion_deviation'] = validity_result['values']['dispersion_deviation']
        result['attenuation_margin'] = validity_result['values']['attenuation_margin']
        result['reconstruction_error'] = validity_result['values']['reconstruction_error']
    else:
        result['validity_passed'] = None
    
    return result


def run_experiment(config_path: str):
    """
    Run full experiment from configuration.
    
    Args:
        config_path: Path to main configuration file
    """
    print(f"Loading configuration from {config_path}")
    main_config = load_config(config_path)
    
    # Create artifact manager
    artifacts = ArtifactManager(main_config.get('output_dir', 'runs'))
    run_dir = artifacts.create_run_directory()
    print(f"Created run directory: {run_dir}")
    
    # Initialize auditor
    auditor = CausalityAuditor(main_config)
    
    # Generate seeds
    base_seed = main_config.get('random_seed_base', 42)
    n_seeds = main_config.get('n_seeds_per_config', 5)
    
    # Collect all results
    all_results = []
    parameter_grids = {}
    
    # Process each model grid
    for model_spec in main_config.get('model_grids', []):
        model_type = model_spec['type']
        grid_config_path = model_spec['config_file']
        
        print(f"\nProcessing {model_type} model grid from {grid_config_path}")
        
        # Load grid configuration
        grid_config = load_config(grid_config_path)
        param_grid = generate_parameter_grid(grid_config)
        
        print(f"Generated {len(param_grid)} parameter configurations")
        parameter_grids[model_type] = param_grid
        
        # Run simulations for each configuration and seed
        for i, config in enumerate(param_grid):
            seeds = generate_seed_sequence(base_seed + i * 1000, n_seeds)
            
            for j, seed in enumerate(seeds):
                print(f"  Running {model_type} config {i+1}/{len(param_grid)}, seed {j+1}/{n_seeds}")
                
                result = run_single_simulation(
                    model_type, config, seed, main_config, auditor
                )
                all_results.append(result)
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save parameter grid
    grid_data = {
        'parameter_grids': parameter_grids,
        'n_seeds_per_config': n_seeds,
        'random_seed_base': base_seed,
    }
    artifacts.save_json(grid_data, 'grid.json')
    print(f"\nSaved parameter grid to grid.json")
    
    # Save results
    artifacts.save_dataframe(results_df, 'results.parquet')
    print(f"Saved results to results.parquet ({len(results_df)} runs)")
    
    # Compute summary statistics
    summary = compute_summary_statistics(results_df, main_config)
    artifacts.save_json(summary, 'summary.json')
    print(f"Saved summary statistics to summary.json")
    
    # Generate plots
    if main_config.get('generate_plots', True):
        print("\nGenerating plots...")
        
        # Chi plot
        fig_chi = plot_chi_vs_theta(results_df)
        artifacts.save_figure(fig_chi, 'fig_chi.png')
        print("  Saved fig_chi.png")
        
        # S_lab plot
        fig_slab = plot_slab_vs_theta(results_df)
        artifacts.save_figure(fig_slab, 'fig_slab.png')
        print("  Saved fig_slab.png")
    
    # Sensitivity analysis
    print("\nPerforming sensitivity analysis...")
    sensitivity_results = analyze_bulk_vs_boundary_sensitivity(results_df)
    summary['sensitivity_analysis'] = sensitivity_results
    
    # Create audit report
    audit_report = auditor.create_audit_report(all_results)
    artifacts.save_json(audit_report, 'audit.json')
    print(f"Saved audit report to audit.json")
    
    # Save updated summary with sensitivity
    artifacts.save_json(summary, 'summary.json')
    
    print(f"\n{'='*60}")
    print(f"Experiment complete!")
    print(f"Results saved to: {run_dir}")
    print(f"{'='*60}")
    print(f"\nSummary:")
    print(f"  Total runs: {summary['n_runs']}")
    print(f"  S_lab mean: {summary['s_lab_mean']:.6e} ± {summary['s_lab_stderr_mean']:.6e}")
    print(f"  S_lab max: {summary['s_lab_max']:.6e}")
    print(f"  Causality pass rate: {summary['causality_pass_rate']:.1%}")
    
    if 'chi_mean' in summary and summary['chi_mean'] is not None:
        print(f"  Chi mean: {summary['chi_mean']:.6e}")
        print(f"  Chi negative count: {summary.get('chi_n_negative', 0)}")
        print(f"  Validity pass rate: {summary.get('validity_pass_rate', 0):.1%}")


def compute_summary_statistics(results_df: pd.DataFrame, config: Dict[str, Any]) -> Dict[str, Any]:
    """Compute summary statistics from results."""
    summary = {
        'n_runs': len(results_df),
        's_lab_mean': float(results_df['s_lab'].mean()),
        's_lab_std': float(results_df['s_lab'].std()),
        's_lab_max': float(results_df['s_lab'].max()),
        's_lab_stderr_mean': float(results_df['s_lab_stderr'].mean()),
        'causality_pass_rate': float(results_df['causality_passed'].mean()),
        'n_significant_s_lab': int(results_df['s_lab_significant'].sum()),
    }
    
    # Chi statistics (lattice only)
    lattice_results = results_df[results_df['model_type'] == 'lattice']
    if len(lattice_results) > 0 and 'chi' in lattice_results.columns:
        chi_valid = lattice_results['chi'].dropna()
        if len(chi_valid) > 0:
            summary['chi_mean'] = float(chi_valid.mean())
            summary['chi_std'] = float(chi_valid.std())
            summary['chi_min'] = float(chi_valid.min())
            summary['chi_max'] = float(chi_valid.max())
            summary['chi_n_negative'] = int((chi_valid < 0).sum())
        
        # Validity statistics
        if 'validity_passed' in lattice_results.columns:
            validity_valid = lattice_results['validity_passed'].dropna()
            if len(validity_valid) > 0:
                summary['validity_pass_rate'] = float(validity_valid.mean())
    
    return summary


def plot_chi_vs_theta(results_df: pd.DataFrame) -> plt.Figure:
    """Plot chi vs various parameters."""
    lattice_results = results_df[results_df['model_type'] == 'lattice']
    
    if len(lattice_results) == 0:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(0.5, 0.5, 'No lattice results available', 
                ha='center', va='center', fontsize=12)
        return fig
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('χ Effective-Time Loop Diagnostic', fontsize=14, fontweight='bold')
    
    # Plot chi vs flow_velocity
    ax = axes[0, 0]
    if 'flow_velocity' in lattice_results.columns and 'chi' in lattice_results.columns:
        valid_data = lattice_results[lattice_results['chi'].notna()]
        ax.scatter(valid_data['flow_velocity'], valid_data['chi'], alpha=0.6)
        ax.axhline(0, color='r', linestyle='--', label='χ = 0')
        ax.set_xlabel('Flow Velocity Ratio')
        ax.set_ylabel('χ')
        ax.set_title('χ vs Flow Velocity')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot chi vs cutoff_scale
    ax = axes[0, 1]
    if 'cutoff_scale' in lattice_results.columns:
        valid_data = lattice_results[lattice_results['chi'].notna()]
        ax.scatter(valid_data['cutoff_scale'], valid_data['chi'], alpha=0.6)
        ax.axhline(0, color='r', linestyle='--', label='χ = 0')
        ax.set_xlabel('Cutoff Scale')
        ax.set_ylabel('χ')
        ax.set_xscale('log')
        ax.set_title('χ vs Cutoff Scale (Bulk)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot chi vs dispersion_strength
    ax = axes[1, 0]
    if 'dispersion_strength' in lattice_results.columns:
        valid_data = lattice_results[lattice_results['chi'].notna()]
        ax.scatter(valid_data['dispersion_strength'], valid_data['chi'], alpha=0.6)
        ax.axhline(0, color='r', linestyle='--', label='χ = 0')
        ax.set_xlabel('Dispersion Strength (Bulk)')
        ax.set_ylabel('χ')
        ax.set_title('χ vs Dispersion Strength')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot chi vs n_sites
    ax = axes[1, 1]
    if 'n_sites' in lattice_results.columns:
        valid_data = lattice_results[lattice_results['chi'].notna()]
        ax.scatter(valid_data['n_sites'], valid_data['chi'], alpha=0.6)
        ax.axhline(0, color='r', linestyle='--', label='χ = 0')
        ax.set_xlabel('Number of Sites (Boundary)')
        ax.set_ylabel('χ')
        ax.set_title('χ vs System Size')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_slab_vs_theta(results_df: pd.DataFrame) -> plt.Figure:
    """Plot S_lab vs various parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('S_lab Lab-Time Signaling Diagnostic', fontsize=14, fontweight='bold')
    
    # CPTP results
    cptp_results = results_df[results_df['model_type'] == 'cptp']
    
    # Plot S_lab vs environment_coupling (CPTP)
    ax = axes[0, 0]
    if len(cptp_results) > 0 and 'environment_coupling' in cptp_results.columns:
        ax.scatter(cptp_results['environment_coupling'], cptp_results['s_lab'], 
                  alpha=0.6, label='S_lab')
        ax.errorbar(cptp_results['environment_coupling'], cptp_results['s_lab'],
                   yerr=cptp_results['s_lab_stderr'], fmt='none', alpha=0.3)
        ax.axhline(0, color='r', linestyle='--', label='S_lab = 0')
        ax.set_xlabel('Environment Coupling')
        ax.set_ylabel('S_lab')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_title('S_lab vs Environment Coupling (CPTP)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot S_lab vs t1_t2_separation (CPTP)
    ax = axes[0, 1]
    if len(cptp_results) > 0 and 't1_t2_separation' in cptp_results.columns:
        ax.scatter(cptp_results['t1_t2_separation'], cptp_results['s_lab'], alpha=0.6)
        ax.axhline(0, color='r', linestyle='--', label='S_lab = 0')
        ax.set_xlabel('t1-t2 Separation')
        ax.set_ylabel('S_lab')
        ax.set_yscale('log')
        ax.set_title('S_lab vs Time Separation (CPTP)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Lattice results
    lattice_results = results_df[results_df['model_type'] == 'lattice']
    
    # Plot S_lab vs flow_velocity (Lattice)
    ax = axes[1, 0]
    if len(lattice_results) > 0 and 'flow_velocity_ratio' in lattice_results.columns:
        ax.scatter(lattice_results['flow_velocity_ratio'], lattice_results['s_lab'], alpha=0.6)
        ax.axhline(0, color='r', linestyle='--', label='S_lab = 0')
        ax.set_xlabel('Flow Velocity Ratio')
        ax.set_ylabel('S_lab')
        ax.set_yscale('log')
        ax.set_title('S_lab vs Flow Velocity (Lattice)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Plot S_lab distribution
    ax = axes[1, 1]
    ax.hist(results_df['s_lab'], bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(0, color='r', linestyle='--', label='S_lab = 0')
    ax.set_xlabel('S_lab')
    ax.set_ylabel('Frequency')
    ax.set_yscale('log')
    ax.set_title('S_lab Distribution (All Models)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run compiled realities simulation')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to main configuration file')
    
    args = parser.parse_args()
    
    run_experiment(args.config)


if __name__ == '__main__':
    main()
