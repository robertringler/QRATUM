# quasim/cli/run_flow.py
# Tiny CLI: run coupled simulation + print objective

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from quasim.control.optimizer import optimize_a
from quasim.viz.renderer import render_frame, FlowFrameSpec


def _generate_video_hash(seed: int, steps: int, N: int, T: float) -> str:
    """Generate a reproducible hash for the video filename based on parameters."""
    param_str = f"seed={seed}_steps={steps}_N={N}_T={T}"
    return hashlib.sha256(param_str.encode()).hexdigest()[:12]


def _create_video_artifacts(
    a_opt: np.ndarray,
    hist: list,
    logs: dict,
    T: float,
    N: int,
    repro_hash: str,
    num_frames: int = 60,
):
    """
    Generate MP4 and GIF artifacts from simulation results.
    
    Args:
        a_opt: Optimized control schedule
        hist: Optimization history
        logs: Simulation logs containing metrics
        T: Total simulation time
        N: Number of simulation steps
        repro_hash: Reproducible hash for filenames
        num_frames: Number of frames to render (default: 60)
    """
    import imageio
    
    # Create artifacts directory
    artifacts_dir = Path("artifacts/flows")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine which indices to sample for frames (evenly spaced)
    frame_indices = np.linspace(0, N - 1, num_frames, dtype=int)
    
    # Get final objective value
    J_final = hist[-1][0]
    
    # Render frames
    print(f"[quasim] Rendering {num_frames} frames...")
    frames = []
    for i, idx in enumerate(frame_indices):
        time = T * idx / N
        spec = FlowFrameSpec(
            frame_idx=i,
            time=time,
            control=a_opt[idx],
            objective=J_final,  # Use final objective value
            w2=logs["W2"][idx],
            fr_speed=logs["FR_speed"][idx],
            bures_dist=logs["Bures_dist"][idx],
            qfi=logs["QFI_mu"][idx],
            fidelity=logs["Fidelity_step"][idx],
            free_energy=logs["FreeEnergy"][idx],
        )
        frame = render_frame(spec)
        frames.append(frame)
    
    # Save MP4
    mp4_path = artifacts_dir / f"quasim_run_{repro_hash}.mp4"
    print(f"[quasim] Saving MP4 to {mp4_path}...")
    imageio.mimsave(mp4_path, frames, fps=10, codec="libx264", quality=8)
    
    # Save GIF
    gif_path = artifacts_dir / f"quasim_run_{repro_hash}.gif"
    print(f"[quasim] Saving GIF to {gif_path}...")
    imageio.mimsave(gif_path, frames, fps=10, loop=0)
    
    # Create symlinks to latest
    latest_mp4 = artifacts_dir / "quasim_run_latest.mp4"
    latest_gif = artifacts_dir / "quasim_run_latest.gif"
    
    # Remove old symlinks if they exist
    if latest_mp4.exists() or latest_mp4.is_symlink():
        latest_mp4.unlink()
    if latest_gif.exists() or latest_gif.is_symlink():
        latest_gif.unlink()
    
    # Create new symlinks (or copy on Windows)
    try:
        latest_mp4.symlink_to(mp4_path.name)
        latest_gif.symlink_to(gif_path.name)
    except (OSError, NotImplementedError):
        # Fall back to copying on systems that don't support symlinks
        import shutil
        shutil.copy2(mp4_path, latest_mp4)
        shutil.copy2(gif_path, latest_gif)
    
    print(f"[quasim] Video artifacts created:")
    print(f"  MP4: {mp4_path} ({mp4_path.stat().st_size / 1024 / 1024:.2f} MB)")
    print(f"  GIF: {gif_path} ({gif_path.stat().st_size / 1024 / 1024:.2f} MB)")


def main():
    parser = argparse.ArgumentParser(
        description="Run geometry–stats–quantum coupled flow and optimize a(t)."
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--N", type=int, default=300)
    parser.add_argument("--T", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument("--emit-json", action="store_true", help="Emit results as JSON")
    args = parser.parse_args()

    a_opt, hist, logs = optimize_a(
        steps=args.steps,
        N=args.N,
        T=args.T,
        seed=args.seed,
        Omega=2.0,
        lam=0.8,
        gamma=0.25,
        omega=0.9,
        mu0=1.0,
        sigma0=0.8,
        alpha=1.0,
        beta=0.05,
        gamma_loss=0.1,
        delta=0.01,
    )

    J_final = hist[-1][0]
    print(f"[quasim] final objective: {J_final:.6f}")
    print(
        f"[quasim] a(t): mean={a_opt.mean():.3f}, min={a_opt.min():.3f}, max={a_opt.max():.3f}"
    )
    print(f"[quasim] logs: keys={list(logs.keys())}")
    
    # Generate reproducible hash for video filenames
    repro_hash = _generate_video_hash(args.seed, args.steps, args.N, args.T)
    
    # Create video artifacts
    _create_video_artifacts(a_opt, hist, logs, args.T, args.N, repro_hash)
    
    # Emit JSON if requested
    if args.emit_json:
        result = {
            "objective": float(J_final),
            "control_mean": float(a_opt.mean()),
            "control_min": float(a_opt.min()),
            "control_max": float(a_opt.max()),
            "seed": args.seed,
            "steps": args.steps,
            "N": args.N,
            "T": args.T,
            "video_hash": repro_hash,
        }
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()