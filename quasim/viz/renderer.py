# quasim/viz/renderer.py
# Frame rendering for QuASIM flow visualization


import numpy as np

from .specs import FlowFrameSpec


def render_frame(spec: FlowFrameSpec, dpi: int = 100) -> np.ndarray:
    """
    Render a single frame of the QuASIM flow visualization.

    Args:
        spec: Frame specification containing all data to render
        dpi: Dots per inch for the rendered frame

    Returns:
        RGB numpy array of shape (height, width, 3) with values in [0, 255]
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")  # Non-interactive backend

    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=dpi)
    fig.suptitle(
        f"QuASIM Flow — Frame {spec.frame_idx} (t={spec.time:.3f})", fontsize=14, fontweight="bold"
    )

    # Control value
    axes[0, 0].text(
        0.5, 0.5, f"{spec.control:.4f}", ha="center", va="center", fontsize=32, fontweight="bold"
    )
    axes[0, 0].set_title("Control a(t)", fontweight="bold")
    axes[0, 0].set_xlim(0, 1)
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].axis("off")

    # Objective
    axes[0, 1].text(
        0.5,
        0.5,
        f"{spec.objective:.4f}",
        ha="center",
        va="center",
        fontsize=32,
        fontweight="bold",
        color="darkred",
    )
    axes[0, 1].set_title("Objective J", fontweight="bold")
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].axis("off")

    # W2 distance
    axes[0, 2].text(
        0.5,
        0.5,
        f"{spec.w2:.4f}",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="darkblue",
    )
    axes[0, 2].set_title("W2 Distance", fontweight="bold")
    axes[0, 2].set_xlim(0, 1)
    axes[0, 2].set_ylim(0, 1)
    axes[0, 2].axis("off")

    # Fisher-Rao speed
    axes[1, 0].text(
        0.5,
        0.5,
        f"{spec.fr_speed:.4f}",
        ha="center",
        va="center",
        fontsize=28,
        fontweight="bold",
        color="darkgreen",
    )
    axes[1, 0].set_title("FR Speed", fontweight="bold")
    axes[1, 0].set_xlim(0, 1)
    axes[1, 0].set_ylim(0, 1)
    axes[1, 0].axis("off")

    # Bures distance & QFI
    axes[1, 1].text(
        0.5,
        0.6,
        f"{spec.bures_dist:.4f}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="purple",
    )
    axes[1, 1].text(
        0.5, 0.3, f"QFI: {spec.qfi:.4f}", ha="center", va="center", fontsize=18, color="purple"
    )
    axes[1, 1].set_title("Bures Dist", fontweight="bold")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].axis("off")

    # Fidelity & Free Energy
    axes[1, 2].text(
        0.5,
        0.6,
        f"{spec.fidelity:.4f}",
        ha="center",
        va="center",
        fontsize=24,
        fontweight="bold",
        color="darkorange",
    )
    axes[1, 2].text(
        0.5,
        0.3,
        f"FE: {spec.free_energy:.4f}",
        ha="center",
        va="center",
        fontsize=18,
        color="darkorange",
    )
    axes[1, 2].set_title("Fidelity", fontweight="bold")
    axes[1, 2].set_xlim(0, 1)
    axes[1, 2].set_ylim(0, 1)
    axes[1, 2].axis("off")

    plt.tight_layout()

    # Convert figure to numpy array
    fig.canvas.draw()
    buf = fig.canvas.buffer_rgba()
    width, height = fig.canvas.get_width_height()
    frame = np.frombuffer(buf, dtype=np.uint8).reshape(height, width, 4)

    # Convert RGBA to RGB
    frame_rgb = frame[:, :, :3].copy()

    plt.close(fig)

    return frame_rgb
