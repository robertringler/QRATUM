"""Streamlit dashboard for aerospace demo."""

import json
from pathlib import Path

import streamlit as st


def main():
    """Main dashboard."""

    st.set_page_config(page_title="Aerospace Demo - QuASIM", layout="wide")

    st.title("🚀 Aerospace: Hot-Staging & MECO Optimization")
    st.markdown("**Target Accounts**: SpaceX, Boeing, Lockheed Martin, Northrop Grumman")

    st.markdown(
        """

    ## Overview

    This demo showcases trajectory optimization for launch vehicles with hot-staging
    and Main Engine Cutoff (MECO) envelope analysis.

    ### Key Performance Indicators (KPIs)
    - **RMSE Altitude**: Root mean squared error vs target trajectory
    - **RMSE Velocity**: Velocity profile accuracy
    - **q_max**: Maximum dynamic pressure
    - **Fuel Margin**: Remaining fuel percentage
    """
    )

    # Try to load latest results
    artifacts_dir = Path("artifacts/aerospace")

    if artifacts_dir.exists():
        # Find latest run
        run_dirs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir()])

        if run_dirs:
            latest_run = run_dirs[-1]
            metrics_file = latest_run / "metrics.json"

            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)

                st.subheader("Latest Run Results")

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("RMSE Altitude", f"{metrics['rmse_altitude']:.1f} m")

                with col2:
                    st.metric("RMSE Velocity", f"{metrics['rmse_velocity']:.1f} m/s")

                with col3:
                    st.metric("Max Dynamic Pressure", f"{metrics['q_max']:.0f} Pa")

                with col4:
                    st.metric("Fuel Margin", f"{metrics['fuel_margin']:.1f}%")

                # Check for video
                video_file = latest_run / "capture.mp4"
                if video_file.exists():
                    st.subheader("Trajectory Visualization")
                    st.video(str(video_file))
            else:
                st.info("No results available yet. Run the demo to generate data.")
        else:
            st.info("No runs available yet. Run the demo to generate data.")
    else:
        st.info("No artifacts directory found. Run the demo to generate data.")

    st.markdown(
        """

    ## How to Run

    ```bash
    # Optimize trajectory
    python -m quasim.demos.aerospace.cli optimize --steps 200 --profile starship

    # Replay with video capture
    python -m quasim.demos.aerospace.cli replay --scenario hot_staging_v1 --capture

    # Run simulation
    python -m quasim.demos.aerospace.cli simulate --seed 42 --capture
    ```

    ## Compliance Notes

    - DO-178C Level A process compatibility (no certification claims)
    - Deterministic reproducibility with seed control
    - NIST 800-53/171 security controls applicable to production deployments
    """
    )


if __name__ == "__main__":
    main()
