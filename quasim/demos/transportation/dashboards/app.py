"""Streamlit dashboard for transportation demo."""

import json
from pathlib import Path

import streamlit as st


def main():
    """Main dashboard."""
    st.set_page_config(page_title="Transportation Demo - QuASIM", layout="wide")

    st.title("🚛 Fleet Routing with Stochastic ETA & Charging")
    st.markdown("**Target Accounts**: UPS, FedEx, Tesla, Maersk")

    st.markdown(
        """
    ## Overview

    Minimize lateness + energy cost under depot constraints

    ### Key Performance Indicators (KPIs)
    - on_time_pct, energy_cost, km_traveled, charge_wait_time
    """
    )

    artifacts_dir = Path("artifacts/transportation")

    if artifacts_dir.exists():
        run_dirs = sorted([d for d in artifacts_dir.iterdir() if d.is_dir()])

        if run_dirs:
            latest_run = run_dirs[-1]
            metrics_file = latest_run / "metrics.json"

            if metrics_file.exists():
                with open(metrics_file) as f:
                    metrics = json.load(f)

                st.subheader("Latest Run Results")
                cols = st.columns(4)
                for i, kpi in enumerate(
                    ["on_time_pct", "energy_cost", "km_traveled", "charge_wait_time"]
                ):
                    with cols[i]:
                        st.metric(kpi, f"{metrics.get(kpi, 0):.2f}")

                video_file = latest_run / "simulation.mp4"
                if video_file.exists():
                    st.subheader("Visualization")
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
    # Run planning
    python -m quasim.demos.transportation.cli plan --steps 200 --seed 42

    # Run simulation with capture
    python -m quasim.demos.transportation.cli simulate --seed 42 --capture
    ```
    """
    )


if __name__ == "__main__":
    main()
