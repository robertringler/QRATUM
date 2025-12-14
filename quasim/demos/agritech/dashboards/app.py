"""Streamlit dashboard for agritech demo."""

import json
from pathlib import Path

import streamlit as st


def main():
    """Main dashboard."""
    st.set_page_config(page_title="Agritech Demo - QuASIM", layout="wide")

    st.title("🌾 Irrigation & Yield Optimization with Weather Uncertainty")
    st.markdown("**Target Accounts**: John Deere, Bayer Crop Science, Corteva, Syngenta")

    st.markdown(
        """
    ## Overview

    Maximize yield per water input under weather/soil dynamics

    ### Key Performance Indicators (KPIs)
    - yield_kg_ha, water_use_eff, risk_of_loss, profit_margin
    """
    )

    artifacts_dir = Path("artifacts/agritech")

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
                    ["yield_kg_ha", "water_use_eff", "risk_of_loss", "profit_margin"]
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
    python -m quasim.demos.agritech.cli plan --steps 200 --seed 42

    # Run simulation with capture
    python -m quasim.demos.agritech.cli simulate --seed 42 --capture
    ```
    """
    )


if __name__ == "__main__":
    main()
