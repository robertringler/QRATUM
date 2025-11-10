"""Streamlit dashboard for manufacturing demo."""

import json
from pathlib import Path

import streamlit as st


def main():
    """Main dashboard."""
    st.set_page_config(page_title="Manufacturing Demo - QuASIM", layout="wide")

    st.title("🏭 Predictive Maintenance & Throughput Control")
    st.markdown("**Target Accounts**: Siemens, GE, Bosch, Toyota")

    st.markdown(
        """
    ## Overview
    
    Minimize downtime, maximize throughput, schedule maintenance
    
    ### Key Performance Indicators (KPIs)
    - MTBF, downtime_pct, throughput_units_hr, false_alarm_rate
    """
    )

    artifacts_dir = Path("artifacts/manufacturing")

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
                    ["MTBF", "downtime_pct", "throughput_units_hr", "false_alarm_rate"]
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
    python -m quasim.demos.manufacturing.cli plan --steps 200 --seed 42
    
    # Run simulation with capture
    python -m quasim.demos.manufacturing.cli simulate --seed 42 --capture
    ```
    """
    )


if __name__ == "__main__":
    main()
