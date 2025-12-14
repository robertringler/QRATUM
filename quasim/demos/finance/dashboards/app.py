"""Streamlit dashboard for finance demo."""

import json
from pathlib import Path

import streamlit as st


def main():
    """Main dashboard."""
    st.set_page_config(page_title="Finance Demo - QuASIM", layout="wide")

    st.title("💰 Intraday Risk & Liquidity Stress with Quantum Tensor Net Greeks")
    st.markdown("**Target Accounts**: JPMorgan, Goldman Sachs, BlackRock, Two Sigma")

    st.markdown(
        """
    ## Overview

    Stable VaR/ES estimation and robust liquidity stress under shocks

    ### Key Performance Indicators (KPIs)
    - VaR_99, ES_97_5, drawdown_max, pnl_cvar_gap
    """
    )

    artifacts_dir = Path("artifacts/finance")

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
                for i, kpi in enumerate(["VaR_99", "ES_97_5", "drawdown_max", "pnl_cvar_gap"]):
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
    python -m quasim.demos.finance.cli plan --steps 200 --seed 42

    # Run simulation with capture
    python -m quasim.demos.finance.cli simulate --seed 42 --capture
    ```
    """
    )


if __name__ == "__main__":
    main()
