"""Visualization helpers for QuASIM simulation output."""
from __future__ import annotations

import plotly.graph_objects as go


def render_histogram(amplitudes):
    fig = go.Figure()
    fig.add_bar(x=list(range(len(amplitudes))), y=[abs(a) for a in amplitudes])
    fig.update_layout(title="QuASIM Amplitude Magnitudes", xaxis_title="Index", yaxis_title="|Amplitude|")
    return fig
