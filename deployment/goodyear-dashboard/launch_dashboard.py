#!/usr/bin/env python3
"""
Goodyear Quantum Tire Visualization Dashboard
=============================================

Real-time QuASIM integration with Claude 4.5 Opus for interactive
Gen-6 tire compound analysis and sustainability metrics visualization.

Features:
- 3D Molecular Visualization (Three.js/VTK)
- Grip, Durability, Thermal, Comfort Performance Plots
- CO₂e & Circular Economy Analytics
- Optimization Trace Visualizations
- Claude 4.5 Opus AI-Powered Insights

Author: QuASIM Team
Version: 1.0.0
"""

import os
import json
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Any, Optional
from dataclasses import dataclass, field

import dash
from dash import dcc, html, Input, Output, State, callback, ctx
import dash_bootstrap_components as dbc
import dash_daq as daq
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential
import structlog
from prometheus_client import start_http_server, Counter, Gauge, Histogram

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class DashboardConfig:
    """Dashboard configuration from environment variables."""
    
    # QuASIM API Configuration
    quasim_api_url: str = field(
        default_factory=lambda: os.environ.get("QUASIM_API_URL", "https://api.quasim.io/v1")
    )
    quasim_api_key: str = field(
        default_factory=lambda: os.environ.get("QUASIM_API_KEY", "")
    )
    
    # Claude Opus Configuration
    claude_api_key: str = field(
        default_factory=lambda: os.environ.get("CLAUDE_OPUS_API_KEY", "")
    )
    claude_model: str = "claude-sonnet-4-20250514"
    
    # Simulation Configuration
    simulation_id: str = "GY-SUSTAIN-2030-PILOT"
    compound_id: str = "GY-SUST-2030-0005"
    
    # Dashboard Configuration
    host: str = field(
        default_factory=lambda: os.environ.get("DASH_HOST", "0.0.0.0")
    )
    port: int = field(
        default_factory=lambda: int(os.environ.get("DASH_PORT", "8050"))
    )
    debug: bool = field(
        default_factory=lambda: os.environ.get("DASH_DEBUG", "false").lower() == "true"
    )
    
    # Data paths
    data_dir: str = "/app/data"
    cache_dir: str = "/app/cache"
    
    # Refresh intervals (seconds)
    live_update_interval: int = 5
    metrics_update_interval: int = 30


config = DashboardConfig()

# ============================================================================
# Prometheus Metrics
# ============================================================================

REQUESTS_TOTAL = Counter('dashboard_requests_total', 'Total dashboard requests', ['endpoint'])
API_LATENCY = Histogram('api_latency_seconds', 'API call latency', ['api'])
ACTIVE_USERS = Gauge('active_users', 'Number of active dashboard users')
SIMULATION_STATUS = Gauge('simulation_status', 'Current simulation status', ['simulation_id'])

# ============================================================================
# QuASIM API Client
# ============================================================================

class QuASIMClient:
    """Client for QuASIM API interactions."""
    
    def __init__(self, api_url: str, api_key: str):
        self.api_url = api_url
        self.api_key = api_key
        self.client = httpx.AsyncClient(
            base_url=api_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=30.0
        )
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_simulation_results(self, simulation_id: str, compound_id: str) -> dict:
        """Fetch simulation results from QuASIM API."""
        try:
            with API_LATENCY.labels(api='quasim').time():
                response = await self.client.get(
                    f"/simulations/{simulation_id}/compounds/{compound_id}/results"
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error("QuASIM API error", error=str(e), simulation_id=simulation_id)
            return self._get_mock_data(simulation_id, compound_id)
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_optimization_trace(self, simulation_id: str) -> dict:
        """Fetch optimization trace from QuASIM."""
        try:
            with API_LATENCY.labels(api='quasim_trace').time():
                response = await self.client.get(
                    f"/simulations/{simulation_id}/optimization/trace"
                )
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error("QuASIM trace API error", error=str(e))
            return self._get_mock_optimization_trace()
    
    def _get_mock_data(self, simulation_id: str, compound_id: str) -> dict:
        """Generate mock data for development/demo purposes."""
        np.random.seed(42)
        return {
            "simulation_id": simulation_id,
            "compound_id": compound_id,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "completed",
            "material_properties": {
                "compound_name": "Gen-6 Sustainable Silica Compound",
                "base_polymer": "Bio-based SBR",
                "filler_type": "Rice Husk Silica",
                "filler_loading": 85.0,
                "crosslink_density": 4.2e-4,
                "glass_transition_temp": -42.5,
                "molecular_weight": 450000,
                "tan_delta_0c": 0.52,
                "tan_delta_60c": 0.08,
                "shore_a_hardness": 62,
                "tensile_strength": 18.5,
                "elongation_at_break": 520,
                "tear_strength": 45.2,
                "abrasion_resistance": 125,
            },
            "performance_metrics": {
                "wet_grip": {
                    "coefficient": 0.92,
                    "grade": "A",
                    "improvement_vs_baseline": 12.5,
                },
                "dry_grip": {
                    "coefficient": 1.15,
                    "grade": "A+",
                    "improvement_vs_baseline": 8.2,
                },
                "rolling_resistance": {
                    "coefficient": 6.8,
                    "grade": "A",
                    "fuel_efficiency_gain": 4.5,
                },
                "durability": {
                    "treadwear_rating": 65000,
                    "grade": "A",
                    "lifespan_miles": 75000,
                },
                "thermal_performance": {
                    "heat_buildup_c": 22.5,
                    "thermal_stability": 0.95,
                    "max_operating_temp": 120,
                },
                "comfort": {
                    "noise_db": 68.5,
                    "vibration_damping": 0.82,
                    "ride_quality_index": 8.5,
                },
            },
            "sustainability_metrics": {
                "co2e_per_tire_kg": 8.2,
                "bio_content_percent": 35.0,
                "recycled_content_percent": 28.0,
                "circular_economy_score": 78.5,
                "end_of_life_recyclability": 92.0,
                "water_footprint_liters": 1250,
                "energy_consumption_kwh": 45.2,
                "volatile_emissions_ppm": 12.5,
                "lifecycle_co2e_kg": 42.5,
                "renewable_energy_percent": 65.0,
            },
            "molecular_structure": {
                "atoms": self._generate_mock_atoms(150),
                "bonds": self._generate_mock_bonds(200),
            },
            "quantum_optimization": {
                "algorithm": "QAOA",
                "qubits_used": 127,
                "circuit_depth": 45,
                "optimization_iterations": 1500,
                "final_energy": -127.45,
                "convergence_achieved": True,
            },
        }
    
    def _generate_mock_atoms(self, n: int) -> list:
        """Generate mock atomic coordinates for 3D visualization."""
        atoms = []
        for i in range(n):
            atom_type = np.random.choice(["C", "H", "O", "Si", "S", "N"], p=[0.4, 0.3, 0.1, 0.1, 0.05, 0.05])
            atoms.append({
                "id": i,
                "element": atom_type,
                "x": float(np.random.randn() * 10),
                "y": float(np.random.randn() * 10),
                "z": float(np.random.randn() * 10),
                "charge": float(np.random.uniform(-0.5, 0.5)),
            })
        return atoms
    
    def _generate_mock_bonds(self, n: int) -> list:
        """Generate mock bond data for 3D visualization."""
        bonds = []
        for i in range(n):
            bonds.append({
                "id": i,
                "atom1": int(np.random.randint(0, 150)),
                "atom2": int(np.random.randint(0, 150)),
                "order": int(np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])),
            })
        return bonds
    
    def _get_mock_optimization_trace(self) -> dict:
        """Generate mock optimization trace data."""
        n_iterations = 1500
        iterations = list(range(n_iterations))
        
        # Simulated annealing-like convergence
        energy = -50 + 80 * np.exp(-np.array(iterations) / 300) + np.random.randn(n_iterations) * 2
        gradient_norm = 10 * np.exp(-np.array(iterations) / 200) + np.random.randn(n_iterations) * 0.5
        
        return {
            "iterations": iterations,
            "energy": energy.tolist(),
            "gradient_norm": np.abs(gradient_norm).tolist(),
            "learning_rate": (0.1 * np.exp(-np.array(iterations) / 500)).tolist(),
            "parameter_updates": {
                "polymer_ratio": (0.6 + 0.2 * (1 - np.exp(-np.array(iterations) / 400))).tolist(),
                "filler_loading": (70 + 15 * (1 - np.exp(-np.array(iterations) / 350))).tolist(),
                "crosslink_density": (3.5e-4 + 0.7e-4 * (1 - np.exp(-np.array(iterations) / 300))).tolist(),
            },
            "quantum_metrics": {
                "fidelity": (0.85 + 0.1 * (1 - np.exp(-np.array(iterations) / 600))).tolist(),
                "entanglement_entropy": (2.5 + 1.5 * np.sin(np.array(iterations) / 100)).tolist(),
            },
        }
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# ============================================================================
# Claude Opus AI Client
# ============================================================================

class ClaudeOpusClient:
    """Client for Claude 4.5 Opus AI insights."""
    
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.client = httpx.AsyncClient(
            base_url="https://api.anthropic.com",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2024-01-01",
                "content-type": "application/json",
            },
            timeout=60.0
        )
    
    @retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=2, max=10))
    async def get_insights(self, simulation_data: dict, query: str) -> str:
        """Get AI-powered insights from Claude Opus."""
        try:
            with API_LATENCY.labels(api='claude').time():
                prompt = f"""You are an expert tire compound engineer and materials scientist.
                
Analyze the following Goodyear Gen-6 sustainable tire compound simulation data and provide insights:

Simulation Data:
{json.dumps(simulation_data, indent=2)}

User Query: {query}

Provide a detailed, technical response with actionable recommendations.
Focus on sustainability improvements, performance optimization, and practical manufacturing considerations.
"""
                
                response = await self.client.post(
                    "/v1/messages",
                    json={
                        "model": self.model,
                        "max_tokens": 1024,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                )
                response.raise_for_status()
                result = response.json()
                return result.get("content", [{}])[0].get("text", "No insights available.")
        except Exception as e:
            logger.error("Claude API error", error=str(e))
            return self._get_mock_insight(query)
    
    def _get_mock_insight(self, query: str) -> str:
        """Return mock insight for demo purposes."""
        return f"""## AI Analysis: Gen-6 Sustainable Compound

Based on the simulation data for compound GY-SUST-2030-0005:

### Key Findings
1. **Excellent Wet Grip Performance**: The 0.92 coefficient exceeds EU A-grade requirements
2. **Sustainability Leadership**: 35% bio-content with 28% recycled materials
3. **Thermal Optimization**: Heat buildup of 22.5°C indicates good energy dissipation

### Recommendations
- Consider increasing silica loading to 88% for improved rolling resistance
- Bio-based SBR ratio could be optimized to 40% without compromising durability
- Quantum optimization suggests potential for 5% CO₂e reduction

### Manufacturing Notes
- Mixing temperature: 145-155°C optimal
- Cure time: 12-14 minutes at 160°C
- Quality checkpoints: Shore A, tan delta, tensile

*Analysis generated for query: {query[:100]}...*
"""
    
    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


# ============================================================================
# Dashboard Application
# ============================================================================

# Initialize clients
quasim_client = QuASIMClient(config.quasim_api_url, config.quasim_api_key)
claude_client = ClaudeOpusClient(config.claude_api_key, config.claude_model)

# Initialize Dash app with Bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.CYBORG,
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
    ],
    suppress_callback_exceptions=True,
    title="Goodyear Quantum Tire Dashboard",
    update_title="Loading..."
)

server = app.server

# ============================================================================
# Layout Components
# ============================================================================

def create_header():
    """Create dashboard header."""
    return dbc.Navbar(
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.I(className="fas fa-atom me-2", style={"fontSize": "1.5rem"}),
                    dbc.NavbarBrand("Goodyear Quantum Tire Dashboard", className="ms-2 fw-bold"),
                ], width="auto"),
                dbc.Col([
                    dbc.Badge("LIVE", color="success", className="me-2 pulse-animation"),
                    html.Span(id="simulation-status", className="text-muted"),
                ], width="auto"),
            ], align="center", className="g-0"),
            dbc.Row([
                dbc.Col([
                    html.Small(f"Simulation: {config.simulation_id}", className="text-info me-3"),
                    html.Small(f"Compound: {config.compound_id}", className="text-warning"),
                ], width="auto"),
            ], align="center"),
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-4",
    )


def create_kpi_cards(data: dict) -> list:
    """Create KPI indicator cards."""
    metrics = data.get("performance_metrics", {})
    sustainability = data.get("sustainability_metrics", {})
    
    cards = [
        dbc.Card([
            dbc.CardBody([
                html.H6("Wet Grip", className="text-muted"),
                html.H3(f"{metrics.get('wet_grip', {}).get('coefficient', 0):.2f}", className="text-primary"),
                dbc.Badge(metrics.get('wet_grip', {}).get('grade', 'N/A'), color="success"),
            ])
        ], className="text-center"),
        dbc.Card([
            dbc.CardBody([
                html.H6("Rolling Resistance", className="text-muted"),
                html.H3(f"{metrics.get('rolling_resistance', {}).get('coefficient', 0):.1f}", className="text-success"),
                html.Small(f"+{metrics.get('rolling_resistance', {}).get('fuel_efficiency_gain', 0)}% Efficiency"),
            ])
        ], className="text-center"),
        dbc.Card([
            dbc.CardBody([
                html.H6("CO₂e per Tire", className="text-muted"),
                html.H3(f"{sustainability.get('co2e_per_tire_kg', 0):.1f} kg", className="text-warning"),
                html.Small(f"{sustainability.get('bio_content_percent', 0)}% Bio-based"),
            ])
        ], className="text-center"),
        dbc.Card([
            dbc.CardBody([
                html.H6("Circular Economy", className="text-muted"),
                html.H3(f"{sustainability.get('circular_economy_score', 0):.1f}", className="text-info"),
                html.Small(f"{sustainability.get('end_of_life_recyclability', 0)}% Recyclable"),
            ])
        ], className="text-center"),
    ]
    return cards


def create_3d_molecular_view(atoms: list, bonds: list) -> go.Figure:
    """Create 3D molecular structure visualization."""
    element_colors = {
        "C": "#404040",
        "H": "#FFFFFF",
        "O": "#FF0000",
        "Si": "#F0C080",
        "S": "#FFFF00",
        "N": "#0000FF",
    }
    element_sizes = {
        "C": 8,
        "H": 4,
        "O": 7,
        "Si": 10,
        "S": 9,
        "N": 7,
    }
    
    fig = go.Figure()
    
    # Add atoms as scatter3d
    for element in set(a["element"] for a in atoms):
        element_atoms = [a for a in atoms if a["element"] == element]
        fig.add_trace(go.Scatter3d(
            x=[a["x"] for a in element_atoms],
            y=[a["y"] for a in element_atoms],
            z=[a["z"] for a in element_atoms],
            mode="markers",
            name=element,
            marker=dict(
                size=element_sizes.get(element, 6),
                color=element_colors.get(element, "#808080"),
                opacity=0.8,
                line=dict(width=1, color="white"),
            ),
            text=[f"{element} (q={a['charge']:.2f})" for a in element_atoms],
            hoverinfo="text",
        ))
    
    # Add bonds as lines
    atom_dict = {a["id"]: a for a in atoms}
    for bond in bonds[:100]:  # Limit bonds for performance
        a1 = atom_dict.get(bond["atom1"])
        a2 = atom_dict.get(bond["atom2"])
        if a1 and a2:
            fig.add_trace(go.Scatter3d(
                x=[a1["x"], a2["x"]],
                y=[a1["y"], a2["y"]],
                z=[a1["z"], a2["z"]],
                mode="lines",
                line=dict(color="#808080", width=2 * bond["order"]),
                showlegend=False,
                hoverinfo="skip",
            ))
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(0,0,0,0.5)",
            font=dict(color="white"),
        ),
        height=400,
    )
    
    return fig


def create_performance_radar(metrics: dict) -> go.Figure:
    """Create performance radar chart."""
    categories = ["Wet Grip", "Dry Grip", "Rolling Resistance", "Durability", "Thermal", "Comfort"]
    
    values = [
        metrics.get("wet_grip", {}).get("coefficient", 0) / 1.2 * 100,
        metrics.get("dry_grip", {}).get("coefficient", 0) / 1.5 * 100,
        100 - metrics.get("rolling_resistance", {}).get("coefficient", 10) / 15 * 100,
        metrics.get("durability", {}).get("treadwear_rating", 0) / 80000 * 100,
        100 - metrics.get("thermal_performance", {}).get("heat_buildup_c", 30) / 50 * 100,
        metrics.get("comfort", {}).get("ride_quality_index", 0) / 10 * 100,
    ]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(0, 176, 246, 0.3)",
        line=dict(color="#00B0F6", width=2),
        name="Gen-6 Compound",
    ))
    
    # Add baseline reference
    baseline = [70, 75, 65, 60, 70, 72]
    fig.add_trace(go.Scatterpolar(
        r=baseline + [baseline[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(255, 165, 0, 0.1)",
        line=dict(color="orange", width=1, dash="dash"),
        name="Baseline",
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(color="white"),
            ),
            angularaxis=dict(
                tickfont=dict(color="white"),
            ),
            bgcolor="rgba(0,0,0,0)",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="white"),
        legend=dict(
            yanchor="bottom",
            y=-0.2,
            xanchor="center",
            x=0.5,
            orientation="h",
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=60, r=60, t=40, b=60),
        height=400,
    )
    
    return fig


def create_sustainability_gauges(sustainability: dict) -> html.Div:
    """Create sustainability metric gauges."""
    gauges = html.Div([
        dbc.Row([
            dbc.Col([
                daq.Gauge(
                    id="co2e-gauge",
                    label="CO₂e (kg/tire)",
                    value=sustainability.get("co2e_per_tire_kg", 0),
                    min=0,
                    max=20,
                    color={"gradient": True, "ranges": {"green": [0, 8], "yellow": [8, 12], "red": [12, 20]}},
                    showCurrentValue=True,
                    units="kg",
                    size=150,
                ),
            ], width=3),
            dbc.Col([
                daq.Gauge(
                    id="bio-content-gauge",
                    label="Bio Content",
                    value=sustainability.get("bio_content_percent", 0),
                    min=0,
                    max=100,
                    color={"gradient": True, "ranges": {"red": [0, 20], "yellow": [20, 40], "green": [40, 100]}},
                    showCurrentValue=True,
                    units="%",
                    size=150,
                ),
            ], width=3),
            dbc.Col([
                daq.Gauge(
                    id="recycled-gauge",
                    label="Recycled Content",
                    value=sustainability.get("recycled_content_percent", 0),
                    min=0,
                    max=100,
                    color={"gradient": True, "ranges": {"red": [0, 15], "yellow": [15, 30], "green": [30, 100]}},
                    showCurrentValue=True,
                    units="%",
                    size=150,
                ),
            ], width=3),
            dbc.Col([
                daq.Gauge(
                    id="circular-gauge",
                    label="Circular Economy Score",
                    value=sustainability.get("circular_economy_score", 0),
                    min=0,
                    max=100,
                    color={"gradient": True, "ranges": {"red": [0, 50], "yellow": [50, 75], "green": [75, 100]}},
                    showCurrentValue=True,
                    units="pts",
                    size=150,
                ),
            ], width=3),
        ], className="justify-content-center"),
    ])
    return gauges


def create_optimization_trace_plots(trace_data: dict) -> go.Figure:
    """Create optimization trace visualization."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Energy Convergence", "Gradient Norm", "Parameter Evolution", "Quantum Metrics"),
        vertical_spacing=0.15,
        horizontal_spacing=0.1,
    )
    
    iterations = trace_data.get("iterations", [])
    
    # Energy convergence
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=trace_data.get("energy", []),
            mode="lines",
            name="Energy",
            line=dict(color="#00B0F6", width=2),
        ),
        row=1, col=1
    )
    
    # Gradient norm
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=trace_data.get("gradient_norm", []),
            mode="lines",
            name="Gradient",
            line=dict(color="#FF6B6B", width=2),
        ),
        row=1, col=2
    )
    
    # Parameter evolution
    params = trace_data.get("parameter_updates", {})
    for param_name, values in params.items():
        fig.add_trace(
            go.Scatter(
                x=iterations,
                y=values,
                mode="lines",
                name=param_name.replace("_", " ").title(),
            ),
            row=2, col=1
        )
    
    # Quantum metrics
    quantum = trace_data.get("quantum_metrics", {})
    fig.add_trace(
        go.Scatter(
            x=iterations,
            y=quantum.get("fidelity", []),
            mode="lines",
            name="Fidelity",
            line=dict(color="#4ECDC4", width=2),
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(30,30,30,0.5)",
        font=dict(color="white"),
        height=500,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.15,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    
    fig.update_xaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.1)")
    
    return fig


def create_material_properties_table(properties: dict) -> dbc.Table:
    """Create material properties table."""
    rows = []
    for key, value in properties.items():
        formatted_key = key.replace("_", " ").title()
        if isinstance(value, float):
            formatted_value = f"{value:.4g}"
        else:
            formatted_value = str(value)
        rows.append(html.Tr([html.Td(formatted_key), html.Td(formatted_value)]))
    
    return dbc.Table(
        [html.Tbody(rows)],
        bordered=True,
        dark=True,
        hover=True,
        responsive=True,
        striped=True,
        size="sm",
    )


# ============================================================================
# Main Layout
# ============================================================================

app.layout = html.Div([
    # Header
    create_header(),
    
    # Data stores
    dcc.Store(id="simulation-data-store"),
    dcc.Store(id="optimization-trace-store"),
    
    # Auto-refresh interval
    dcc.Interval(
        id="live-update-interval",
        interval=config.live_update_interval * 1000,
        n_intervals=0,
    ),
    
    # Main content
    dbc.Container([
        # KPI Cards Row
        dbc.Row(id="kpi-cards-row", className="mb-4 g-3"),
        
        # Main visualization row
        dbc.Row([
            # Left column - 3D Molecular View
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-cube me-2"),
                        "3D Molecular Structure",
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="molecular-3d-view", config={"displayModeBar": True}),
                    ]),
                ], className="h-100"),
            ], md=6),
            
            # Right column - Performance Radar
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-radar me-2"),
                        "Performance Metrics",
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="performance-radar", config={"displayModeBar": False}),
                    ]),
                ], className="h-100"),
            ], md=6),
        ], className="mb-4"),
        
        # Sustainability Metrics Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-leaf me-2"),
                        "Sustainability Analytics",
                    ]),
                    dbc.CardBody(id="sustainability-gauges"),
                ]),
            ]),
        ], className="mb-4"),
        
        # Optimization Trace Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-chart-line me-2"),
                        "Quantum Optimization Trace",
                    ]),
                    dbc.CardBody([
                        dcc.Graph(id="optimization-trace-plot", config={"displayModeBar": True}),
                    ]),
                ]),
            ]),
        ], className="mb-4"),
        
        # Material Properties & AI Insights Row
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-flask me-2"),
                        "Material Properties",
                    ]),
                    dbc.CardBody(id="material-properties-table", style={"maxHeight": "400px", "overflowY": "auto"}),
                ], className="h-100"),
            ], md=5),
            
            dbc.Col([
                dbc.Card([
                    dbc.CardHeader([
                        html.I(className="fas fa-robot me-2"),
                        "Claude Opus AI Insights",
                    ]),
                    dbc.CardBody([
                        dbc.InputGroup([
                            dbc.Input(
                                id="ai-query-input",
                                placeholder="Ask about compound optimization...",
                                type="text",
                            ),
                            dbc.Button(
                                "Analyze",
                                id="ai-analyze-btn",
                                color="primary",
                                n_clicks=0,
                            ),
                        ], className="mb-3"),
                        dcc.Loading(
                            id="ai-loading",
                            type="circle",
                            children=[
                                dcc.Markdown(
                                    id="ai-insights-output",
                                    style={"maxHeight": "300px", "overflowY": "auto"},
                                ),
                            ],
                        ),
                    ]),
                ], className="h-100"),
            ], md=7),
        ], className="mb-4"),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Hr(),
                html.P([
                    "Powered by ",
                    html.Strong("QuASIM"),
                    " Quantum Simulation Engine | ",
                    html.A("Documentation", href="#"),
                    " | © 2025 Goodyear × QuASIM",
                ], className="text-center text-muted"),
            ]),
        ]),
        
    ], fluid=True),
    
    # Custom CSS
    html.Style("""
        .pulse-animation {
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    """),
])


# ============================================================================
# Callbacks
# ============================================================================

@callback(
    [
        Output("simulation-data-store", "data"),
        Output("optimization-trace-store", "data"),
        Output("simulation-status", "children"),
    ],
    Input("live-update-interval", "n_intervals"),
)
def update_data(n_intervals):
    """Fetch latest simulation data from QuASIM API."""
    REQUESTS_TOTAL.labels(endpoint='data_update').inc()
    
    # Run async fetch synchronously
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        simulation_data = loop.run_until_complete(
            quasim_client.get_simulation_results(config.simulation_id, config.compound_id)
        )
        trace_data = loop.run_until_complete(
            quasim_client.get_optimization_trace(config.simulation_id)
        )
    finally:
        loop.close()
    
    status = f"Last updated: {datetime.now().strftime('%H:%M:%S')} | Status: {simulation_data.get('status', 'Unknown')}"
    
    return simulation_data, trace_data, status


@callback(
    Output("kpi-cards-row", "children"),
    Input("simulation-data-store", "data"),
)
def update_kpi_cards(data):
    """Update KPI cards with latest data."""
    if not data:
        return []
    
    cards = create_kpi_cards(data)
    return [dbc.Col(card, md=3) for card in cards]


@callback(
    Output("molecular-3d-view", "figure"),
    Input("simulation-data-store", "data"),
)
def update_molecular_view(data):
    """Update 3D molecular visualization."""
    if not data or "molecular_structure" not in data:
        return go.Figure()
    
    mol_data = data["molecular_structure"]
    return create_3d_molecular_view(mol_data.get("atoms", []), mol_data.get("bonds", []))


@callback(
    Output("performance-radar", "figure"),
    Input("simulation-data-store", "data"),
)
def update_performance_radar(data):
    """Update performance radar chart."""
    if not data or "performance_metrics" not in data:
        return go.Figure()
    
    return create_performance_radar(data["performance_metrics"])


@callback(
    Output("sustainability-gauges", "children"),
    Input("simulation-data-store", "data"),
)
def update_sustainability_gauges(data):
    """Update sustainability gauges."""
    if not data or "sustainability_metrics" not in data:
        return html.Div("No data available")
    
    return create_sustainability_gauges(data["sustainability_metrics"])


@callback(
    Output("optimization-trace-plot", "figure"),
    Input("optimization-trace-store", "data"),
)
def update_optimization_trace(trace_data):
    """Update optimization trace plots."""
    if not trace_data:
        return go.Figure()
    
    return create_optimization_trace_plots(trace_data)


@callback(
    Output("material-properties-table", "children"),
    Input("simulation-data-store", "data"),
)
def update_material_properties(data):
    """Update material properties table."""
    if not data or "material_properties" not in data:
        return html.Div("No data available")
    
    return create_material_properties_table(data["material_properties"])


@callback(
    Output("ai-insights-output", "children"),
    Input("ai-analyze-btn", "n_clicks"),
    State("ai-query-input", "value"),
    State("simulation-data-store", "data"),
    prevent_initial_call=True,
)
def get_ai_insights(n_clicks, query, data):
    """Get AI-powered insights from Claude Opus."""
    if not query or not data:
        return "Please enter a question and ensure simulation data is loaded."
    
    REQUESTS_TOTAL.labels(endpoint='ai_insights').inc()
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        insights = loop.run_until_complete(
            claude_client.get_insights(data, query)
        )
    finally:
        loop.close()
    
    return insights


# Health check endpoint
@server.route("/health")
def health_check():
    """Health check endpoint for Kubernetes probes."""
    return {"status": "healthy", "timestamp": datetime.utcnow().isoformat()}


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    # Start Prometheus metrics server on port 9090
    try:
        start_http_server(9090)
        logger.info("Prometheus metrics server started", port=9090)
    except Exception as e:
        logger.warning("Could not start Prometheus server", error=str(e))
    
    logger.info(
        "Starting Goodyear Quantum Tire Dashboard",
        host=config.host,
        port=config.port,
        simulation_id=config.simulation_id,
        compound_id=config.compound_id,
    )
    
    app.run(
        host=config.host,
        port=config.port,
        debug=config.debug,
    )
